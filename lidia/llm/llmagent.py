from langchain_community.chat_models import ChatOllama, ChatOpenAI
from langchain_community.llms import HuggingFacePipeline
from langchain.agents import AgentExecutor
from langchain.agents.structured_chat.base import StructuredChatAgent
from langchain.schema import HumanMessage, AIMessage, SystemMessage
from langchain.tools import Tool
from langchain.prompts import MessagesPlaceholder
from langchain.memory import ConversationBufferMemory
from langchain.chains import LLMChain
from lidia.apis.screenshot_api import ScreenshotAPI
from lidia.apis.datetime_api import get_current_datetime
from lidia.config.config import global_config
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
from logging import getLogger
from os.path import expanduser
from typing import List, Dict, Any
from langchain.tools import StructuredTool
from pydantic import BaseModel, Field

logger = getLogger(__name__)

class LLMAgent:
    def __init__(self, mode, model_path=global_config["llm"], ollama_model=global_config["ollama_model"]):
        self.mode = mode.lower()
        self.ollama_model = ollama_model
        
        # Initialize the appropriate LLM
        if self.mode == "local":
            tokenizer = AutoTokenizer.from_pretrained(model_path)
            model = AutoModelForCausalLM.from_pretrained(model_path)
            pipe = pipeline(
                "text-generation",
                model=model,
                tokenizer=tokenizer,
                max_new_tokens=512,
                temperature=0.7,
                top_p=0.95,
                repetition_penalty=1.15
            )
            self.llm = HuggingFacePipeline(pipeline=pipe)
            logger.info("Local LLM model loaded with LangChain integration.")
            
        elif self.mode == "ollama":
            self.llm = ChatOllama(model=ollama_model)
            logger.info("Ollama LLM initialized with LangChain.")
            
        elif self.mode == "openai":
            api_key_path = expanduser(global_config["openai"]["api_key_path"])
            with open(api_key_path, 'r') as f:
                api_key = f.read().strip()
            self.llm = ChatOpenAI(
                api_key=api_key,
                model="gpt-4",  # Use GPT-4 for OpenAI path
                temperature=global_config["openai"]["temperature"],
                streaming=True
            )
            logger.info("OpenAI GPT-4 initialized with LangChain.")
        else:
            raise ValueError("Invalid mode. Use 'local', 'ollama', or 'openai'.")

        # Initialize tools and agent
        self.tools = self._setup_tools()
        self.memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
        self.agent_executor = self._setup_agent()

    def _setup_tools(self) -> List[Tool]:
        """Set up LangChain tools for API interactions."""
        
        class EmptyInput(BaseModel):
            """Empty input schema for tools that don't require input."""
            pass
            
        class ScreenshotInput(BaseModel):
            """Input schema for screenshot tool."""
            monitor_index: int = Field(default=None, description="Optional monitor index to capture")
        
        # Initialize the screenshot API
        screenshot = ScreenshotAPI()
        
        tools = [
            StructuredTool(
                name="get_current_datetime",
                func=lambda _=None: get_current_datetime(),
                description="Gets the current date and time. Use this when asked about current time or date.",
                args_schema=EmptyInput,
                return_direct=True
            ),
            StructuredTool(
                name="take_screenshot",
                func=lambda params=None: screenshot.execute(params),
                description="Takes a screenshot of the desktop and analyzes it using OCR. Use this when asked to look at or read the screen.",
                args_schema=ScreenshotInput,
                return_direct=False
            )
        ]
        
        return tools

    def _setup_agent(self) -> AgentExecutor:
        name = global_config['voice_settings']['name']
        system_message = SystemMessage(content=f"Your name is {name}" + """, you are an AI assistant that can perceive the environment through vision and interact through speech.
        When a user asks about time or date, ALWAYS use the get_current_datetime tool.
        When asked to look at or analyze the screen, ALWAYS use the take_screenshot tool.
        
        Tools available: {tools}
        
        Format your responses in a natural, conversational way.""")

        prompt = [
            system_message,
            MessagesPlaceholder(variable_name="chat_history"),
            HumanMessage(content="{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad"),
        ]

        # For o1-mini, convert system message to user message
        if self.mode == "openai" and global_config["openai"]["model"] == "o1-mini":
            # Convert system message to user message
            system_content = system_message.content
            system_message = HumanMessage(content=f"Instructions: {system_content}")
            
        agent = StructuredChatAgent.from_llm_and_tools(
            llm=self.llm,
            tools=self.tools,
            system_message=system_message
        )

        return AgentExecutor(
            agent=agent,
            tools=self.tools,
            memory=self.memory,
            verbose=True,
            handle_parsing_errors=True,
            max_iterations=2
        )

    def _convert_messages_to_langchain(self, messages: List[Dict[str, str]]) -> List[Any]:
        """Convert message history to LangChain format."""
        langchain_messages = []
        
        for msg in messages:
            content = msg["content"]
            if msg["role"] == "user":
                langchain_messages.append(HumanMessage(content=content))
            elif msg["role"] == "assistant":
                langchain_messages.append(AIMessage(content=content))
            elif msg["role"] == "system" or msg["role"] == "visual":
                # For o1-mini, convert system/visual messages to user messages
                if self.mode == "openai" and global_config["openai"]["model"] == "o1-mini":
                    prefix = "Instructions: " if msg["role"] == "system" else "Visual Context: "
                    langchain_messages.append(HumanMessage(content=f"{prefix}{content}"))
                else:
                    content_with_prefix = f"Visual Context: {content}" if msg["role"] == "visual" else content
                    langchain_messages.append(SystemMessage(content=content_with_prefix))
        
        return langchain_messages

    def stream_response(self, messages: List[Dict[str, str]], meta_context: str = "", max_length: int = 50):
        """Stream the response word by word."""
        # Convert message history to LangChain format
        chat_history = self._convert_messages_to_langchain(messages)
        
        # Get the last user message
        last_user_message = next((msg["content"] for msg in reversed(messages) 
                                if msg["role"] == "user"), "")
        
        try:
            # Run the agent
            result = self.agent_executor.invoke({
                "input": last_user_message,
                "chat_history": chat_history
            })
            
            # Extract the output from the result
            response = result.get("output", "")
            if not response:
                response = "I apologize, but I wasn't able to generate a proper response."
            
            # Stream the response word by word
            for word in response.split():
                yield word
                
        except Exception as e:
            logger.error("Error in agent execution: %s", str(e))
            error_msg = f"I encountered an error: {str(e)}"
            for word in error_msg.split():
                yield word

    def get_processed_response(self, messages: List[Dict[str, str]], meta_context: str = "", max_length: int = 50) -> str:
        """Get the full processed response."""
        filtered_messages = [msg for msg in messages 
                           if msg["role"] != "system" or not msg["content"].startswith("API (")]
        response_words = list(self.stream_response(filtered_messages, meta_context, max_length))
        return " ".join(response_words)