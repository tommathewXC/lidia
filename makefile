.PHONY: clean

clean:
	@echo "Cleaning all __pycache__ directories..."
	@find . -type d -name "__pycache__" -exec rm -rf {} +
	@echo "Clean complete."
