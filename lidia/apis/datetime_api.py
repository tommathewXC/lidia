"""
API for obtaining the current date and time.
"""
from datetime import datetime
from logging import getLogger

logger = getLogger(__name__)

def get_current_datetime():
    """Gets the current date and time."""
    now = datetime.now()
    logger.info("Current date and time: %s", now)
    return now.strftime("%Y-%m-%d %H:%M:%S")