import logging
from config import parentConfig as config
import os
# Create a custom logger
logger = logging.getLogger("my_logger")
logger.setLevel(logging.DEBUG)  # Set the minimum logging level

file_path = os.path.join(config.logs_dir + "logfile.txt")
console_handler = logging.StreamHandler()
file_handler = logging.FileHandler("logfile.txt")

console_handler.setLevel(logging.DEBUG)
file_handler.setLevel(logging.DEBUG)

formatter = logging.Formatter(
    '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)

logger.debug("This is a debug message")
logger.info("This is an info message")
logger.warning("This is a warning")
