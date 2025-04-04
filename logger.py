from loguru import logger

logger.remove()
logger.add("logs/logs.log", enqueue=True, level="DEBUG", mode="w")