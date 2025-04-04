from loguru import logger

logger.add("logs/logs.log", enqueue=True, level="DEBUG", mode="w")