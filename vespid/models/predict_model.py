import logging
from vespid import DATETIME_FORMAT, LOG_FORMAT

logging.basicConfig(
    format=LOG_FORMAT,
    level=logging.INFO,
    datefmt=DATETIME_FORMAT)

logger = logging.getLogger(__name__)

if __name__ == '__main__':
    logger.info("it works")