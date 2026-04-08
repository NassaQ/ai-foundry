import logging
import sys

logger = logging.getLogger("ai-foundry")
logger.setLevel(logging.INFO)

handler = logging.StreamHandler(sys.stdout)
handler.setLevel(logging.INFO)


class LevelFormatter(logging.Formatter):
    """Formatter that adds colored level prefix like uvicorn."""

    LEVEL_COLORS = {
        logging.DEBUG: "\033[36m",
        logging.INFO: "\033[32m",
        logging.WARNING: "\033[33m",
        logging.ERROR: "\033[31m",
        logging.CRITICAL: "\033[1;31m",
    }
    RESET = "\033[0m"

    def format(self, record):
        color = self.LEVEL_COLORS.get(record.levelno, "")
        record.levelprefix = f"{color}{record.levelname}:{self.RESET}    "
        return super().format(record)


handler.setFormatter(LevelFormatter(fmt="%(levelprefix)s %(message)s"))
logger.addHandler(handler)
logger.propagate = False
