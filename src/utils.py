import atexit
import logging
import queue
from pathlib import Path

ROOT = Path(__file__).parent.parent
LOGGING_PATH = ROOT / ".app.log"


def setup_logger():
    file_handler = logging.FileHandler(LOGGING_PATH)
    file_handler.setLevel(logging.DEBUG)
    file_format = logging.Formatter(
        "%(asctime)s, %(name)s %(levelname)s %(message)s %(extras)s"
    )
    file_handler.setFormatter(file_format)

    stdout_handler = logging.StreamHandler()
    stdout_handler.setLevel(logging.WARNING)
    stdout_format = logging.Formatter("%(levelname)s %(message)s")
    stdout_handler.setFormatter(stdout_format)
    log_queue = queue.Queue(-1)
    queue_handler = logging.handlers.QueueHandler(log_queue)
    listener = logging.handlers.QueueListener(
        log_queue, file_handler, stdout_handler, respect_handler_level=True
    )
    listener.start()
    atexit.register(listener.stop)

    logger = logging.getLogger("forecasting_logger")
    logger.setLevel(logging.DEBUG)
    logger.handlers.clear()
    logger.addHandler(queue_handler)
    return logger


class FlexibleLoggerAdapter(logging.LoggerAdapter):
    def process(self, msg, kwargs):
        extras = kwargs.pop("extra", "")
        kwargs["extra"] = {"extras": extras}
        return msg, kwargs


logger = FlexibleLoggerAdapter(setup_logger(), extra={})
