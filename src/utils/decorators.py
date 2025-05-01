import time
from functools import wraps
from src.setting.settings import setup_logger

logger = setup_logger()


def start_finish_function(func):
    @wraps(func)
    def wrapper(*args, **kwargs):

        cls_name = None
        if args and hasattr(args[0], '__class__'):
            cls_name = args[0].__class__.__name__

        start_time = time.time()

        if cls_name:
            logger.info(f"Старт выполнения метода '{cls_name}.{func.__name__}'")
        else:
            logger.info(f"Старт выполнения функции '{func.__name__}'")

        result = func(*args, **kwargs)

        end_time = time.time()
        elapsed = end_time - start_time

        if cls_name:
            logger.info(f"Завершение выполнения метода '{cls_name}.{func.__name__}'. Время выполнения: {elapsed:.2f} секунд")
        else:
            logger.info(f"Завершение выполнения функции '{func.__name__}'. Время выполнения: {elapsed:.2f} секунд")

        return result

    return wrapper
