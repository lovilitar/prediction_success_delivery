import logging
import inspect
import os
import sys
from datetime import datetime

from sqlalchemy.orm import declarative_base

if 'AIRFLOW_HOME' in os.environ and '/usr/local/lib/python3.9/dist-packages/airflow/example_dags/' not in sys.path:
    sys.path.append('/usr/local/lib/python3.9/dist-packages/airflow/example_dags/')

from stepan_library.configuration import ConfigManager
from stepan_library.sql_wrap_orm_class import BaseDatabaseManager


dir_name = os.path.dirname(__file__)
config_file = os.path.join(dir_name, 'config.yaml')
# Считывание конфига
config_manager = ConfigManager(filepath=config_file, working_space='console')
# Подключение в бд
dbm = BaseDatabaseManager(config_manager)


# Настройки логирования
class CustomFormatter(logging.Formatter):
    def format(self, record):
        # Получаем стек вызовов
        stack = inspect.stack()

        # Пропускаем служебные вызовы логгера
        for frame_info in stack:
            if "logging" not in frame_info.filename and frame_info.function != "format":
                frame = frame_info.frame
                filename = os.path.splitext(os.path.basename(frame_info.filename))[0]
                func_name = frame_info.function
                break

        # Попробуем получить имя класса, если self присутствует
        cls_name = None
        if 'self' in frame.f_locals:
            cls_name = frame.f_locals['self'].__class__.__name__

        # Сборка локации: файл.Класс.метод или файл.функция
        if cls_name:
            location = f"{filename}.{cls_name}.{func_name}"
        else:
            location = f"{filename}.{func_name}"

        # Время
        current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

        return f"[{current_time}] [{record.levelname}] [{location}] {record.getMessage()}"


def setup_logger():
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)  # Общий уровень логгера

    # Консольный хендлер
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.DEBUG)

    # Кастомный форматтер
    formatter = CustomFormatter()
    console_handler.setFormatter(formatter)

    # Очищаем старые хендлеры (если скрипт выполняется несколько раз)
    logger.handlers = []
    logger.addHandler(console_handler)

    return logger

