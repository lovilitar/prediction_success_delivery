import os
import sys
import logging

from src.utils.logger import CustomFormatter

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


# Настройка логера
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

