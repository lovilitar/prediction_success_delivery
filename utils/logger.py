import logging
import inspect
import os
from datetime import datetime


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