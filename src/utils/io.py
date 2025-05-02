import os
import joblib
import json
from datetime import datetime
from typing import Any

from src.setting.settings import setup_logger
from src.utils.decorators import start_finish_function

logger = setup_logger()


class BasePath:
    def __init__(self, path: str):
        self.path = self.resolve_path(path)
        os.makedirs(os.path.dirname(self.path), exist_ok=True)

    def resolve_path(self, relative_path: str) -> str:
        project_root = self._find_project_root()
        return os.path.join(project_root, relative_path)

    def _find_project_root(self, markers=("pyproject.toml", ".venv", "src")) -> str:
        """Рекурсивный подъём вверх по директориям до нахождения корня проекта по маркерам."""
        current = os.path.abspath(os.getcwd())

        while current != os.path.dirname(current):
            if any(os.path.exists(os.path.join(current, marker)) for marker in markers):
                return current
            current = os.path.dirname(current)

        raise RuntimeError("Корень проекта не найден")


class JoblibModelIO(BasePath):
    def __init__(self, path: str):
        super().__init__(path)

    @start_finish_function
    def save_model(self, model):
        joblib.dump(model, self.path)
        logger.info(f"Модель сохранена в файл: {self.path}")

    @start_finish_function
    def load_model(self):
        logger.info(f"Загрузка модели с параметрами из файла {self.path}")
        return joblib.load(self.path)


class JsonAppendLogger(BasePath):
    _global_version: str | None = None

    def __init__(self, path: str, key_name: str, prefix: str = "xgb"):
        super().__init__(path)

        self.key_name = key_name
        self.prefix = prefix

    @start_finish_function
    def _get_or_generate_version(self) -> str:
        if JsonAppendLogger._global_version is None:
            JsonAppendLogger._global_version = f"{self.prefix}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        return JsonAppendLogger._global_version

    @start_finish_function
    def append(self, data: dict[str, Any]) -> None:
        version = self._get_or_generate_version()

        entry = {
            "version": version,
            self.key_name: data,
            "timestamp": datetime.now().isoformat()
        }

        existing = []
        if os.path.isfile(self.path):
            with open(self.path, "r", encoding='utf-8') as f:
                try:
                    existing = json.load(f)
                except json.JSONDecodeError:
                    pass

        existing.append(entry)

        with open(self.path, "w", encoding="utf-8") as f:
            json.dump(existing, f, indent=4, ensure_ascii=False)


if __name__ == "__main__":
    print(BasePath(path="artifacts/params.json").path)

