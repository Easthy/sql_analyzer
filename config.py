# config.py
import os
from pathlib import Path

# Корень проекта (предполагаем, что config.py находится в корне или рядом)
# BASE_DIR = Path(__file__).resolve().parent
# Если скрипт будет запускаться из другого места, лучше задать абсолютный путь
# или путь относительно места запуска. Давайте сделаем его настраиваемым.
# Укажите АБСОЛЮТНЫЙ путь к папке с вашими SQL моделями
# Пример: SQL_MODELS_DIR = Path("/path/to/your/sql_models")
# Или относительный путь от места запуска скрипта:
SQL_MODELS_DIR = Path("./sql_models") # Пример относительного пути

# Файл для сохранения состояния графа зависимостей
STATE_FILE = Path("./dependency_state.json")

# Диалект SQL (важно для корректного парсинга)
# Примеры: "postgres", "mysql", "snowflake", "bigquery", "clickhouse"
SQL_DIALECT = "redshift"

# Расширение файлов с моделями
SQL_FILE_EXTENSION = ".sql"

# Настройки логирования
LOG_LEVEL = "INFO"
LOG_FORMAT = '%(asctime)s - %(levelname)s - %(message)s'

# --- Проверка существования папки ---
# Раскомментируйте, если хотите проверку при импорте конфига
# if not SQL_MODELS_DIR.is_dir():
#     raise FileNotFoundError(
#         f"Папка с SQL моделями не найдена: {SQL_MODELS_DIR}. "
#         f"Проверьте путь в config.py"
#     )