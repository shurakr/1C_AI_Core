# Copyright 2026 by Alexander Krokhmal' (shurakr@gmail.com)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
from qdrant_client import QdrantClient
import pandas as pd
import json

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(CURRENT_DIR)
ENV_QDRANT_PATH: str = os.getenv("QDRANT_PATH", r"./storage/qdrant")
QDRANT_PATH = os.path.normpath(os.path.join(BASE_DIR, ENV_QDRANT_PATH))
COLLECTION_NAME: str = os.getenv("QDRANT_COLLECTION", "MDM_LLM_1C")
EXPORT_FILE = os.path.join(BASE_DIR, "qdrant_export.xlsx")

def export_to_excel():
    """Экспортирует payload без векторов; сервис должен освободить локальное хранилище Qdrant."""
    print(f"Подключаемся к локальной базе: {QDRANT_PATH}...")
    try:
        client = QdrantClient(path=QDRANT_PATH)
        
        collections = [c.name for c in client.get_collections().collections]
        if COLLECTION_NAME not in collections:
            print(f"Ошибка: Коллекция '{COLLECTION_NAME}' не найдена!")
            print(f"Доступные коллекции: {collections}")
            return

        all_data = []
        offset = None
        total_fetched = 0

        print(f"Начинаем выгрузку из '{COLLECTION_NAME}'...")

        while True:
            records, next_offset = client.scroll(
                collection_name=COLLECTION_NAME,
                limit=100, # Берем по 100 штук за раз
                offset=offset,
                with_payload=True,
                with_vectors=False # Векторы (числа) нам в Excel не нужны
            )

            for record in records:
                row = {
                    "Системный_ID (Qdrant)": str(record.id)
                }

                if record.payload:
                    for key, value in record.payload.items():
                        if isinstance(value, (dict, list)):
                            row[f"Данные_{key}"] = json.dumps(value, ensure_ascii=False)
                        else:
                            row[f"Данные_{key}"] = value

                all_data.append(row)
            
            total_fetched += len(records)
            print(f"Выгружено записей: {total_fetched}...")

            offset = next_offset
            if offset is None:
                break 


        print("Закрываем соединение с базой...")
        client.close()
        
        if not all_data:
            print("База пуста! Выгружать нечего.")
            return


        print("Формируем Excel файл...")
        df = pd.DataFrame(all_data)
        
        cols = df.columns.tolist()
        priority_cols = ["Системный_ID (Qdrant)", "Данные_ref_uuid", "Данные_target_uuid", "Данные_target_name", "Данные_metadata_name", "Данные_hash"]
        ordered_cols = [c for c in priority_cols if c in cols] + [c for c in cols if c not in priority_cols]
        df = df[ordered_cols]

        df.to_excel(EXPORT_FILE, index=False)
        print(f"✅ Успех! Вся база ({total_fetched} объектов) сохранена в файл: {EXPORT_FILE}")

    except Exception as e:
        print(f"❌ Произошла ошибка: {e}")

if __name__ == "__main__":
    export_to_excel()