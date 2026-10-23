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

from pydantic import BaseModel
from dotenv import load_dotenv
import os

load_dotenv()

class Settings(BaseModel):
    openrouter_api_key: str = os.getenv("OPENROUTER_API_KEY", "")
    openrouter_model: str = os.getenv("OPENROUTER_MODEL", "google/gemini-2.5-flash")
    base_url: str = os.getenv("BASE_URL", "https://openrouter.ai/api/v1")
    qdrant_path: str = os.getenv("QDRANT_PATH", r"./storage/qdrant")
    models_cache: str = os.getenv("MODELS_CACHE", r"./.models_cache")
    collection_name: str = os.getenv("QDRANT_COLLECTION", "MDM_LLM_1C")
    top_k: int = int(os.getenv("TOP_K", "5"))
    host: str = os.getenv("HOST", "127.0.0.1")
    port: int = int(os.getenv("PORT", "8000"))
    embedding_model: str = os.getenv("EMBEDDING_MODEL", "intfloat/multilingual-e5-base")

    STRICT_KEYS: list = [
    "инн", 
    "кпп", 
    "артикул", 
    "код",
    "штрихкод",         
    "огрн",             
    "бик",              
    "снилс"             
]

settings = Settings()
