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

from pathlib import Path
from qdrant_client import QdrantClient, models
from fastembed import TextEmbedding
from fastembed.common.model_description import PoolingType, ModelSource
from .config import settings


EMBEDDING_MODEL = getattr(settings, "embedding_model", "intfloat/multilingual-e5-small")
VECTOR_NAME = f"fast-{EMBEDDING_MODEL.split('/')[-1].lower()}"
MODEL_MARKER_FILE = Path(settings.qdrant_path) / ".embedding_model"

# Размерности для известных моделей
_DIMS = {
    "intfloat/multilingual-e5-small": 384,
    "intfloat/multilingual-e5-base": 768,
    "intfloat/multilingual-e5-large": 1024,
    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2": 384,
    "sentence-transformers/paraphrase-multilingual-mpnet-base-v2": 768,
}
EMBEDDING_DIM = _DIMS.get(EMBEDDING_MODEL, 384)

TextEmbedding.add_custom_model(
    model=EMBEDDING_MODEL,
    pooling=PoolingType.MEAN,
    normalization=True,
    sources=ModelSource(hf=EMBEDDING_MODEL),
    dim=EMBEDDING_DIM,
    model_file="onnx/model.onnx",
)

def get_model_info():
    """
    Возвращает зарегистрированные в FastEmbed параметры модели.
    """
    # Список включает встроенные и зарегистрированные пользовательские модели.
    models_list = TextEmbedding.list_supported_models()
    model_data = next((m for m in models_list if m['model'] == EMBEDDING_MODEL), None)
    
    if model_data:
        return {
            "model_name": model_data['model'],
            "dim": model_data['dim'],
            # Провайдер формируется как 'fast-' + короткое имя модели
            "provider": f"fast-{model_data['model'].split('/')[-1]}"
        }
    return {"model_name": EMBEDDING_MODEL, "dim": "unknown", "provider": "fastembed"}

_client = None

def get_qdrant_client() -> QdrantClient:
    global _client
    if _client is None:
        _client = QdrantClient(path=settings.qdrant_path)
        _client.set_model(EMBEDDING_MODEL, cache_dir=settings.models_cache)
    return _client


def _read_stored_model() -> str | None:
    try:
        return MODEL_MARKER_FILE.read_text(encoding="utf-8").strip() or None
    except FileNotFoundError:
        return None


def _write_stored_model() -> None:
    MODEL_MARKER_FILE.parent.mkdir(parents=True, exist_ok=True)
    MODEL_MARKER_FILE.write_text(EMBEDDING_MODEL, encoding="utf-8")


def ensure_collection():
    client = get_qdrant_client()
    collections = [c.name for c in client.get_collections().collections]
    stored_model = _read_stored_model()

    # Несовпадение или отсутствие маркера модели приводит к удалению коллекции.
    # Для заполнения новой коллекции потребуется повторная загрузка истории.
    if settings.collection_name in collections and stored_model != EMBEDDING_MODEL:
        client.delete_collection(collection_name=settings.collection_name)
        collections = [c.name for c in client.get_collections().collections]

    if settings.collection_name not in collections:
        client.create_collection(
            collection_name=settings.collection_name,
            vectors_config=client.get_fastembed_vector_params(on_disk=True),
        )

    _write_stored_model()


def close_client():
    client = get_qdrant_client()
    try:
        client.close()
    except Exception:
        pass


def upsert_objects(ids, texts, payloads):
    client = get_qdrant_client()
    points = [
        models.PointStruct(
            id=idx,
            vector={
                VECTOR_NAME: models.Document(
                    text=txt, 
                    model=EMBEDDING_MODEL
                )
            },
            payload=payl
        )
        for idx, txt, payl in zip(ids, texts, payloads)
    ]
    
    client.upsert(
        collection_name=settings.collection_name,
        points=points
    )


def search_similar(query_text: str, limit: int = settings.top_k, metadata_name: str | None = None, extra_filters: dict | None = None):
    client = get_qdrant_client()
    filter_conditions = []

    # Точное совпадение полного имени метаданных, а не только класса объекта.
    if metadata_name:
        filter_conditions.append(
            models.FieldCondition(key="metadata_name", match=models.MatchValue(value=metadata_name))
        )

    # Дополнительные условия соединяются через AND; поиск не ослабляет их автоматически.
    if extra_filters:
        for key, value in extra_filters.items():
            if value:
                filter_conditions.append(
                    models.FieldCondition(key=key, match=models.MatchValue(value=value))
                )

    query_filter = models.Filter(must=filter_conditions) if filter_conditions else None

    response = client.query_points(
        collection_name=settings.collection_name,
        query=models.Document(text=query_text, model=EMBEDDING_MODEL),
        using=VECTOR_NAME,
        query_filter=query_filter,
        limit=limit,
    )
    return response.points