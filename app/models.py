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

from pydantic import BaseModel, Field
from typing import Any, Dict, List, Literal, Optional
from qdrant_client.http.models import ScoredPoint

class EnhancedScoredPoint(ScoredPoint):
    composite_score: float = 0.0
    custom_metrics: Dict[str, Any] = {}

class OneCDocument(BaseModel):
    # source_config содержит имя конфигурации; UUID относится к объекту источника.
    source_config: str
    metadata_name: str
    ref_uuid: str
    
    # Системные поля (Состояние объекта)
    hash: Optional[str] = None
    number: Optional[str] = None 
    date: Optional[str] = None   
    posted: Optional[bool] = None      # Для документов (Проведен/Нет)
    is_folder: bool = False            # Для справочников (Папка/Элемент)
    hierarchy: Optional[str] = "" # Путь (Папка1 / Папка2 / ...)
    
    # Целевые поля (Если это справочник - 1С заполнит их ссылкой на самого себя)
    target_name: Optional[str] = None
    target_uuid: Optional[str] = None

    # Контейнеры данных (Любые реквизиты и ТЧ)
    normalized: Optional[Dict[str, Any]] = None
    raw_fields: Dict[str, Any] = Field(default_factory=dict)
    table_parts: Dict[str, List[Dict[str, Any]]] = Field(default_factory=dict)


class HistoryBatch(BaseModel):
    objects: List[OneCDocument]

class FieldComparison(BaseModel):
    field: str
    label: str
    source_value: Any = None
    candidate_value: Any = None
    status: Literal["match", "different", "missing_source", "missing_candidate", "uncomparable"]


class AnalyzeResponse(BaseModel):
    # Что нашли в истории (ссылка на эталон)
    suggested_target_uuid: Optional[str] = None
    suggested_target_name: Optional[str] = None
    
    # Служебная информация
    metadata_name: str =""
    confidence: int = 0
    reason: str = ""
    result: Optional[str] = None 
    model_version: str = ""
    source_config: str = ""
    similar_docs: List[Dict[str, Any]] = Field(default_factory=list)
    comparison: List[FieldComparison] = Field(default_factory=list)
    comparison_summary: str = ""

class ChatMessage(BaseModel):
    role: str
    content: str

class ChatSearchRequest(BaseModel):
    query: str = Field(..., description="Запрос пользователя в свободной форме")
    history: Optional[List[ChatMessage]] = Field(default_factory=list, description="История диалога")
    session_id: Optional[str] = Field(None, description="ID сессии для логов")

class FoundChatObject(BaseModel):
    target_name: str
    target_uuid: str
    metadata_name: str
    confidence_score: float
    description: Optional[str] = ""
    attributes: Optional[str] = ""

class ChatSearchResponse(BaseModel):
    answer: str = Field(..., description="Ответ агента")
    found_objects: List[FoundChatObject] = Field(default_factory=list, description="Массив найденных объектов")
