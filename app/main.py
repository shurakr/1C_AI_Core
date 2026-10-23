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

import socket
from contextlib import asynccontextmanager
from fastapi import FastAPI, APIRouter
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage, BaseMessage
from .models import OneCDocument, HistoryBatch, AnalyzeResponse, ChatSearchRequest, ChatSearchResponse, FoundChatObject
from .qdrant_store import ensure_collection, upsert_objects, search_similar, close_client, get_qdrant_client, get_model_info
from .normalizer import build_search_text, build_payload, extract_candidate_fields
from .rag_pipeline import analyze_document_with_llm
from .indexer import prepare_batch_for_indexing
from typing import List
from .config import settings
from .chat_graph import chat_app
import uuid
import uvicorn

def get_qdrant_id(source_config: str, ref_uuid: str) -> str:
    """Формирует стабильный ID из source_config и UUID объекта 1С."""
    unique_string = f"{source_config}_{ref_uuid}"
    return str(uuid.uuid5(uuid.NAMESPACE_DNS, unique_string))

async def get_changed_objects(objects: List[OneCDocument]):
    """
    Сравнивает хеши входящих объектов с теми, что уже есть в Qdrant.
    Возвращает кортеж: (объекты_для_обновления, результаты_пропущенных)
    """
    client = get_qdrant_client()
    incoming_ids = [get_qdrant_id(doc.source_config, doc.ref_uuid) for doc in objects]
    
    # 1. Запрашиваем только метаданные (payload) существующих точек
    existing_points = client.retrieve(
        collection_name=settings.collection_name,
        ids=incoming_ids,
        with_payload=True,
        with_vectors=False
    )
    
    db_hashes = {str(p.id): p.payload.get("hash") for p in existing_points if p.payload}
    
    to_process = []
    skipped_results = []
    
    for doc in objects:
        qdrant_id = get_qdrant_id(doc.source_config, doc.ref_uuid)
        existing_hash = db_hashes.get(qdrant_id)
        
        if existing_hash and existing_hash == doc.hash:
            # Если данные не изменились, помечаем как "skipped"
            skipped_results.append({
                "ref_uuid": doc.ref_uuid,
                "status": "skipped",
                "source_config": doc.source_config,
                "metadata_name": doc.metadata_name,
                "message": "No changes detected (hash match)"
            })
        else:
            # Объект новый или изменился
            to_process.append(doc)
            
    return to_process, skipped_results


def get_local_ip() -> str:
    """Определяет актуальный IP-адрес компьютера в локальной сети."""
    try:
        # Создаем фиктивное UDP-соединение. Оно не отправляет данные в сеть,
        # но заставляет ОС определить локальный IP, смотрящий "наружу".
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception:
        # Адрес для локального подключения, если определить сетевой интерфейс не удалось.
        return "127.0.0.1"

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Создаёт коллекцию при запуске и освобождает клиент Qdrant при остановке."""

    # Инициализация хранилища и вывод адресов подключения.
    ensure_collection()
    local_ip = get_local_ip()
    print("\n" + "="*75)
    print("🚀 Сервер 1C AI успешно запущен!")
    print(f"📍 Локальный адрес (с этого же ПК): http://127.0.0.1:{settings.port}")
    print(f"🌐 Адрес для 1С (с других ПК):       http://{local_ip}:{settings.port}")
    info = get_model_info()
    print(f"🧠 Embedding model: {info['model_name']} | "
          f"vector: {info['provider']} | "
          f"dim: {info['dim']}")
    print("="*75 + "\n")
        
    yield
    # Локальное хранилище освобождается после завершения обработки запросов.
    close_client()


app = FastAPI(title="1C AI MDM", lifespan=lifespan)
core_router = APIRouter(prefix="/api", tags=["Core API"])


@core_router.get("/health")
async def health() -> dict:
    qdrant_status = "error"
    llm_status = "not_ready"
    
    try:
        # Проверяем доступность локального хранилища.
        client = get_qdrant_client()
        client.get_collections()
        qdrant_status = "connected"
    except Exception as e:
        print(f"Health check error: {e}")
        qdrant_status = "disconnected"

    # Наличие ключа не проверяет доступность провайдера и корректность авторизации.
    if settings.openrouter_api_key:
        llm_status = "ready"
    
    # Общий статус системы
    is_ok = qdrant_status == "connected" and llm_status == "ready"
    
    return {
        "status": "ok" if is_ok else "warning",
        "components": {
            "vector_store": qdrant_status,
            "llm_gateway": llm_status
        },
        "version": "1.0.0" 
    }


@core_router.post("/upload_history")
async def upload_history(batch: HistoryBatch) -> dict:
    total_count = len(batch.objects)
    print(f"\n[HISTORY] Получен пакет на выгрузку: {total_count} объектов.")

    to_process, skipped_results = await get_changed_objects(batch.objects)
    
    num_to_process = len(to_process)
    print(f"[FILTER] К индексации: {num_to_process}, Пропущено (хеш совпал): {len(skipped_results)}")  

    if not to_process:
        return {
            "status": "batch_completed",
            "total_count": total_count,
            "success_count": total_count,
            "results": skipped_results,
            "model_version": f"FastEmbed ({settings.embedding_model})"
        }

    # Нормализация и оценка качества текста; пропуск по хешу выполнен выше.
    ids, texts, payloads, object_results = prepare_batch_for_indexing(to_process, get_qdrant_id)
    
    # Объединяем результаты
    all_results = skipped_results + object_results

    if ids:
        print(f"[QDRANT] Начинаю индексацию {len(ids)} векторов...")
        try:
            upsert_objects(ids, texts, payloads)
            for res in all_results:
                if res.get("status") == "prepared":
                    res["status"] = "success"
                    res["message"] = "Vectored and saved"
        except Exception as e:
            error_bulk = f"Bulk upsert error: {str(e)}"
            print(f"[CRITICAL] Ошибка Qdrant: {error_bulk}")       
            for res in all_results:
                if res.get("status") == "prepared":
                    res["status"] = "error"
                    res["message"] = error_bulk

    success_num = len([r for r in all_results if r["status"] in ["success", "skipped"]])
    print(f"[FINISH] Обработка завершена: {success_num} успешно, {total_count - success_num} с ошибками.\n")    

    return {
        "status": "batch_completed",
        "total_count": total_count,
        "success_count": success_num,
        "results": all_results,
        "model_version": f"FastEmbed ({settings.embedding_model})"
    }


@core_router.post("/analyze_document", response_model=AnalyzeResponse)
async def analyze_document(doc: OneCDocument) -> AnalyzeResponse:
    """Ищет аналог объекта и возвращает результат выбора со сравнением реквизитов."""
    norm = extract_candidate_fields(doc)
    query_text = build_search_text(doc)


    hits = search_similar(
        query_text=query_text,
        limit= settings.top_k, 
        metadata_name=norm["metadata_name"],
        extra_filters=norm.get("strict_filters", {})
    )    

    result = await analyze_document_with_llm(doc, hits)
    return AnalyzeResponse(**result)

chat_router = APIRouter(prefix="/api/v1/chat", tags=["AI Chat"])

@chat_router.post("/search", response_model=ChatSearchResponse)
async def chat_search_endpoint(request: ChatSearchRequest):
    # 1. Системный промпт
    messages: List[BaseMessage] = [
        SystemMessage(content="""Ты ИИ-ассистент по поиску корпоративных данных 1С. 
Твоя задача — помочь пользователю найти объекты (справочники, документы), используя предоставленные инструменты.
Никогда не выдумывай UUID. Формулируй ответ на русском языке кратко и вежливо.""")
    ]
    
    # История диалога передаётся клиентом; отдельного хранения сессий здесь нет.
    for msg in (request.history or []):
        role = msg.role if hasattr(msg, "role") else msg.get("role", "")
        content = msg.content if hasattr(msg, "content") else msg.get("content", "")
        if role == "user":
            messages.append(HumanMessage(content=content))
        elif role == "assistant":
            messages.append(AIMessage(content=content))
            
    # 3. Текущий запрос
    messages.append(HumanMessage(content=request.query))
    
    # 4. Запуск агента LangGraph
    state = await chat_app.ainvoke({"messages": messages, "found_objects": []})
    
    final_text = state["messages"][-1].content
    objects_raw = state.get("found_objects", [])
    
    # 5. Форматирование ответа (обогащенный набор данных)
    found_objects = [FoundChatObject(**obj) for obj in objects_raw]
    
    return ChatSearchResponse(
        answer=final_text,
        found_objects=found_objects
    )

app.include_router(core_router)
app.include_router(chat_router)

if __name__ == "__main__":
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)
    