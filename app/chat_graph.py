import json
from typing import Annotated, List, Sequence, TypedDict
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain_core.tools import tool
from pydantic import SecretStr
from qdrant_client import models
from .qdrant_store import get_qdrant_client

from .qdrant_store import search_similar
from .normalizer import canonicalize_name
from .config import settings

# 1. Схема состояния графа (State)
class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]
    found_objects: List[dict] # Найденные объекты для структурированной части ответа API.

# Инструменты читают выгруженные объекты из Qdrant, а не из живой базы 1С.
@tool
def search_1c_objects(query: str, metadata_name: str = None) -> str:
    """
    Выполняет семантический поиск выгруженных объектов 1С в индексе Qdrant.
    :param query: Бизнес-суть для поиска (например, ИНН, название контрагента, суть договора).
    :param metadata_name: (Опционально) Точный тип метаданных 1С, например 'Справочник.ДоговорыКонтрагентов'.
    """
    # Мягкая канонизация сырого текста пользователя
    cleaned_query = canonicalize_name(query, metadata_name or "")
    
    # Поиск не изменяет записи индекса.
    hits = search_similar(
        query_text=cleaned_query,
        limit=5,
        metadata_name=metadata_name
    )
    
    if not hits:
        return json.dumps({"status": "not_found", "message": "Объекты не найдены"})
        
    results = []
    for hit in hits:
        p = hit.payload or {}
        norm = p.get("normalized", {})
        results.append({
            "target_name": p.get("target_name") or norm.get("target_name", "Неизвестно"),
            "target_uuid": p.get("target_uuid") or norm.get("target_uuid", ""),
            "metadata_name": p.get("metadata_name", ""),
            "confidence_score": float(hit.score),
            "description": norm.get("description", "")
        })
    
    return json.dumps(results, ensure_ascii=False)

@tool
def get_all_objects_by_type(metadata_name: str, limit: int = 50) -> str:
    """
    Выводит до limit выгруженных объектов заданного типа из Qdrant без семантического поиска.
    Используй, когда пользователь просит "показать все организации" или "вывести список договоров".
    :param metadata_name: Точный тип метаданных 1С (например, 'Справочник.Организации').
    :param limit: Количество возвращаемых записей (по умолчанию 50, чтобы не перегрузить ответ).
    """
    client = get_qdrant_client()
    scroll_filter = models.Filter(
        must=[models.FieldCondition(key="metadata_name", match=models.MatchValue(value=metadata_name))]
    )
    
    try:
        # scroll отдает элементы без участия векторов (как обычная БД)
        records, next_page = client.scroll(
            collection_name=settings.collection_name,
            scroll_filter=scroll_filter,
            limit=limit,
            with_payload=True,
            with_vectors=False
        )
        
        if not records:
            return json.dumps({"status": "not_found", "message": f"Объекты типа {metadata_name} не найдены"}, ensure_ascii=False)
            
        results = []
        for record in records:
            p = record.payload or {}
            norm = p.get("normalized", {})
            results.append({
                "target_name": p.get("target_name") or norm.get("target_name", "Неизвестно"),
                "target_uuid": p.get("target_uuid") or norm.get("target_uuid", ""),
                "metadata_name": p.get("metadata_name", ""),
                "confidence_score": 1.0, # Условное значение для выборки по типу, не оценка семантического сходства.
                "description": norm.get("description", ""),
                "attributes": norm.get("data_points", "")
            })
        
        return json.dumps({
            "items": results,
            "total_returned": len(results),
            "has_more": next_page is not None
        }, ensure_ascii=False)
        
    except Exception as e:
        return json.dumps({"error": str(e)}, ensure_ascii=False)

@tool
def count_1c_objects(metadata_name: str) -> str:
    """
    Возвращает количество загруженных в Qdrant записей заданного типа; это не полный подсчёт в живой базе 1С.
    Используй, когда пользователь спрашивает "сколько у нас организаций" или "какое количество договоров".
    :param metadata_name: Точный тип метаданных 1С (например, 'Справочник.Организации').
    """
    client = get_qdrant_client()
    count_filter = models.Filter(
        must=[models.FieldCondition(key="metadata_name", match=models.MatchValue(value=metadata_name))]
    )
    try:
        # Подсчёт выполняется в Qdrant без выгрузки payload объектов.
        result = client.count(
            collection_name=settings.collection_name,
            count_filter=count_filter
        )
        return json.dumps({"count": result.count, "metadata_name": metadata_name}, ensure_ascii=False)
    except Exception as e:
        return json.dumps({"error": str(e)}, ensure_ascii=False)

# Провайдер и модель LLM задаются в настройках.
llm = ChatOpenAI(
    model=settings.openrouter_model,
    api_key=SecretStr(settings.openrouter_api_key),
    base_url=settings.base_url,
    temperature=0.1
)
llm_with_tools = llm.bind_tools([search_1c_objects, get_all_objects_by_type, count_1c_objects])

# 4. Узлы графа (Nodes)
async def agent_node(state: AgentState):
    response = await llm_with_tools.ainvoke(state["messages"])
    return {"messages": [response]}

async def tool_node(state: AgentState):
    messages = state["messages"]
    last_message = messages[-1]
    
    tool_responses = []
    current_found = state.get("found_objects", [])
    
    if hasattr(last_message, "tool_calls"):
        for tool_call in last_message.tool_calls:
            # 1. Семантический поиск
            if tool_call["name"] == "search_1c_objects":
                result_str = search_1c_objects.invoke(tool_call["args"])
                try:
                    parsed = json.loads(result_str)
                    if isinstance(parsed, list):
                        current_found.extend(parsed)
                except json.JSONDecodeError:
                    pass
                tool_responses.append(ToolMessage(content=result_str, tool_call_id=tool_call["id"]))
            
            # Выборка ограниченного списка объектов по типу.
            elif tool_call["name"] == "get_all_objects_by_type":
                result_str = get_all_objects_by_type.invoke(tool_call["args"])
                try:
                    parsed = json.loads(result_str)
                    if "items" in parsed:
                        current_found.extend(parsed["items"]) # Объекты включаются в структурированный ответ API.
                except json.JSONDecodeError:
                    pass
                tool_responses.append(ToolMessage(content=result_str, tool_call_id=tool_call["id"]))
            
            # 3. Подсчет объектов
            elif tool_call["name"] == "count_1c_objects":
                result_str = count_1c_objects.invoke(tool_call["args"])
                # Результат подсчёта передаётся модели, но не добавляется в список объектов.
                tool_responses.append(ToolMessage(content=result_str, tool_call_id=tool_call["id"]))

    return {"messages": tool_responses, "found_objects": current_found}

def should_continue(state: AgentState):
    last_message = state["messages"][-1]
    if not last_message.tool_calls:
        return END
    return "tools"

# 5. Сборка StateGraph
workflow = StateGraph(AgentState)
workflow.add_node("agent", agent_node)
workflow.add_node("tools", tool_node)

workflow.add_edge(START, "agent")
workflow.add_conditional_edges("agent", should_continue)
workflow.add_edge("tools", "agent")

# Компилируем приложение LangGraph
chat_app = workflow.compile()