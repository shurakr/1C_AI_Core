from typing import List, Tuple, Dict, Any
from .models import OneCDocument
from .normalizer import extract_candidate_fields, build_search_text, build_payload

def is_high_quality_text(text: str, min_words: int = 3, ttr_threshold: float = 0.5) -> bool:
    """
    Проверяет длину текста и долю уникальных слов (TTR), без оценки смысла.
    """
    if not text:
        return False
        
    words = [w.lower() for w in text.split() if len(w) > 1]
    total_words = len(words)
    
    if total_words < min_words:
        return False
        
    unique_words = len(set(words))
    ttr = unique_words / total_words
    
    if ttr < ttr_threshold:
        return False
        
    return True

def prepare_batch_for_indexing(raw_docs: List[OneCDocument], get_qdrant_id_func) -> Tuple[List[str], List[str], List[dict], List[dict]]:
    """
    Подготавливает пакет объектов к индексации и помечает тексты с низким TTR.
    Записи с низким качеством не исключаются из загрузки.
    Возвращает списки: ids, texts, payloads, object_results (для ответа 1С).
    """
    ids = []
    texts = []
    payloads = []
    object_results = []
    
    seen_master_records = {} 

    for i, doc in enumerate(raw_docs, 1):
        try:
            norm = extract_candidate_fields(doc)
            search_text = build_search_text(doc)
            
            is_garbage = not is_high_quality_text(search_text)

            # Подготовка к векторизации
            payload = build_payload(doc)
            payload["is_low_quality"] = is_garbage

            ids.append(get_qdrant_id_func(doc.source_config, doc.ref_uuid))
            texts.append(search_text)
            payloads.append(payload)
            
            object_results.append({
                "ref_uuid": doc.ref_uuid,
                "status": "prepared",
                "source_config": doc.source_config,
                "metadata_name": doc.metadata_name
            })
            
        except Exception as e:
            error_msg = f"Data preparation failed: {str(e)}"
            object_results.append({
                "ref_uuid": doc.ref_uuid,
                "status": "error",
                "source_config": doc.source_config,
                "metadata_name": getattr(doc, 'metadata_name', 'unknown'),
                "message": error_msg
            })

    return ids, texts, payloads, object_results