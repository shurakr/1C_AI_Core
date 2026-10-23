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

import json
from rapidfuzz.distance import JaroWinkler
from openai import AsyncOpenAI
from typing import List, Optional, Dict, Any
from .normalizer import extract_candidate_fields, normalize_field_names
from .config import settings
from datetime import datetime
from .models import EnhancedScoredPoint
from .object_comparison import compare_objects


def calculate_jaccard(list1: List[str], list2: List[str]) -> float:
    """Вычисляет отношение пересечения множеств слов к их объединению (Jaccard)."""
    set1, set2 = set(list1), set(list2)
    if not set1 and not set2:
        return 0.0
    intersection = len(set1.intersection(set2))
    union = len(set1.union(set2))
    return intersection / union

def extract_amount(data_points: dict) -> float:
    """Читает СуммаДокумента или Сумма; при отсутствии или ошибке формата возвращает 0."""
    if not data_points:
        return 0.0
    fields = normalize_field_names(data_points)
    val = fields.get("суммадокумента") or fields.get("сумма")
    if not val:
        return 0.0
    try:
        # Убираем неразрывные пробелы, обычные пробелы и меняем запятую на точку
        clean_str = str(val).replace('\xa0', '').replace(' ', '').replace(',', '.')
        return float(clean_str)
    except (ValueError, TypeError):
        return 0.0

def evaluate_fast_track(request_norm: Dict[str, Any], qdrant_hits: List[Any]) -> Optional[Any]:
    """
    Выбирает первый вариант, прошедший условия Fast-Track.
    При непустом списке без такого варианта возвращает лучший по составному рейтингу.
    """
    # --- 1. ИЗВЛЕКАЕМ ДАННЫЕ ЗАПРОСА ---
    req_name = request_norm.get("canonical_name", "")
    req_keywords = request_norm.get("keywords", [])
    req_strict = request_norm.get("strict_filters", {}) 
    
    req_data = normalize_field_names(request_norm.get("data_points", {}))
    req_amount = extract_amount(req_data)
    req_currency = str(req_data.get("валюта", "")).strip().lower()
    
    # Флаг: ищем ли мы папку или элемент? (По умолчанию считаем, что элемент)
    req_is_folder = request_norm.get("is_folder", False)
    
    current_year = datetime.now().year
    fast_track_candidate = None
    enhanced_results = []

    # --- 2. АНАЛИЗИРУЕМ КАЖДОГО КАНДИДАТА ---
    for hit in qdrant_hits:
        vector_score = hit.score
        
        # Данные из Payload
        payload = hit.payload or {}
        hit_norm = payload.get("normalized", {})
        hit_name = hit_norm.get("canonical_name", "")
        hit_keywords = hit_norm.get("keywords", [])
        hit_strict = hit_norm.get("strict_filters", {})
        hit_data = normalize_field_names(hit_norm.get("data_points", {}))
        hit_is_low_quality = payload.get("is_low_quality", False)

        # Системные флаги 1С из Payload
        hit_is_folder = payload.get("is_folder", False)
        hit_posted = payload.get("posted") # Может быть None, True, False
        hit_date_str = payload.get("date", "")
        

        # --- ТЕКСТОВЫЕ МЕТРИКИ И STRICT KEYS ---
        string_score = JaroWinkler.similarity(req_name, hit_name)
        jaccard_score = calculate_jaccard(req_keywords, hit_keywords)

        strict_match = None
        shared_keys = set(req_strict.keys()).intersection(set(hit_strict.keys()))
        if shared_keys:
            strict_match = all(req_strict[k] == hit_strict[k] for k in shared_keys)

        # Бонусы и штрафы для составного рейтинга.
        business_bonus = 0.0

        # Совпадение сумм с допуском 1% даёт +0.03; валюты — ещё +0.01.
        hit_amount = extract_amount(hit_data)
        if req_amount > 0 and hit_amount > 0:
            if abs(req_amount - hit_amount) / req_amount <= 0.01:
                business_bonus += 0.03
        
        hit_currency = str(hit_data.get("валюта", "")).strip().lower()
        if req_currency and hit_currency and req_currency == hit_currency:
             business_bonus += 0.01

        # Совпадение всех общих строгих ключей даёт +0.05.
        if strict_match is True:
             business_bonus += 0.05

        # Учитываем различие между папкой и элементом.
        # Папка получает штраф, если исходный объект — элемент.
        if req_is_folder is False and hit_is_folder is True:
            business_bonus -= 0.10
        # Совпадение признака папки даёт +0.01.
        elif req_is_folder == hit_is_folder:
            business_bonus += 0.01

        # Непроведённый документ получает штраф -0.05.
        # При posted=None статус проведения не влияет на рейтинг.
        if hit_posted is False:
            business_bonus -= 0.05

        # Год не раньше предыдущего даёт +0.02; возраст больше трёх лет — штраф -0.02.
        if hit_date_str and len(hit_date_str) >= 4:
            try:
                hit_year = int(hit_date_str[:4])
                age = current_year - hit_year
                if age <= 1:
                    business_bonus += 0.02
                elif age > 3:
                    business_bonus -= 0.02
            except ValueError:
                pass

        # Низкая оценка качества текста уменьшает рейтинг на 0.50.
        # Этот штраф влияет на рейтинг, но сам по себе не запрещает Fast-Track.
        if hit_is_low_quality is True:
            business_bonus -= 0.50

        enhanced_hit = EnhancedScoredPoint(**hit.model_dump())
        enhanced_hit.composite_score = vector_score + business_bonus

        enhanced_hit.custom_metrics = {
            "string_score": round(string_score, 3),
            "jaccard_score": round(jaccard_score, 3),
            "strict_match": strict_match,
            "business_bonus": round(business_bonus, 3),
            "composite_score": round(enhanced_hit.composite_score, 3),
            "flags": {
                "is_folder": hit_is_folder,
                "is_draft": hit_posted is False,
                "is_old": (business_bonus < 0 and hit_date_str),
                "is_low_quality": hit_is_low_quality
            }
        }

        enhanced_results.append(enhanced_hit)

        # --- 4. ЛОГИКА FAST-TRACK (Мгновенный ответ) ---
        
        # Конфликт общих ключей или папка вместо элемента исключают прямой Fast-Track.
        if strict_match is False or (req_is_folder is False and hit_is_folder is True):
            enhanced_hit.composite_score -= 0.50 # Дополнительный штраф для запасного выбора по рейтингу.
            continue 

        # Требуются вектор > 0.92 и строки ≥ 0.90; дополнительно — совпадение общих ключей
        # либо Jaccard ≥ 0.80, если общих ключей нет.
        if not fast_track_candidate and vector_score > 0.92 and string_score >= 0.90:
            if strict_match is True:
                fast_track_candidate = enhanced_hit
            elif not shared_keys and jaccard_score >= 0.80:
                fast_track_candidate = enhanced_hit

    # Запасной выбор возвращает кандидата даже без прохождения порогов Fast-Track.
    if not fast_track_candidate and enhanced_results:
        enhanced_results.sort(key=lambda x: x.composite_score, reverse=True)
        return enhanced_results[0]
    
    return fast_track_candidate

async def analyze_document_with_llm(doc, similar_hits):
    """
    Универсальный RAG-пайплайн для анализа любого объекта 1С.
    """
    # 1. Нормализация входящего объекта
    norm = extract_candidate_fields(doc)

    # 2. Обработка пустой истории
    if not similar_hits:
        reason_text = "В истории базы знаний нет похожих объектов."
        return {
            "suggested_target_name": "",
            "suggested_target_uuid": "",
            "metadata_name": doc.metadata_name,
            "source_config": doc.source_config,
            "confidence": 0,
            "reason": reason_text,
            "similar_docs": [],
            "model_version": "qdrant-empty",
            "result": f"Не найдено. Причина: {reason_text}"
        }

    # При непустой выдаче evaluate_fast_track возвращает прямой или запасной выбор.
    best_hit = evaluate_fast_track(request_norm=norm, qdrant_hits=similar_hits)
    
    if best_hit:
        p = best_hit.payload
        target_name = p.get("target_name") or ""
        target_uuid = p.get("target_uuid") or ""
        target_meta = p.get("metadata_name") or ""
        # Фиксированное значение ответа; не вычисленная вероятность совпадения.
        confidence_pct = 99
        comparison = compare_objects(doc, p)

        return {
            "suggested_target_name": target_name,
            "suggested_target_uuid": target_uuid,
            "metadata_name": target_meta,
            "source_config": doc.source_config,
            "confidence": confidence_pct,
            "reason": comparison["comparison_summary"],
            **comparison,
            "similar_docs": [{
                "candidate_no": 1,
                "score":  float(best_hit.score),
                "target_name": target_name,
                "target_uuid": target_uuid,
                "metadata_name": target_meta,
                "ref_uuid": p.get("ref_uuid")
            }],
            "model_version": "fast-track",
            "result": (f"Успешно (Fast-Track): {target_name} (Уверенность: {confidence_pct}%). "
                       f"{comparison['comparison_summary']}")
        }

    # 4. ПОДГОТОВКА КОНТЕКСТА ИСТОРИИ (history_lines и compact_similar_docs)
    history_lines = []
    compact_similar_docs = []
    
    for i, hit in enumerate(similar_hits, 1):
        p = hit.payload
        hit_norm = p.get("normalized", {})
        # Исходные hits не обогащаются арбитром; у них custom_metrics может отсутствовать.
        metrics = getattr(hit, "custom_metrics", {})
        comparison = compare_objects(doc, p)
        
        # Строка для LLM (человекочитаемая)
        line = (f"КАНДИДАТ №{i}:\n"
                f"- Название: {p.get('target_name')}\n"
                f"- UUID: {p.get('target_uuid')}\n"
                f"- Описание: {hit_norm.get('description', '')}\n"
                f"- Реквизиты: {json.dumps(hit_norm.get('data_points', {}), ensure_ascii=False)}\n"
                f"- ПРОВЕРЕННОЕ СРАВНЕНИЕ: {json.dumps(comparison['comparison'], ensure_ascii=False)}\n"
                f"- МЕТРИКИ СХОДСТВА (Подсказка для тебя): Вектор={hit.score:.3f}, "
                f"Текст={metrics.get('string_score', 0)}, "
                f"Токены={metrics.get('jaccard_score', 0)}, "
                f"Совпадение_Ключей={metrics.get('strict_match', 'Не применимо')}")
        history_lines.append(line)        
        
        # Структура для возврата в 1С 
        compact_similar_docs.append({
            "candidate_no": i,
            "score": float(hit.score),
            "target_name": p.get("target_name"),
            "target_uuid": p.get("target_uuid"),
            "metadata_name": p.get("metadata_name"), 
            "ref_uuid": p.get("ref_uuid")
        })

    history_context = "\n".join(history_lines)

    # 5. СИСТЕМНЫЙ ПРОМПТ СО СТРОГИМИ ПРАВИЛАМИ
    system_prompt = f"""Ты — ИИ-эксперт по анализу данных в системах 1С. 
Твоя задача: подобрать наиболее подходящий Целевой Объект (suggested_target_name) для нового объекта системы на основе исторических прецедентов.
Объектами могут быть как Справочники (поиск дублей/эталонов), так и Документы (подбор аналитики на основе похожих исторических операций).

НОВЫЙ ОБЪЕКТ ДЛЯ АНАЛИЗА:
- Тип: {doc.metadata_name}
- Суть: {norm['description']}
- Реквизиты: {json.dumps(norm['data_points'], ensure_ascii=False)}

КРИТИЧЕСКИЕ ПРАВИЛА:
1. Выбирай целевой объект (suggested_target_name) и его UUID ТОЛЬКО из списка кандидатов ниже.
2. КАТЕГОРИЧЕСКИ ЗАПРЕЩЕНО придумывать новые названия, UUID или реконструировать его из текста или менять его наименование.
3. ИГНОРИРОВАНИЕ ТЕХНИЧЕСКОГО МУСОРА: Системы (ERP, УПП) содержат служебные поля (пути к папкам, коды обмена, системные идентификаторы). Игнорируй их. Сравнивай объекты ТОЛЬКО по бизнес-сути (ИНН, наименования, суммы, контрагенты, назначения платежей, артикулы).
4. Если среди кандидатов нет надежного варианта (confidence ниже 60), верни пустые строки в suggested_target_name и suggested_target_uuid (или UUID из нулей).
5. СТРОГОЕ СОВПАДЕНИЕ ТИПОВ: Кандидат должен относиться к тому же кластеру метаданных, что и анализируемый объект. Категорически запрещено предлагать Документ в качестве эталона для Справочника, и наоборот.

ИНСТРУКЦИЯ ПО ВЫБОРУ:
- Внимательно изучи все доступные поля нового объекта (описания, комментарии, наименования).
- СОПОСТАВЬ ключевые бизнес-реквизиты (данные из data_points) нового объекта с данными кандидатов.
- УЧИТЫВАЙ МЕТРИКИ СХОДСТВА (предоставлены у каждого кандидата): Значение Совпадение_Ключей=False — это критический сигнал о различии объектов. Отбраковывай кандидата, если есть расхождения в смысле или суммах. Исключение, если все остальные данные идентичны — в таком случае выбери объект, но обязательно укажи на конфликт ключей в поле reason и снизь confidence.
- Если "Совпадение_Ключей=True" или метрика "Текст" > 0.90 (при отсутствии явного конфликта ключей), это сильный сигнал, что перед тобой дубль или искомый объект.
- Для Справочников обращай первоочередное внимание на уникальные идентификаторы (ИНН, Артикул). Для Документов — на логику операции (схожие контрагенты, суммы, статьи).
- Найди кандидата, чей общий логический смысл и набор свойств максимально совпадают с новым объектом.
- Если уверенный прецедент найден, возьми suggested_target_name, suggested_target_uuid  СТРОГО из него.
- Оцени уверенность своего выбора по шкале от 0 до 100.
- В reason объясни выбор через существенные совпадения и различия ТОЛЬКО выбранного кандидата.
- Проверенное сравнение реквизитов рассчитано Python: не изменяй его факты, суммы, валюты и идентификаторы.
- Отсутствие данных не означает совпадения или различия. Не объявляй разные названия разными объектами без оснований.
- При смысловой оценке назначения платежа укажи, какие фрагменты исходного и найденного текста её подтверждают.
- Не объявляй отличие суммы, даты или договора допустимым без правил сценария; обозначь ограничение оценки.

ОТВЕТЬ СТРОГО В ФОРМАТЕ JSON:
{{
  "suggested_target_name": "имя из истории или пустая строка",
  "suggested_target_uuid": "uuid из истории или пустая строка",
  "confidence": число от 0 до 100,
  "reason": "Краткое объяснение выбора, существенных совпадений, различий и недостающих данных на русском языке"
}}
"""

    # 6. ВЫЗОВ МОДЕЛИ (history_context ПЕРЕДАЕТСЯ в сообщении пользователя)
    try:
        client = AsyncOpenAI(
            base_url=settings.base_url,
            api_key=settings.openrouter_api_key
        )
        
        response = await client.chat.completions.create(
            model=settings.openrouter_model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"СПИСОК КАНДИДАТОВ ИЗ ИСТОРИИ:\n{history_context}"}
            ],
            response_format={"type": "json_object"},
            temperature=0.0  # Снижает вариативность ответа, но не гарантирует его корректность.
        )

        data = json.loads(response.choices[0].message.content)
        # Сравнение возвращается только из Python и только для принятого кандидата.
        data["comparison"] = []
        data["comparison_summary"] = ""

        # Проверяем UUID по списку кандидатов и минимальный confidence; это не проверка всех фактов.
        allowed_uuids = {h["target_uuid"] for h in compact_similar_docs if h["target_uuid"]}
        selected_uuid = data.get("suggested_target_uuid")
        conf = int(data.get("confidence", 0))

        if selected_uuid not in allowed_uuids or conf < 60:
            data.update({
                "suggested_target_name": "",
                "suggested_target_uuid": "",
                "metadata_name": doc.metadata_name,
                "source_config": doc.source_config,
                "confidence": 0,
                "reason": f"Результат отклонен (уверенность {conf}% ниже порога 60% или UUID не из списка)."
            })

        # 8. ФОРМИРОВАНИЕ ПОЛЯ RESULT ДЛЯ 1С
        if data.get("suggested_target_uuid"):
            selected_hit = next(hit for hit in similar_hits
                                if (hit.payload or {}).get("target_uuid") == data["suggested_target_uuid"])
            comparison = compare_objects(doc, selected_hit.payload or {})
            data.update(comparison)
            data["reason"] = (str(data.get("reason", "")).strip() + " "
                              + comparison["comparison_summary"]).strip()
            t_name = data.get("suggested_target_name", "Неизвестный объект")
            data["result"] = f"Успешно: {t_name} (Уверенность: {conf}%). {data['reason']}"
        else:
            reason_text = data.get("reason", "Нет подходящих вариантов.")
            data["result"] = f"Не найдено. Причина: {reason_text}"

    except Exception as e:
        logger.error(f"LLM Error: {str(e)}")
        return {
            "source_config": doc.source_config,
            "metadata_name": doc.metadata_name,
            "confidence": 0,
            "result": f"Ошибка анализа. {str(e)}",
            "reason": str(e),
            "similar_docs": compact_similar_docs,
            "model_version": settings.openrouter_model
        }

    data["model_version"] = settings.openrouter_model
    data["similar_docs"] = compact_similar_docs
    data["source_config"] = doc.source_config
    data["metadata_name"] = doc.metadata_name
    return data
