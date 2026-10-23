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

import re
from .models import OneCDocument
from .config import settings

def normalize_field_names(fields: dict) -> dict:
    """Единый регистр реквизитов для выгрузок 1С и ранее сохранённых данных."""
    return {key.lower(): value for key, value in fields.items()}


def canonicalize_name(value: str, metadata_name: str = "") -> str:
    """
    Контекстно-зависимая подготовка имени к нечеткому сравнению.
    Учитывает тип объекта 1С, чтобы не удалить важные технические данные.
    """
    if not value:
        return ""
    
    name = str(value).upper()
    meta_lower = metadata_name.lower()
    
    # Для контрагентов и организаций применяется отдельное правило очистки.
    is_contractor = "контрагент" in meta_lower or "организаци" in meta_lower
    
    if is_contractor:
        # Удаляем кавычки и фрагменты в скобках у контрагентов и организаций.
        name = re.sub(r'["\'«»]', '', name) # Убираем кавычки
        name = re.sub(r'\(.*?\)', '', name)   # Убираем комментарии в скобках
    else:
        # Для номенклатуры, ОС, договоров, статей затрат и прочего - мягкая чистка.
        # Убираем только текстовые кавычки-елочки, оставляем прямые кавычки (дюймы) и скобки (свойства).
        name = re.sub(r'[«»]', '', name)

    # Сводим переносы строк и повторные пробелы к одному пробелу.
    return cleanup_text(name)

def extract_keywords(text: str, metadata_name: str = "") -> list:
    """
    Выделяет ключевые токены. Для номенклатуры сохраняет даже короткие слова.
    """
    # Правила очистки зависят от типа объекта.
    text = canonicalize_name(text, metadata_name)
    keywords = []
    
    meta_lower = metadata_name.lower()
    is_contractor = "контрагент" in meta_lower or "организаци" in meta_lower
    
    for word in text.split():
        # Для контрагентов и организаций минимальная длина слова — три символа.
        # Для остальных объектов — два; токены с цифрами сохраняются независимо от длины.
        min_length = 3 if is_contractor else 2
        
        if len(word) >= min_length or any(char.isdigit() for char in word):
            keywords.append(word)
            
    return keywords

def clean_identifier(value: any) -> str:
    """Оставляет в идентификаторе буквы, цифры и дефис; удаляет пробелы и прочие символы."""
    if not value:
        return ""
    return re.sub(r'[^a-zA-Zа-яА-Я0-9\-]', '', str(value))

def cleanup_text(value: str) -> str:
    """Заменяет переносы строк и повторные пробелы одним пробелом."""
    if not value:
        return ""
    value = str(value).replace("\n", " ").replace("\r", " ")
    return re.sub(r"\s+", " ", value).strip()

def extract_candidate_fields(doc: OneCDocument) -> dict:
    """
    Формирует структурированное описание объекта для LLM, 
    учитывая иерархию и системные признаки 1С.
    """
    f = normalize_field_names(doc.raw_fields)
    
    # 1. Системные префиксы (Hierarchy & Folder)
    h_path = getattr(doc, 'hierarchy', "")
    is_f = getattr(doc, 'is_folder', False)
    
    hierarchy_prefix = f"[{h_path}] " if h_path else ""
    folder_label = "ПАПКА: " if is_f else ""
    
    # 2. Определение типа метаданных
    is_catalog = "справочник" in doc.metadata_name.lower()
    type_name = doc.metadata_name.split('.')[-1]

    # 3. Сборка target_name в зависимости от типа сущности
    if is_f:
        # Для объекта с is_folder формируем название с признаком папки.
        raw_name = f.get("наименование") or doc.target_name
        final_name = f"{hierarchy_prefix}{folder_label}{raw_name}"
    
    elif is_catalog:
        # Для справочников: [Путь] Контрагент код 001 (Наименование)
        val_id = f.get("код") or getattr(doc, 'number', 'Б/К')
        item_name = f.get("наименование") or ""
        final_name = f"{hierarchy_prefix}{type_name} код {val_id} ({item_name})".strip()
    
    else:
        # Для документов: [Путь] Списание безналичных ДС №333 от 06.04.2026 (Сумма)
        val_id = f.get("номер") or getattr(doc, 'number', 'Б/Н')
        val_date = f.get("дата") or getattr(doc, 'date', "")
        pretty_date = val_date[:10].replace("-", ".") if val_date else ""
        
        # Добавляем сумму и валюту в представление документа.
        amount = f.get("суммадокумента") or f.get("сумма")
        currency = f.get("валюта") or "руб."
        summary = f" ({amount} {currency})" if amount else ""
        
        id_str = f"№ {val_id}" + (f" от {pretty_date}" if pretty_date else "")
        final_name = f"{hierarchy_prefix}{type_name} {id_str}{summary}"
    
    final_target_uuid = doc.target_uuid if doc.target_uuid else doc.ref_uuid
    final_target_name = doc.target_name if doc.target_name else final_name
   
    object_description = cleanup_text(f"{hierarchy_prefix}{folder_label}{final_name}")

    strict_filters = {}
    for key in settings.STRICT_KEYS:
        val = f.get(key)
        if val:
            cleaned_val = clean_identifier(val)
            if cleaned_val:
                strict_filters[f"raw_fields.{key}"] = cleaned_val

    # 4. Собираем значимые реквизиты (data_points)
    # Сохраняем реквизиты после исключения отдельных пустых и технических значений.
    clean_fields = {}
    canonical_name = "" # Для канонизации имени

    for k, v in f.items():
        val_str = cleanup_text(str(v))
        # Исключаем пустые строки, строковый ноль, указанную пустую дату и поля с _УИД в имени.
        if not val_str or v == "0" or v == "01.01.0001 0:00:00" or "_уид" in k:
            continue
        clean_fields[k] = v

        # Ищем главное поле с учетом контекста метаданных
        if not canonical_name and k.upper() in ["НАИМЕНОВАНИЕ", "ПОЛНОЕНАИМЕНОВАНИЕ", "НАЗНАЧЕНИЕПЛАТЕЖА"]:
            canonical_name = canonicalize_name(val_str, doc.metadata_name)

    # Если в полях не нашли подходящего имени (например, это документ), берем из target_name
    if not canonical_name:
        canonical_name = canonicalize_name(final_target_name, doc.metadata_name)        

    return {
        "ref_uuid": doc.ref_uuid,
        "description": object_description,
        "data_points": clean_fields,
        "target_uuid": final_target_uuid, 
        "target_name": final_target_name,
        "hierarchy": doc.hierarchy,
        "is_folder": doc.is_folder,
        "posted": doc.posted,
        "metadata_name": doc.metadata_name,
        "strict_filters": strict_filters,
        "canonical_name": canonical_name,
        "keywords": extract_keywords(canonical_name, doc.metadata_name)
    }

def build_search_text(doc: OneCDocument) -> str:
    """
    Создаёт текст для векторизации, ограниченный 1600 символами.
    После типа объекта размещает строгие ключи, затем бизнес-реквизиты и данные ТЧ.
    """
    x = extract_candidate_fields(doc)
    fields = normalize_field_names(doc.raw_fields)
    parts = []

    # Тип объекта задаёт контекст поискового описания.
    parts.append(f"Тип: {doc.metadata_name}")

    # Строгие ключи размещаются перед остальными реквизитами.
    for key in settings.STRICT_KEYS:
        val = fields.get(key)
        if val:
            cleaned_val = clean_identifier(val)
            if cleaned_val:
                parts.append(f"{key.upper()}: {cleaned_val}")

    # 3. Финансы (Сумма и валюта)
    data_pts = x.get("data_points", {})
    amount = data_pts.get("суммадокумента") or data_pts.get("сумма")
    currency = data_pts.get("валюта") or "руб."
    if amount:
        parts.append(f"ФИНАНСЫ: {amount} {currency}")
        
    # 4. Мягкие текстовые ключи (Организация/Контрагент)
    org = fields.get("организация") or fields.get("контрагент")
    if org:
        parts.append(f"КОНТРАГЕНТ: {cleanup_text(str(org))}")

    # Иерархия и системные признаки объекта.
    if doc.hierarchy:
        parts.append(f"Путь: {doc.hierarchy}")
    if doc.is_folder:
        parts.append(f"Это_Папка: Да")
    if doc.posted is not None:
        parts.append(f"Статус: {'Проведен' if doc.posted else 'Черновик'}")
        
    if x.get('description'):
        parts.append(f"Суть: {x['description']}")
    
    # Для остальных реквизитов включаем значения длиннее двух символов.
    # Уже включённые реквизиты повторно не добавляются.
    exclude_keys = {"суммадокумента", "сумма", "валюта", "организация", "контрагент"}
    exclude_keys.update(settings.STRICT_KEYS) 

    for key, val in data_pts.items():
        if key.lower() not in exclude_keys:
            val_str = cleanup_text(str(val))
            if len(val_str) > 2:
                parts.append(f"{key}: {val_str}")
            
    # 7. Данные из табличных частей (ТЧ)
    for tp_name, rows in doc.table_parts.items():
        tp_values = []
        for row in rows:
            for col_key, col_val in row.items():
                val_str = cleanup_text(str(col_val))
                if len(val_str) > 2:
                    tp_values.append(val_str)
        if tp_values:
            # Берем уникальные значения, чтобы не раздувать вектор повторами
            parts.append(f"ТЧ_{tp_name}: {', '.join(list(set(tp_values))[:10])}")

    full_text = " | ".join(filter(None, parts))

    return full_text[:1600]

def build_payload(doc: OneCDocument) -> dict:
    """
    Формирует полный Payload для хранения в Qdrant.
    """
    norm = extract_candidate_fields(doc)

    clean_date = ""
    if doc.date and not doc.date.startswith("0001-01-01"):
        clean_date = doc.date.strip()
        
    clean_number = doc.number.strip().upper() if doc.number else ""    

    clean_raw_fields = {}
    for k, v in normalize_field_names(doc.raw_fields).items():
        if k in settings.STRICT_KEYS:
            clean_raw_fields[k] = clean_identifier(v)
        else:
            clean_raw_fields[k] = v

    return {
        # Идентификация источника
        "source_config": doc.source_config,
        "ref_uuid": doc.ref_uuid,
        "hash": doc.hash,            
        
        # Основные реквизиты (вынесены на верхний уровень для фильтров Qdrant)
        "metadata_name": doc.metadata_name,
        "number": clean_number,
        "date": clean_date,
        
        # Данные целевого объекта для последующего выбора аналога.
        "target_name": norm["target_name"],
        "target_uuid": norm["target_uuid"],
        
        # Иерархия и признаки
        "hierarchy": norm.get("hierarchy", ""),
        "is_folder": doc.is_folder,
        "posted": doc.posted,
        
        # Контейнеры данных
        "normalized": norm,
        "raw_fields": clean_raw_fields,
        "table_parts": doc.table_parts
    }
