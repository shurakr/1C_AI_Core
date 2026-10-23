"""Проверяемые различия исходного объекта и одного выбранного прецедента."""

from decimal import Decimal, InvalidOperation

from .config import settings
from .normalizer import clean_identifier, cleanup_text, normalize_field_names


def compare_objects(doc, payload: dict) -> dict:
    # raw_fields сохраняет нули и пустые значения, которые нормализатор мог удалить.
    source = normalize_field_names(doc.raw_fields)
    candidate = normalize_field_names(payload.get("normalized", {}).get("data_points", {}))
    candidate.update(normalize_field_names(payload.get("raw_fields", {})))
    if "справочник" in doc.metadata_name.lower():
        source.setdefault("наименование", doc.target_name)
        candidate.setdefault("наименование", payload.get("target_name"))
    fields = [
        ("сумма", "Сумма", ("суммадокумента", "сумма")),
        ("валюта", "Валюта", ("валюта", "валютадокумента")),
        ("организация", "Организация", ("организация",)),
        ("контрагент", "Контрагент", ("контрагент",)),
        ("договор", "Договор", ("договорконтрагента", "договор")),
        ("видоперации", "Вид операции", ("видоперации", "видхозяйственнойоперации")),
        ("статьязатрат", "Статья затрат", ("статьязатрат",)),
        ("статьядвиженияденежныхсредств", "Статья движения денежных средств",
         ("статьядвиженияденежныхсредств",)),
        ("назначениеплатежа", "Назначение платежа", ("назначениеплатежа",)),
        ("наименование", "Наименование", ("наименование",)),
    ]
    fields.extend((key, key.upper(), (key,)) for key in settings.STRICT_KEYS)
    fields.extend([
        ("date", "Дата", ("дата",)),
        ("posted", "Проведён", ("проведен",)),
    ])
    source = dict(source)
    for data, date, posted in ((source, doc.date, doc.posted),
                               (candidate, payload.get("date"), payload.get("posted"))):
        if "дата" not in data:
            data["дата"] = date
        if "проведен" not in data:
            data["проведен"] = posted

    comparison = []
    for field, label, aliases in fields:
        values = []
        for data in (source, candidate):
            value = next((data[key] for key in aliases
                          if key in data and data[key] is not None
                          and str(data[key]).strip() != ""), None)
            if field == "date" and value is not None and str(value).startswith("0001"):
                value = None
            values.append(value)
        left, right = values
        if left is None and right is None:
            continue
        if left is None:
            status = "missing_source"
        elif right is None:
            status = "missing_candidate"
        elif field == "сумма":
            try:
                numbers = [Decimal(str(value).replace("\xa0", "").replace(" ", "").replace(",", "."))
                           for value in values]
                status = ("match" if numbers[0] == numbers[1] else "different") if all(
                    number.is_finite() for number in numbers) else "uncomparable"
            except InvalidOperation:
                status = "uncomparable"
        elif field in settings.STRICT_KEYS:
            cleaned = [clean_identifier(value).casefold() for value in values]
            status = ("match" if cleaned[0] == cleaned[1] else "different") if all(cleaned) else "uncomparable"
        else:
            status = "match" if cleanup_text(str(left)).casefold() == cleanup_text(str(right)).casefold() else "different"
        comparison.append(dict(field=field, label=label, source_value=left,
                               candidate_value=right, status=status))

    parts = []
    matches = [item["label"] for item in comparison if item["status"] == "match"]
    if matches:
        parts.append("Совпадают значения: " + ", ".join(matches) + ".")
    for item in comparison:
        label, left, right, status = (item[key] for key in
                                      ("label", "source_value", "candidate_value", "status"))
        if status == "different":
            parts.append(f"{label}: в исходном объекте «{left}», в найденном «{right}».")
        elif status == "missing_source":
            parts.append(f"{label}: нет данных в исходном объекте; в найденном «{right}».")
        elif status == "missing_candidate":
            parts.append(f"{label}: в исходном объекте «{left}»; в найденном нет данных.")
        elif status == "uncomparable":
            parts.append(f"{label}: не удалось сопоставить значения «{left}» и «{right}».")
    if any(item["status"] == "different" and item["field"] in
           {"наименование", "назначениеплатежа", "контрагент", "организация", "договор"}
           for item in comparison):
        parts.append("Различия в текстовых значениях сами по себе не доказывают различие объектов или операций.")
    return {"comparison": comparison,
            "comparison_summary": " ".join(parts) or "Недостаточно данных для сравнения реквизитов."}
