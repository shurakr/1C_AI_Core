import json
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from qdrant_client.http.models import ScoredPoint

from app.models import AnalyzeResponse, OneCDocument
from app.object_comparison import compare_objects
from app.rag_pipeline import analyze_document_with_llm


def document(fields=None, **kwargs):
    return OneCDocument(source_config="TEST", metadata_name="Документ.Платеж",
                        ref_uuid="source", raw_fields=fields or {}, **kwargs)


def hit(uuid="chosen", fields=None, score=0.95):
    return ScoredPoint(id=1, version=0, score=score, payload={
        "target_uuid": uuid, "target_name": uuid,
        "metadata_name": "Документ.Платеж", "raw_fields": fields or {},
        "normalized": {"canonical_name": "ПЛАТЕЖ", "keywords": ["ПЛАТЕЖ"]},
    })


class ComparisonTests(unittest.TestCase):
    def test_values_case_formats_and_missing_data(self):
        doc = document({"СуммаДокумента": "15\u00a0000,50", "Валюта": "USD",
                        "ИНН": "123 456", "Контрагент": "Поставщик"})
        result = compare_objects(doc, {"raw_fields": {
            "сумма": 15000.5, "валюта": "usd", "инн": "123456", "договор": "Договор 15"}})
        items = {item["field"]: item for item in result["comparison"]}
        for key in ("сумма", "валюта", "инн"):
            self.assertEqual(items[key]["status"], "match")
        self.assertEqual(items["контрагент"]["status"], "missing_candidate")
        self.assertEqual(items["договор"]["status"], "missing_source")
        AnalyzeResponse(**result)

    def test_zero_amount_is_not_missing(self):
        result = compare_objects(document({"сумма": 0}), {"raw_fields": {"сумма": 100}})
        self.assertEqual(result["comparison"][0]["status"], "different")
        self.assertEqual(result["comparison"][0]["source_value"], 0)

    def test_invalid_amount_is_not_a_match(self):
        for value in ("не число", "NaN", "Infinity"):
            result = compare_objects(document({"сумма": value}), {"raw_fields": {"сумма": value}})
            self.assertEqual(result["comparison"][0]["status"], "uncomparable")

    def test_old_history_fallback_and_raw_precedence(self):
        payload = {"normalized": {"data_points": {"СуммаДокумента": 100}},
                   "raw_fields": {"СуммаДокумента": 0}}
        result = compare_objects(document({"сумма": 0}), payload)
        self.assertEqual(result["comparison"][0]["status"], "match")
        payload.pop("raw_fields")
        result = compare_objects(document({"сумма": 100}), payload)
        self.assertEqual(result["comparison"][0]["status"], "match")

    def test_dates_posting_and_empty_data(self):
        result = compare_objects(document(date="2026-10-22", posted=False),
                                 {"date": "2025-10-22", "posted": True})
        self.assertEqual([item["status"] for item in result["comparison"]], ["different", "different"])
        empty = compare_objects(document(), {"date": "0001-01-01"})
        self.assertEqual(empty["comparison"], [])
        self.assertIn("Недостаточно данных", empty["comparison_summary"])


class PipelineTests(unittest.IsolatedAsyncioTestCase):
    async def test_fast_track_selected_only_and_no_llm(self):
        chosen = hit(fields={"сумма": 42000, "валюта": "USD"})
        other = hit("other", {"сумма": 999999})
        with patch("app.rag_pipeline.evaluate_fast_track", return_value=chosen), \
             patch("app.rag_pipeline.AsyncOpenAI") as client:
            raw = await analyze_document_with_llm(document({"сумма": 45000, "валюта": "USD"}),
                                                  [other, chosen])
        client.assert_not_called()
        result = AnalyzeResponse(**raw)
        self.assertEqual(result.suggested_target_uuid, "chosen")
        self.assertIn("42000", result.reason)
        self.assertNotIn("999999", result.reason)
        self.assertIn(result.comparison_summary, result.result)
        self.assertEqual(result.confidence, 99)

    async def llm_result(self, selected_uuid, confidence=85):
        response = {"suggested_target_uuid": selected_uuid, "suggested_target_name": selected_uuid,
                    "confidence": confidence, "reason": "Сходное назначение операции.",
                    "comparison": [{"invented": True}], "comparison_summary": "Выдуманные факты"}
        create = AsyncMock(return_value=SimpleNamespace(choices=[
            SimpleNamespace(message=SimpleNamespace(content=json.dumps(response)))]))
        client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
        with patch("app.rag_pipeline.evaluate_fast_track", return_value=None), \
             patch("app.rag_pipeline.AsyncOpenAI", return_value=client):
            result = await analyze_document_with_llm(document({"сумма": 45000}),
                                                     [hit("other", {"сумма": 999999}),
                                                      hit("chosen", {"сумма": 42000})])
        self.assertIn("ПРОВЕРЕННОЕ СРАВНЕНИЕ", create.call_args.kwargs["messages"][1]["content"])
        return AnalyzeResponse(**result)

    async def test_llm_uses_accepted_candidate_and_python_facts(self):
        result = await self.llm_result("chosen")
        self.assertIn("42000", result.comparison_summary)
        self.assertNotIn("999999", result.comparison_summary)
        self.assertNotIn("Выдуманные", result.comparison_summary)
        self.assertIn("Сходное назначение", result.reason)

    async def test_rejected_candidate_has_no_comparison(self):
        for selected_uuid, confidence in (("invented", 90), ("chosen", 50), ("", 0)):
            result = await self.llm_result(selected_uuid, confidence)
            self.assertEqual(result.comparison, [])
            self.assertEqual(result.comparison_summary, "")
            self.assertFalse(result.suggested_target_uuid)

    async def test_no_history(self):
        result = AnalyzeResponse(**await analyze_document_with_llm(document(), []))
        self.assertEqual(result.comparison, [])
        self.assertEqual(result.confidence, 0)


if __name__ == "__main__":
    unittest.main()
