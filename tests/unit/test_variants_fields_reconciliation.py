import unittest

import numpy as np

from kyc_engine.contracts import (
    BoundingBox,
    DocumentProfile,
    FieldDefinition,
    FieldStatus,
    OCRCandidate,
    VariantBatch,
    VariantInfo,
)
from kyc_engine.fields import FieldLocalizer
from kyc_engine.quality import QualityAssessmentPipeline, StructuralQualityGate
from kyc_engine.reconciliation import CandidateReconciler
from kyc_engine.variants import BalancedVariantPolicy


class VariantTests(unittest.TestCase):
    def test_balanced_policy_produces_ten_variants_without_mutation(self) -> None:
        image = np.arange(100 * 160 * 3, dtype=np.uint8).reshape((100, 160, 3))
        original = image.copy()
        batches = BalancedVariantPolicy().generate(image)
        self.assertEqual(10, sum(len(batch.variants) for batch in batches))
        self.assertTrue(np.array_equal(image, original))
        self.assertEqual("original", batches[0].variants[0].name)

    def test_structural_gate_rejects_constant_variants(self) -> None:
        constant = VariantBatch(
            "original",
            (VariantInfo("constant", np.zeros((20, 20), dtype=np.uint8), {}),),
        )
        assessments = QualityAssessmentPipeline().assess((constant,))
        self.assertEqual((), StructuralQualityGate().select((constant,), assessments))


class FieldTests(unittest.TestCase):
    def setUp(self) -> None:
        self.field = FieldDefinition(
            "full_name",
            BoundingBox(10, 10, 60, 20),
            normalizer="name",
            validator="name",
        )
        self.profile = DocumentProfile(
            "test/front/v1",
            "test",
            "front",
            100,
            60,
            "reviewed",
            (self.field,),
        )

    def test_localizes_every_field_for_every_variant(self) -> None:
        batch = VariantBatch(
            "original",
            (
                VariantInfo("a", np.ones((60, 100, 3), dtype=np.uint8), {}),
                VariantInfo("b", np.ones((60, 100), dtype=np.uint8), {}),
            ),
        )
        crops = FieldLocalizer().localize((batch,), self.profile)
        self.assertEqual(("a", "b"), tuple(crop.variant_name for crop in crops))
        self.assertTrue(all(crop.image.shape[:2] == (20, 60) for crop in crops))

    def test_reconciliation_prefers_consensus_and_preserves_provenance(self) -> None:
        candidates = (
            self._candidate("MARIA SILVA", 0.72, "original"),
            self._candidate("Maria   Silva", 0.81, "gamma"),
            self._candidate("MARIA S1LVA", 0.95, "threshold"),
        )
        result = CandidateReconciler().reconcile(self.profile, candidates)
        field = result.fields["full_name"]
        self.assertEqual(FieldStatus.VALID, field.status)
        self.assertEqual("Maria   Silva", field.raw_value)
        self.assertEqual("gamma", field.selected_candidate.variant_name)
        self.assertEqual(2, len(field.alternatives))

    def test_missing_required_field_is_reported(self) -> None:
        result = CandidateReconciler().reconcile(self.profile, ())
        self.assertEqual(FieldStatus.MISSING, result.fields["full_name"].status)
        self.assertEqual("FIELD_MISSING", result.issues[0].code)

    def _candidate(self, value: str, confidence: float, variant: str) -> OCRCandidate:
        return OCRCandidate(
            field_name="full_name",
            raw_value=value,
            confidence=confidence,
            generator="test",
            variant_name=variant,
            bounding_box=self.field.bounding_box,
            ocr_engine="fake",
            ocr_model_version="1",
        )


if __name__ == "__main__":
    unittest.main()
