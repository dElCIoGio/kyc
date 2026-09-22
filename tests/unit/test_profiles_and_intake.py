import io
import unittest

import numpy as np
from PIL import Image

from kyc_engine.contracts import BoundingBox, DocumentProfile, FieldDefinition
from kyc_engine.intake import ImageIntake, IntakeError, IntakeLimits
from kyc_engine.profiles import (
    ProfileRegistry,
    load_default_profile,
    load_default_profiles,
    validate_profile,
)


class ProfileTests(unittest.TestCase):
    def test_loads_default_provisional_profile(self) -> None:
        profile = load_default_profile()
        self.assertEqual("ao_id_card/front/v1", profile.profile_id)
        self.assertEqual((718, 467), (profile.canonical_width, profile.canonical_height))
        self.assertEqual(
            ("full_name", "father_name", "mother_name", "id_number"),
            tuple(field.name for field in profile.fields),
        )
        self.assertEqual("provisional", profile.review_status)
        self.assertEqual(
            ("full_name", "id_number"),
            tuple(field.name for field in profile.fields if field.required),
        )
        father_name = next(field for field in profile.fields if field.name == "father_name")
        self.assertEqual(BoundingBox(34, 262, 300, 52), father_name.bounding_box)
        self.assertEqual(0, father_name.padding)

    def test_rejects_duplicate_fields(self) -> None:
        field = FieldDefinition("name", BoundingBox(0, 0, 10, 10))
        profile = DocumentProfile("p", "doc", "front", 100, 50, "reviewed", (field, field))
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            validate_profile(profile)

    def test_loads_provisional_back_profile(self) -> None:
        profile = next(item for item in load_default_profiles() if item.side == "back")
        self.assertEqual("ao_id_card/back/v1", profile.profile_id)
        self.assertEqual((718, 467), (profile.canonical_width, profile.canonical_height))
        self.assertEqual(
            (
                "residence",
                "place_of_birth",
                "province",
                "date_of_birth",
                "sex",
                "height_meters",
                "marital_status",
                "issue_date",
                "expiry_date",
            ),
            tuple(field.name for field in profile.fields),
        )
        self.assertEqual("provisional", profile.review_status)
        date_of_birth = next(field for field in profile.fields if field.name == "date_of_birth")
        self.assertEqual("date", date_of_birth.value_type)
        self.assertLessEqual(date_of_birth.bounding_box.right, profile.canonical_width)
        self.assertLessEqual(date_of_birth.bounding_box.bottom, profile.canonical_height)
        self.assertIsNotNone(profile.qr_code)
        assert profile.qr_code is not None
        self.assertLessEqual(profile.qr_code.bounding_box.right, profile.canonical_width)
        self.assertLessEqual(profile.qr_code.bounding_box.bottom, profile.canonical_height)

    def test_registry_resolves_document_and_side(self) -> None:
        profile = load_default_profile()
        registry = ProfileRegistry((profile,))
        self.assertIs(profile, registry.resolve("ao_id_card", "front"))


class IntakeTests(unittest.TestCase):
    def test_array_is_copied_normalized_and_made_read_only(self) -> None:
        source = np.zeros((20, 30, 4), dtype=np.uint8)
        source[:, :, 1] = 40
        original = source.copy()
        result = ImageIntake().load(source, processing_id="test")
        self.assertEqual((20, 30, 3), result.image.shape)
        self.assertFalse(result.image.flags.writeable)
        self.assertTrue(np.array_equal(source, original))

    def test_decodes_png_bytes(self) -> None:
        image = Image.new("RGB", (30, 20), "white")
        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        result = ImageIntake().load(buffer.getvalue(), processing_id="test")
        self.assertEqual("png", result.source_format)
        self.assertEqual((30, 20, 3), (result.width, result.height, result.channels))

    def test_rejects_unsupported_encoded_format(self) -> None:
        image = Image.new("RGB", (10, 10), "white")
        buffer = io.BytesIO()
        image.save(buffer, format="GIF")
        with self.assertRaisesRegex(IntakeError, "JPEG and PNG"):
            ImageIntake().load(buffer.getvalue())

    def test_rejects_excessive_pixel_count(self) -> None:
        source = np.zeros((20, 20, 3), dtype=np.uint8)
        intake = ImageIntake(IntakeLimits(max_pixels=100))
        with self.assertRaises(IntakeError) as raised:
            intake.load(source)
        self.assertEqual("IMAGE_TOO_LARGE", raised.exception.code)


if __name__ == "__main__":
    unittest.main()
