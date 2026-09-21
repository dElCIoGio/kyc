"""Smoke tests for field_extractor.py using a fake OCREngine (no PaddleOCR needed)."""

import io

import numpy as np
from PIL import Image

from field_extractor import (
    BoundingBox,
    FieldDefinition,
    FieldExtractor,
    ImageVariant,
    OCREngine,
    OCRLine,
    OCRReadResult,
)


class FakeOCREngine(OCREngine):
    """Returns scripted results based on the crop's mean pixel value, so
    different bounding boxes on our synthetic image yield different text."""

    def __init__(self, confidence_strategy="min"):
        self._confidence_strategy = confidence_strategy
        self.calls = []

    def read_text(self, image: np.ndarray) -> OCRReadResult:
        self.calls.append(image.shape)
        mean_val = float(image.mean())
        if mean_val > 200:  # bright/white crop -> "empty" field
            return OCRReadResult()
        if mean_val > 100:
            return OCRReadResult(lines=(OCRLine("SINGLE_LINE", 0.9),))
        return OCRReadResult(
            lines=(OCRLine("LINE ONE", 0.95), OCRLine("line two", 0.4))
        )


class RaisingOCREngine(OCREngine):
    def read_text(self, image: np.ndarray) -> OCRReadResult:
        raise RuntimeError("engine exploded")


def make_test_image() -> np.ndarray:
    # 100x300 image: left third white, middle third mid-gray, right third dark
    img = np.full((100, 300, 3), 255, dtype=np.uint8)
    img[:, 100:200] = 150
    img[:, 200:300] = 30
    return img


def test_basic_extraction_and_confidence_min():
    image = make_test_image()
    fields = [
        FieldDefinition(name="empty_field", bounding_box=BoundingBox(0, 0, 100, 100)),
        FieldDefinition(name="single_line_field", bounding_box=BoundingBox(100, 0, 100, 100)),
        FieldDefinition(name="multi_line_field", bounding_box=BoundingBox(200, 0, 100, 100), multi_line=True),
    ]
    engine = FakeOCREngine(confidence_strategy="min")
    extractor = FieldExtractor(field_definitions=fields, ocr_engine=engine)

    result = extractor.extract_fields(image, variant_id="v1")
    assert result.variant_id == "v1"

    empty = result.get("empty_field")
    assert empty.value == "" and empty.confidence == 0.0 and empty.error is None

    single = result.get("single_line_field")
    assert single.value == "SINGLE_LINE" and single.confidence == 0.9

    multi = result.get("multi_line_field")
    assert multi.value == "LINE ONE\nline two"  # joined with newline
    assert multi.confidence == 0.4  # min strategy -> weakest line wins

    print("test_basic_extraction_and_confidence_min: PASS")


def test_confidence_mean_strategy():
    image = make_test_image()
    fields = [FieldDefinition(name="multi_line_field", bounding_box=BoundingBox(200, 0, 100, 100), multi_line=True)]
    engine = FakeOCREngine(confidence_strategy="mean")
    extractor = FieldExtractor(field_definitions=fields, ocr_engine=engine)
    result = extractor.extract_fields(image)
    multi = result.get("multi_line_field")
    assert abs(multi.confidence - ((0.95 + 0.4) / 2)) < 1e-9
    print("test_confidence_mean_strategy: PASS")


def test_single_line_join_uses_space():
    # force a region that returns two lines but field is NOT multi_line -> should join with space
    image = make_test_image()
    fields = [FieldDefinition(name="f", bounding_box=BoundingBox(200, 0, 100, 100), multi_line=False)]
    engine = FakeOCREngine()
    extractor = FieldExtractor(field_definitions=fields, ocr_engine=engine)
    result = extractor.extract_fields(image)
    assert result.get("f").value == "LINE ONE line two"
    print("test_single_line_join_uses_space: PASS")


def test_out_of_bounds_box_produces_error_not_crash():
    image = make_test_image()
    fields = [
        FieldDefinition(name="ok_field", bounding_box=BoundingBox(0, 0, 50, 50)),
        FieldDefinition(name="oob_field", bounding_box=BoundingBox(1000, 1000, 50, 50)),
    ]
    engine = FakeOCREngine()
    extractor = FieldExtractor(field_definitions=fields, ocr_engine=engine)
    result = extractor.extract_fields(image)

    oob = result.get("oob_field")
    assert oob.error is not None
    assert oob.confidence == 0.0
    assert not oob.succeeded

    ok = result.get("ok_field")
    assert ok.succeeded
    print("test_out_of_bounds_box_produces_error_not_crash: PASS")


def test_engine_exception_isolated_per_field():
    image = make_test_image()
    fields = [FieldDefinition(name="f1", bounding_box=BoundingBox(0, 0, 50, 50))]
    extractor = FieldExtractor(field_definitions=fields, ocr_engine=RaisingOCREngine())
    result = extractor.extract_fields(image)
    f1 = result.get("f1")
    assert f1.error == "engine exploded"
    assert f1.confidence == 0.0
    print("test_engine_exception_isolated_per_field: PASS")


def test_batch_extraction_multiple_variants():
    fields = [FieldDefinition(name="single_line_field", bounding_box=BoundingBox(100, 0, 100, 100))]
    engine = FakeOCREngine()
    extractor = FieldExtractor(field_definitions=fields, ocr_engine=engine)

    variants = [
        ImageVariant(variant_id="orig", image=make_test_image()),
        ImageVariant(variant_id="enhanced", image=make_test_image()),
    ]
    results = extractor.extract_fields_batch(variants)
    assert [r.variant_id for r in results] == ["orig", "enhanced"]
    assert all(r.get("single_line_field").value == "SINGLE_LINE" for r in results)
    print("test_batch_extraction_multiple_variants: PASS")


def test_image_source_variants_all_load():
    # numpy array
    arr = make_test_image()
    # PIL Image
    pil_img = Image.fromarray(arr)
    # bytes
    buf = io.BytesIO()
    pil_img.save(buf, format="PNG")
    img_bytes = buf.getvalue()
    # file path
    path = "/tmp/test_field_extractor_image.png"
    pil_img.save(path)

    fields = [FieldDefinition(name="single_line_field", bounding_box=BoundingBox(100, 0, 100, 100))]
    engine = FakeOCREngine()
    extractor = FieldExtractor(field_definitions=fields, ocr_engine=engine)

    for source in (arr, pil_img, img_bytes, path):
        result = extractor.extract_fields(source)
        assert result.get("single_line_field").value == "SINGLE_LINE", f"failed for {type(source)}"
    print("test_image_source_variants_all_load: PASS")


def test_duplicate_field_names_rejected():
    fields = [
        FieldDefinition(name="dup", bounding_box=BoundingBox(0, 0, 10, 10)),
        FieldDefinition(name="dup", bounding_box=BoundingBox(10, 10, 10, 10)),
    ]
    try:
        FieldExtractor(field_definitions=fields, ocr_engine=FakeOCREngine())
        assert False, "expected ValueError"
    except ValueError as e:
        assert "dup" in str(e)
    print("test_duplicate_field_names_rejected: PASS")


def test_bounding_box_validation():
    try:
        BoundingBox(x=0, y=0, width=0, height=10)
        assert False
    except ValueError:
        pass
    try:
        BoundingBox(x=-1, y=0, width=10, height=10)
        assert False
    except ValueError:
        pass
    print("test_bounding_box_validation: PASS")


def test_to_dict_serialization():
    image = make_test_image()
    fields = [FieldDefinition(name="single_line_field", bounding_box=BoundingBox(100, 0, 100, 100))]
    extractor = FieldExtractor(field_definitions=fields, ocr_engine=FakeOCREngine())
    result = extractor.extract_fields(image, variant_id="v1")
    d = result.to_dict()
    assert d["variant_id"] == "v1"
    assert d["fields"]["single_line_field"]["value"] == "SINGLE_LINE"
    print("test_to_dict_serialization: PASS")


if __name__ == "__main__":
    test_basic_extraction_and_confidence_min()
    test_confidence_mean_strategy()
    test_single_line_join_uses_space()
    test_out_of_bounds_box_produces_error_not_crash()
    test_engine_exception_isolated_per_field()
    test_batch_extraction_multiple_variants()
    test_image_source_variants_all_load()
    test_duplicate_field_names_rejected()
    test_bounding_box_validation()
    test_to_dict_serialization()
    print("\nAll tests passed.")