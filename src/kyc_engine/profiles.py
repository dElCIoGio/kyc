from __future__ import annotations

import json
from importlib import resources
from pathlib import Path
from typing import Any, Mapping

from .contracts import BoundingBox, DocumentProfile, FieldDefinition

SUPPORTED_OCR_MODES = {"single_line", "multiline"}
SUPPORTED_COMPARISONS = {"casefold_whitespace", "alphanumeric_upper", "date"}
SUPPORTED_NORMALIZERS = {"text", "name", "document_id", "date", "enum"}
SUPPORTED_VALIDATORS = {
    "non_empty",
    "name",
    "document_id_basic",
    "date",
    "enum",
}


class ProfileRegistry:
    def __init__(self, profiles: tuple[DocumentProfile, ...]) -> None:
        if not profiles:
            raise ValueError("At least one document profile is required")
        self._profiles: dict[str, DocumentProfile] = {}
        self._by_document: dict[tuple[str, str], DocumentProfile] = {}
        for profile in profiles:
            validate_profile(profile)
            if profile.profile_id in self._profiles:
                raise ValueError(f"Duplicate profile id: {profile.profile_id}")
            key = (profile.document_type, profile.side)
            if key in self._by_document:
                raise ValueError(f"Duplicate document profile for {key[0]}/{key[1]}")
            self._profiles[profile.profile_id] = profile
            self._by_document[key] = profile

    def get(self, profile_id: str) -> DocumentProfile:
        try:
            return self._profiles[profile_id]
        except KeyError as exc:
            raise KeyError(f"Unknown document profile: {profile_id}") from exc

    def resolve(self, document_type: str, side: str) -> DocumentProfile:
        try:
            return self._by_document[(document_type, side)]
        except KeyError as exc:
            raise KeyError(f"No profile for {document_type}/{side}") from exc

    @property
    def profile_ids(self) -> tuple[str, ...]:
        return tuple(self._profiles)


def load_profile(path: str | Path) -> DocumentProfile:
    profile_path = Path(path)
    try:
        raw = json.loads(profile_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Could not load document profile: {profile_path}") from exc
    return profile_from_mapping(raw)


def load_default_profile() -> DocumentProfile:
    profile_resource = resources.files("kyc_engine").joinpath(
        "configs/ao_id_card_front_v1.json"
    )
    raw = json.loads(profile_resource.read_text(encoding="utf-8"))
    return profile_from_mapping(raw)


def profile_from_mapping(raw: object) -> DocumentProfile:
    if not isinstance(raw, Mapping):
        raise TypeError("Document profile must be a mapping")
    fields_raw = _required(raw, "fields")
    if not isinstance(fields_raw, list):
        raise TypeError("Document profile fields must be a list")

    fields = tuple(_field_from_mapping(item) for item in fields_raw)
    profile = DocumentProfile(
        profile_id=_required_str(raw, "profile_id"),
        document_type=_required_str(raw, "document_type"),
        side=_required_str(raw, "side"),
        canonical_width=_required_int(raw, "canonical_width"),
        canonical_height=_required_int(raw, "canonical_height"),
        review_status=_required_str(raw, "review_status"),
        fields=fields,
    )
    validate_profile(profile)
    return profile


def validate_profile(profile: DocumentProfile) -> None:
    if not profile.profile_id.strip():
        raise ValueError("Profile id cannot be empty")
    if profile.canonical_width <= 0 or profile.canonical_height <= 0:
        raise ValueError("Canonical profile dimensions must be positive")
    if profile.review_status not in {"provisional", "reviewed"}:
        raise ValueError("Profile review_status must be provisional or reviewed")
    if not profile.fields:
        raise ValueError("Document profile must contain at least one field")

    names: set[str] = set()
    for item in profile.fields:
        if not item.name.strip():
            raise ValueError("Field name cannot be empty")
        if item.name in names:
            raise ValueError(f"Duplicate field name: {item.name}")
        names.add(item.name)
        if item.bounding_box.right > profile.canonical_width or item.bounding_box.bottom > profile.canonical_height:
            raise ValueError(f"Field '{item.name}' lies outside canonical dimensions")
        if item.padding < 0:
            raise ValueError(f"Field '{item.name}' padding cannot be negative")
        if item.ocr_mode not in SUPPORTED_OCR_MODES:
            raise ValueError(f"Unsupported OCR mode for '{item.name}': {item.ocr_mode}")
        if item.comparison not in SUPPORTED_COMPARISONS:
            raise ValueError(f"Unsupported comparison for '{item.name}': {item.comparison}")
        if item.normalizer not in SUPPORTED_NORMALIZERS:
            raise ValueError(f"Unsupported normalizer for '{item.name}': {item.normalizer}")
        if item.validator not in SUPPORTED_VALIDATORS:
            raise ValueError(f"Unsupported validator for '{item.name}': {item.validator}")


def _field_from_mapping(raw: object) -> FieldDefinition:
    if not isinstance(raw, Mapping):
        raise TypeError("Each field definition must be a mapping")
    return FieldDefinition(
        name=_required_str(raw, "name"),
        bounding_box=BoundingBox(
            x=_required_int(raw, "x"),
            y=_required_int(raw, "y"),
            width=_required_int(raw, "width"),
            height=_required_int(raw, "height"),
        ),
        required=_optional_bool(raw, "required", True),
        value_type=str(raw.get("value_type", "text")),
        multiline=_optional_bool(raw, "multiline", False),
        padding=_optional_int(raw, "padding", 0),
        ocr_mode=str(raw.get("ocr_mode", "single_line")),
        comparison=str(raw.get("comparison", "casefold_whitespace")),
        normalizer=str(raw.get("normalizer", "text")),
        validator=str(raw.get("validator", "non_empty")),
    )


def _required(raw: Mapping[str, Any], name: str) -> Any:
    if name not in raw:
        raise ValueError(f"Document profile is missing required key '{name}'")
    return raw[name]


def _required_str(raw: Mapping[str, Any], name: str) -> str:
    value = _required(raw, name)
    if not isinstance(value, str) or not value.strip():
        raise TypeError(f"Profile key '{name}' must be a non-empty string")
    return value


def _required_int(raw: Mapping[str, Any], name: str) -> int:
    value = _required(raw, name)
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"Profile key '{name}' must be an integer")
    return value


def _optional_int(raw: Mapping[str, Any], name: str, default: int) -> int:
    value = raw.get(name, default)
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"Profile key '{name}' must be an integer")
    return value


def _optional_bool(raw: Mapping[str, Any], name: str, default: bool) -> bool:
    value = raw.get(name, default)
    if not isinstance(value, bool):
        raise TypeError(f"Profile key '{name}' must be a boolean")
    return value
