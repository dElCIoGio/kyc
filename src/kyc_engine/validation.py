from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from datetime import datetime

from .contracts import FieldDefinition, FieldStatus


@dataclass(frozen=True)
class ValueAssessment:
    normalized_value: str | None
    status: FieldStatus
    warnings: tuple[str, ...] = ()


class FieldValueProcessor:
    def assess(self, field: FieldDefinition, raw_value: str) -> ValueAssessment:
        normalized = self.normalize(field.normalizer, raw_value)
        return self.validate(field.validator, normalized)

    def comparison_key(self, field: FieldDefinition, raw_value: str) -> str:
        normalized = unicodedata.normalize("NFKC", raw_value)
        if field.comparison == "casefold_whitespace":
            return _collapse_whitespace(normalized).casefold()
        if field.comparison == "alphanumeric_upper":
            return "".join(character for character in normalized.upper() if character.isalnum())
        if field.comparison == "date":
            return self.normalize("date", normalized) or _collapse_whitespace(normalized)
        raise KeyError(f"Unknown comparison strategy: {field.comparison}")

    def normalize(self, strategy: str, value: str) -> str | None:
        normalized = unicodedata.normalize("NFKC", value)
        if strategy in {"text", "name"}:
            return _collapse_whitespace(normalized)
        if strategy == "residence":
            return _strip_leading_label(normalized, "residência")
        if strategy == "place_of_birth":
            return _strip_leading_label(normalized, "natural de")
        if strategy == "document_id":
            return re.sub(r"\s+", "", normalized).upper()
        if strategy == "enum":
            return _collapse_whitespace(normalized).upper()
        if strategy == "date":
            compact = _collapse_whitespace(normalized)
            for date_format in ("%d/%m/%Y", "%d-%m-%Y", "%d.%m.%Y"):
                try:
                    return datetime.strptime(compact, date_format).date().isoformat()
                except ValueError:
                    continue
            return None
        raise KeyError(f"Unknown normalizer: {strategy}")

    def validate(self, strategy: str, value: str | None) -> ValueAssessment:
        if value is None or not value:
            return ValueAssessment(value, FieldStatus.INVALID, ("Value is empty or could not be normalized",))
        if strategy == "non_empty":
            return ValueAssessment(value, FieldStatus.VALID)
        if strategy == "name":
            valid_characters = all(character.isalpha() or character in " '-" for character in value)
            if len(value) < 2 or not any(character.isalpha() for character in value) or not valid_characters:
                return ValueAssessment(value, FieldStatus.INVALID, ("Value does not match a conservative name format",))
            return ValueAssessment(value, FieldStatus.VALID)
        if strategy == "document_id_basic":
            if not 5 <= len(value) <= 30 or not all(character.isalnum() or character in "-/" for character in value):
                return ValueAssessment(value, FieldStatus.INVALID, ("Value does not match the basic document-id format",))
            return ValueAssessment(value, FieldStatus.VALID, ("Only basic document-id format validation was applied",))
        if strategy == "date":
            return ValueAssessment(value, FieldStatus.VALID)
        if strategy == "enum":
            return ValueAssessment(value, FieldStatus.VALID, ("No profile-specific enum set was configured",))
        raise KeyError(f"Unknown validator: {strategy}")


def _collapse_whitespace(value: str) -> str:
    return " ".join(value.split())


def _strip_leading_label(value: str, label: str) -> str:
    pattern = rf"^\s*{re.escape(label)}\s*:\s*"
    return _collapse_whitespace(re.sub(pattern, "", value, count=1, flags=re.IGNORECASE))
