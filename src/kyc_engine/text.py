from __future__ import annotations

"""Conservative handling for text returned by external decoders."""

import unicodedata


def repair_utf8_mojibake(value: str) -> str:
    """Repair a high-confidence single-byte decoding of UTF-8, if present.

    This is deliberately not a general encoding detector.  A value is changed
    only when re-encoding it through a single-byte codec produces strict UTF-8
    and the representation contains structural mojibake evidence which is
    removed by that conversion.  Otherwise the original Unicode string is
    preserved exactly.
    """
    if not value or _mojibake_score(value) == 0:
        return value

    original_score = _mojibake_score(value)
    candidates: list[str] = []
    for encoding in ("latin-1", "cp1252"):
        try:
            repaired = value.encode(encoding).decode("utf-8")
        except (UnicodeEncodeError, UnicodeDecodeError):
            continue
        if repaired == value or _has_unsafe_characters(repaired):
            continue
        if _mojibake_score(repaired) < original_score:
            candidates.append(repaired)

    if not candidates:
        return value
    return min(candidates, key=_mojibake_score)


def _mojibake_score(value: str) -> int:
    """Count byte-shaped UTF-8 sequences embedded in decoded text."""
    score = sum(2 for character in value if "\u0080" <= character <= "\u009f")
    for index, character in enumerate(value[:-1]):
        if _single_byte_value(character) in range(0xC2, 0xF5) and _is_continuation(value[index + 1]):
            score += 2
    return score


def _single_byte_value(character: str) -> int | None:
    for encoding in ("latin-1", "cp1252"):
        try:
            encoded = character.encode(encoding)
        except UnicodeEncodeError:
            continue
        if len(encoded) == 1:
            return encoded[0]
    return None


def _is_continuation(character: str) -> bool:
    value = _single_byte_value(character)
    return value is not None and 0x80 <= value <= 0xBF


def _has_unsafe_characters(value: str) -> bool:
    return "\ufffd" in value or any(
        unicodedata.category(character) == "Cc" and character not in "\t\n\r"
        for character in value
    )


__all__ = ["repair_utf8_mojibake"]
