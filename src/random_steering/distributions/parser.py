from __future__ import annotations

import re
from dataclasses import dataclass

_NUMBER_PATTERN = re.compile(r"^[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?$")


@dataclass(frozen=True)
class ParsedNumber:
    raw: str
    valid: bool
    value: float | None
    error_code: str | None


def parse_single_number(text: str) -> ParsedNumber:
    token = text.strip()
    if not token:
        return ParsedNumber(raw=text, valid=False, value=None, error_code="empty")
    if not _NUMBER_PATTERN.fullmatch(token):
        return ParsedNumber(raw=text, valid=False, value=None, error_code="malformed")
    return ParsedNumber(raw=text, valid=True, value=float(token), error_code=None)


def parse_batch_numbers(text: str) -> list[ParsedNumber]:
    raw_parts = text.strip().split(",")
    parsed: list[ParsedNumber] = []
    for part in raw_parts:
        parsed.append(parse_single_number(part))
    return parsed
