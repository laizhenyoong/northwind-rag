"""Build strict metadata filters for current and historical document versions."""

from __future__ import annotations

from datetime import date
from typing import Any


def current_version_filter(document_id: str) -> dict[str, Any]:
    """Restrict retrieval to the current revision in one document family."""
    return {
        "$and": [
            {"document_family": {"$eq": document_id}},
            {"is_current_version": {"$eq": True}},
        ]
    }


def exact_version_filter(document_id: str, version: str) -> dict[str, Any]:
    """Restrict retrieval to one explicitly requested historical version."""
    return {
        "$and": [
            {"document_family": {"$eq": document_id}},
            {"version": {"$eq": version}},
        ]
    }


def as_of_date_filter(document_id: str, as_of_date: date) -> dict[str, Any]:
    """Restrict retrieval to the version valid on the requested calendar date."""
    ordinal = as_of_date.toordinal()
    return {
        "$and": [
            {"document_family": {"$eq": document_id}},
            {"effective_date_ordinal": {"$lte": ordinal}},
            {"expiry_date_ordinal": {"$gte": ordinal}},
        ]
    }


def parse_as_of_date(value: str) -> date:
    """Parse the CLI's explicit ISO date safely."""
    try:
        return date.fromisoformat(value)
    except ValueError as error:
        raise ValueError("as_of_date must use ISO format YYYY-MM-DD") from error
