from datetime import date

import pytest

from rag.retrieval.version_filters import (
    as_of_date_filter,
    current_version_filter,
    exact_version_filter,
    parse_as_of_date,
)


def test_current_version_filter_uses_document_family_and_explicit_current_flag() -> None:
    assert current_version_filter("POL-FIN-004") == {
        "$and": [
            {"document_family": {"$eq": "POL-FIN-004"}},
            {"is_current_version": {"$eq": True}},
        ]
    }


def test_historical_filter_uses_numeric_effective_and_expiry_dates() -> None:
    as_of = date(2025, 6, 1)

    assert as_of_date_filter("POL-FIN-004", as_of) == {
        "$and": [
            {"document_family": {"$eq": "POL-FIN-004"}},
            {"effective_date_ordinal": {"$lte": as_of.toordinal()}},
            {"expiry_date_ordinal": {"$gte": as_of.toordinal()}},
        ]
    }
    assert exact_version_filter("POL-FIN-004", "1.4")["$and"][1] == {
        "version": {"$eq": "1.4"}
    }


def test_parse_as_of_date_requires_iso_format() -> None:
    assert parse_as_of_date("2025-06-01") == date(2025, 6, 1)
    with pytest.raises(ValueError, match="YYYY-MM-DD"):
        parse_as_of_date("01-06-2025")
