"""Compatibility exports moved to ``play.views``.

This module stays so existing Bazel deps on ``//replay:recorder`` keep working.
"""

from __future__ import annotations

from play.views import action_detail as _action_detail
from play.views import batched_to_snapshot, cards_table, nobles_table

__all__ = [
    "_action_detail",
    "batched_to_snapshot",
    "cards_table",
    "nobles_table",
]
