"""Per-run ablation switches — toggle context-engineering signals via state["ablation"] without restart."""
from __future__ import annotations

import config

# ablation key → config default attribute name
_DEFAULTS: dict[str, str] = {
    "metadata_filter":       "METADATA_FILTER_ENABLED",
    "rerank_boost":          "RERANK_BOOST_ENABLED",
    "memory":                "MEMORY_ENABLED",
    "reuse_gate":            "REUSE_GATE_ENABLED",
    "outcome_modulator":     "OUTCOME_MODULATOR_ENABLED",
    "metadata_in_generator": "METADATA_IN_GENERATOR_ENABLED",
}


def enabled(state, key: str) -> bool:
    if key not in _DEFAULTS:
        raise KeyError(f"unknown ablation key: {key!r} (known: {sorted(_DEFAULTS)})")
    overrides = state.get("ablation") or {}
    if key in overrides:
        return bool(overrides[key])
    return bool(getattr(config, _DEFAULTS[key]))
