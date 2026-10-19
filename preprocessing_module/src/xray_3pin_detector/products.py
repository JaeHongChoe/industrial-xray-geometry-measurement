"""Product-group classification used by the inference pipeline."""

from __future__ import annotations


PRODUCT_GROUPS: dict[str, list[str]] = {
    "JIG미조립": ["TYPE_B-1", "TYPE_A-U", "TYPE_A-1-G002"],
    "TYPE_C": ["TYPE_C"],
    "TYPE_A-1": ["TYPE_A-1"],
    "TYPE_A_미조립": [
        "TYPE_A-G033",
        "TYPE_A-G034",
        "TYPE_A-G035",
        "TYPE_A-G038",
        "TYPE_A-G039",
        "TYPE_A-G041",
        "TYPE_A-G042",
        "TYPE_A-G043",
        "TYPE_A-G044",
        "TYPE_A-G045",
        "TYPE_A-G046",
        "TYPE_A-G053",
        "TYPE_A-G054",
        "TYPE_A-G055",
    ],
    "TYPE_A_조립": [
        "TYPE_A-G062",
        "TYPE_A-G063",
        "TYPE_A-G064",
        "TYPE_A-G065",
        "TYPE_A-G066",
        "TYPE_A-G067",
        "TYPE_A-G068",
        "TYPE_A-G069",
        "TYPE_A-G070",
        "TYPE_A-G071",
        "TYPE_A-G072",
        "TYPE_A-G076",
        "TYPE_A-G078",
        "TYPE_A-G079",
    ],
    "TYPE_D": ["TYPE_D"],
}


def classify_group(name: str) -> str:
    """Classify an image stem into a product group by exact or prefix match."""
    for group, prefixes in PRODUCT_GROUPS.items():
        for prefix in prefixes:
            if name == prefix or name.startswith(prefix):
                return group
    return "unknown"
