"""
Authoritative Feature Registry for Phase 3.

Maps every computed feature name to:
- Group ID (G1 through G6)
- Finding Citation Tag ([A1] - [F3] per Whitepaper §10)
- Mathematical Definition / Formula
- Expected Value / Modeling Role
"""

from dataclasses import asdict, dataclass
from typing import Dict, List, Optional


@dataclass
class FeatureMeta:
    name: str
    group: str  # "G1", "G2", "G3", "G4", "G5", "G6"
    citation: str  # e.g. "[D1, D2]"
    description: str
    formula: str


class FeatureRegistry:
    """Central registry tracking all active features across G1-G6."""

    def __init__(self):
        self._registry: Dict[str, FeatureMeta] = {}

    def register(self, name: str, group: str, citation: str, description: str, formula: str) -> FeatureMeta:
        meta = FeatureMeta(
            name=name,
            group=group,
            citation=citation,
            description=description,
            formula=formula,
        )
        self._registry[name] = meta
        return meta

    def get(self, name: str) -> Optional[FeatureMeta]:
        return self._registry.get(name)

    def list_features(self, group: Optional[str] = None) -> List[FeatureMeta]:
        if group is not None:
            return [f for f in self._registry.values() if f.group == group]
        return list(self._registry.values())

    def feature_names(self, group: Optional[str] = None) -> List[str]:
        if group is not None:
            return [name for name, f in self._registry.items() if f.group == group]
        return list(self._registry.keys())

    def count(self, group: Optional[str] = None) -> int:
        return len(self.feature_names(group))

    def summary(self) -> Dict[str, int]:
        groups = ["G1", "G2", "G3", "G4", "G5", "G6"]
        return {g: self.count(g) for g in groups}


# Global Registry Instance
REGISTRY = FeatureRegistry()
