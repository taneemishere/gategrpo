from __future__ import annotations

from .models import CandidateRecord, EvidencePacket


ROUTES = {
    "scope_guard": "scope_repair",
    "patch_applies": "patch_format_repair",
    "patch_apply_runtime": "patch_format_repair",
    "python_ast": "syntax_repair",
    "secret_scan": "safety_repair",
    "visible_tests": "behavior_repair",
    "release_gate_regressions": "regression_repair",
}

# The repair routes the controller understands. Used to validate metadata a
# generator declares about its own candidate so a model cannot inject routes
# outside the known taxonomy.
KNOWN_ROUTES = frozenset(ROUTES.values()) | {"general_repair"}


class CleanContextCritic:
    def route_next(self, records: list[CandidateRecord], evidence: EvidencePacket | None) -> str:
        if evidence is None:
            return "promote_or_stop"
        if len(records) >= 2 and records[-1].failure_type == records[-2].failure_type:
            return f"diversify_from_{ROUTES.get(evidence.gate, 'general_repair')}"
        return ROUTES.get(evidence.gate, "general_repair")
