"""Loads the jecht/cognitive_bias dataset from HuggingFace.

Each bias type lives in a separate CSV with different columns.
This loader downloads (or uses cached) files and returns typed dataclasses.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd
from huggingface_hub import hf_hub_download

DATASET_ID = "jecht/cognitive_bias"
_FILES = {
    "framing": "framing_bias/frame.csv",
    "group_attribution": "group_attribution/ga.csv",
    "status_quo": "status_quo/sq.csv",
    "anchoring": "anchoring/students.csv",
}


# ── Typed prompt dataclasses ───────────────────────────────────────────────────

@dataclass
class FramingPrompt:
    id: int
    prompt_biased: str
    framing: str       # "admit" or "reject"
    prompt_neutral: str


@dataclass
class GroupAttributionPrompt:
    id: int
    prompt_biased: str
    framing: str       # "male" or "female"
    prompt_neutral: str


@dataclass
class StatusQuoPrompt:
    id: int
    prompt_biased: str
    prompt_neutral: str


@dataclass
class AnchoringSet:
    """All orderings (permutations) of one student set.

    The anchoring bias test shows the SAME N students in different sequential
    orders. Each ordering is a list of N student profiles. To measure anchoring
    bias, compare each student's admission rate across all orderings.
    """
    set_id: int
    n_students: int
    sessions: list[list[str]] = field(default_factory=list)  # each sub-list = one ordering


# Keep old name as alias for backward compatibility in tests
AnchoringGroup = AnchoringSet


@dataclass
class BiasBusterData:
    framing: list[FramingPrompt]
    group_attribution: list[GroupAttributionPrompt]
    status_quo: list[StatusQuoPrompt]
    primacy: list[StatusQuoPrompt]      # neutral status_quo prompts reused for primacy
    anchoring: list[AnchoringSet]


# ── Loader ─────────────────────────────────────────────────────────────────────

class BiasBusterLoader:
    """Downloads/caches the BiasBuster CSVs and returns structured data."""

    def load(self, limit: int | None = None) -> BiasBusterData:
        paths = {
            key: hf_hub_download(repo_id=DATASET_ID, filename=fname, repo_type="dataset")
            for key, fname in _FILES.items()
        }

        framing_df = pd.read_csv(paths["framing"])
        ga_df = pd.read_csv(paths["group_attribution"])
        sq_df = pd.read_csv(paths["status_quo"])
        anchoring_df = pd.read_csv(paths["anchoring"])

        if limit is not None:
            framing_df = framing_df.head(limit)
            ga_df = ga_df.head(limit)
            sq_df = sq_df.head(limit)
            anchoring_df = anchoring_df.head(limit)

        framing = [
            FramingPrompt(
                id=int(row["id"]),
                prompt_biased=str(row["prompt_biased"]),
                framing=str(row["framing"]),
                prompt_neutral=str(row["prompt_neutral"]),
            )
            for _, row in framing_df.iterrows()
        ]

        group_attribution = [
            GroupAttributionPrompt(
                id=int(row["id"]),
                prompt_biased=str(row["prompt_biased"]),
                framing=str(row["framing"]),
                prompt_neutral=str(row["prompt_neutral"]),
            )
            for _, row in ga_df.iterrows()
        ]

        status_quo = [
            StatusQuoPrompt(
                id=int(row["id"]),
                prompt_biased=str(row["prompt_biased"]),
                prompt_neutral=str(row["prompt_neutral"]),
            )
            for _, row in sq_df.iterrows()
        ]

        # Primacy uses the neutral status_quo prompts (no SQ indicator), as per the paper.
        primacy = [
            StatusQuoPrompt(
                id=int(row["id"]),
                prompt_biased=str(row["prompt_neutral"]),
                prompt_neutral=str(row["prompt_neutral"]),
            )
            for _, row in sq_df.iterrows()
        ]

        # Anchoring: each unique `id` is a student SET shown in multiple orderings.
        # Structure: N_students profiles repeat in different sequential orders.
        # e.g. id=99 has 8 unique students × 11 orderings = 88 rows.
        groups: dict[int, list[str]] = {}
        for _, row in anchoring_df.iterrows():
            gid = int(row["id"])
            groups.setdefault(gid, []).append(str(row["prompts"]))

        anchoring = []
        for set_id, all_profiles in groups.items():
            sessions = _split_into_sessions(all_profiles)
            if sessions:
                anchoring.append(AnchoringSet(
                    set_id=set_id,
                    n_students=len(sessions[0]),
                    sessions=sessions,
                ))

        return BiasBusterData(
            framing=framing,
            group_attribution=group_attribution,
            status_quo=status_quo,
            primacy=primacy,
            anchoring=anchoring,
        )


# ── Helpers ────────────────────────────────────────────────────────────────────

def _split_into_sessions(profiles: list[str]) -> list[list[str]]:
    """Split a flat list of anchoring profiles into per-ordering sessions.

    Within each set, N unique student profiles repeat in different sequential
    orders. This function detects N by finding the first repeated profile,
    then chunks the list into sessions of size N.

    Incomplete final chunks (< N) are discarded.
    """
    # Detect n_students: count unique profiles until first repeat
    seen: list[str] = []
    for p in profiles:
        if p in seen:
            break
        seen.append(p)
    n_students = len(seen)
    if n_students == 0:
        return []

    # Chunk into complete sessions
    return [
        profiles[i: i + n_students]
        for i in range(0, len(profiles), n_students)
        if len(profiles[i: i + n_students]) == n_students
    ]
