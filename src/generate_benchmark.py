#!/usr/bin/env python3
"""Create a deterministic benchmark for ZFR prioritisation experiments."""

from __future__ import annotations

import csv
import random
from pathlib import Path


OUT = Path("data/sample_candidates.csv")
RNG = random.Random(20260525)
CHROMOSOMES = ["chr1", "chr2", "chr3", "chr4", "chr5", "chrX"]
EVIDENCE_TYPES = ["experimental", "cross_tool", "predicted_motif", "model_only"]


def clamp(value: float, low: float = 0.0, high: float = 100.0) -> float:
    return max(low, min(high, value))


def tss_score(distance: float) -> float:
    distance = abs(distance)
    if distance <= 250:
        return 100.0
    if distance <= 1_000:
        return 90.0 - (distance - 250.0) * (20.0 / 750.0)
    if distance <= 5_000:
        return 70.0 - (distance - 1_000.0) * (35.0 / 4_000.0)
    if distance <= 20_000:
        return 35.0 - (distance - 5_000.0) * (25.0 / 15_000.0)
    return 0.0


def trapezoid(value: float, a: float, b: float, c: float, d: float) -> float:
    if a <= value < b:
        return 100.0 * (value - a) / (b - a)
    if b <= value <= c:
        return 100.0
    if c < value <= d:
        return 100.0 * (d - value) / (d - c)
    return 0.0


def choose_evidence(index: int) -> str:
    pattern = ["experimental", "cross_tool", "predicted_motif", "model_only", "cross_tool", "model_only", "experimental", "predicted_motif"]
    return pattern[index % len(pattern)]


def make_candidate(index: int) -> dict[str, object]:
    evidence_type = choose_evidence(index)
    chromosome = CHROMOSOMES[index % len(CHROMOSOMES)]
    length_bp = RNG.randint(8, 400)
    start = 80_000 + index * 4_819 + RNG.randint(0, 2_500)
    end = start + length_bp

    if evidence_type == "experimental":
        zdna_score = RNG.uniform(0.42, 0.88)
        z_dnabert = clamp(zdna_score * 100.0 + RNG.gauss(3.0, 9.0)) / 100.0
        tss_distance = int(abs(RNG.gauss(950, 1_400)))
        promoter = RNG.random() < 0.72
        cpx = clamp(RNG.gauss(56, 18))
        repeat = clamp(RNG.gauss(15, 10))
        independent = RNG.random() < 0.36
    elif evidence_type == "cross_tool":
        zdna_score = RNG.uniform(0.55, 0.94)
        z_dnabert = clamp(zdna_score * 100.0 + RNG.gauss(0.0, 7.0)) / 100.0
        tss_distance = int(abs(RNG.gauss(1_600, 2_400)))
        promoter = RNG.random() < 0.62
        cpx = clamp(RNG.gauss(49, 20))
        repeat = clamp(RNG.gauss(21, 13))
        independent = RNG.random() < 0.72
    elif evidence_type == "predicted_motif":
        zdna_score = RNG.uniform(0.65, 0.97)
        z_dnabert = clamp(zdna_score * 100.0 + RNG.gauss(-18.0, 11.0)) / 100.0
        tss_distance = int(abs(RNG.gauss(8_200, 10_000)))
        promoter = RNG.random() < 0.35
        cpx = clamp(RNG.gauss(34, 23))
        repeat = clamp(RNG.gauss(43, 20))
        independent = RNG.random() < 0.22
    else:
        zdna_score = RNG.uniform(0.38, 0.82)
        z_dnabert = clamp(zdna_score * 100.0 + RNG.gauss(9.0, 13.0)) / 100.0
        tss_distance = int(abs(RNG.gauss(4_400, 6_000)))
        promoter = RNG.random() < 0.42
        cpx = clamp(RNG.gauss(38, 24))
        repeat = clamp(RNG.gauss(23, 15))
        independent = RNG.random() < 0.38

    regulatory_marks = min(3, max(0, int(round(RNG.gauss(1.3 + (1 if promoter else 0), 0.9)))))
    primer_uniqueness = clamp(RNG.gauss(86 - 0.42 * repeat + 0.04 * min(length_bp, 384), 9))

    zdna_scaled = zdna_score * 100.0
    model_probability = z_dnabert * 100.0
    consensus = clamp(100.0 - abs(zdna_scaled - model_probability))
    z_signal = clamp(0.46 * zdna_scaled + 0.54 * model_probability)
    regulatory_context = clamp(
        0.42 * tss_score(tss_distance)
        + 0.24 * cpx
        + (16.0 if promoter else 0.0)
        + min(24.0, 7.0 * regulatory_marks)
    )
    source_support = {"experimental": 93.0, "cross_tool": 80.0, "predicted_motif": 56.0, "model_only": 49.0}[evidence_type]
    source_bias = {"experimental": 10.0, "cross_tool": 24.0, "predicted_motif": 60.0, "model_only": 43.0}[evidence_type]
    evidence_support = clamp(0.68 * source_support + 0.22 * consensus + (10.0 if independent else 0.0))
    bias_risk = clamp(0.62 * repeat + 0.25 * source_bias + 0.13 * (100.0 - consensus))
    length_score = trapezoid(length_bp, 8.0, 8.0, 400.0, 450.0)
    validation_feasibility = clamp(0.50 * primer_uniqueness + 0.30 * length_score + 0.20 * (100.0 - repeat))

    expert = (
        0.25 * z_signal
        + 0.25 * regulatory_context
        + 0.22 * evidence_support
        + 0.14 * validation_feasibility
        + 0.04 * consensus
        - 0.18 * bias_risk
        + 12.0
    )
    if evidence_type == "experimental":
        expert += 4.0
    if evidence_type == "cross_tool":
        expert += 2.5
    if evidence_type == "predicted_motif" and repeat > 45:
        expert -= 10.0
    if evidence_type == "model_only" and not independent:
        expert -= 4.0
    if promoter and cpx > 55:
        expert += 3.5
    if tss_distance < 500:
        expert += 3.0
    if length_bp < 8 or length_bp > 400:
        expert -= 3.0
    expert = clamp(expert + RNG.gauss(0.0, 3.8))

    if index < 192:
        split = "train"
    elif index < 256:
        split = "validation"
    else:
        split = "test"

    return {
        "candidate_id": f"ZFR_{index + 1:04d}",
        "species": "generic_eukaryote",
        "chromosome": chromosome,
        "start": start,
        "end": end,
        "length_bp": length_bp,
        "evidence_type": evidence_type,
        "zdna_score": f"{zdna_score:.4f}",
        "z_dnabert_probability": f"{z_dnabert:.4f}",
        "cpx_overlap_pct": f"{cpx:.1f}",
        "tss_distance_bp": tss_distance,
        "promoter_overlap": str(promoter).lower(),
        "regulatory_marks": regulatory_marks,
        "repeat_overlap_pct": f"{repeat:.1f}",
        "independent_support": str(independent).lower(),
        "primer_uniqueness": f"{primer_uniqueness:.1f}",
        "split": split,
        "expert_priority_score": f"{expert:.1f}",
    }


def main() -> None:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    rows = [make_candidate(index) for index in range(320)]
    columns = list(rows[0].keys())
    with OUT.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} rows to {OUT}")


if __name__ == "__main__":
    main()
