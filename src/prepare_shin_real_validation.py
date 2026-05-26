#!/usr/bin/env python3
"""Prepare a real per-locus Shin validation set from available overlay files."""

from __future__ import annotations

import csv
from bisect import bisect_left
from pathlib import Path
from typing import Dict, Iterable, List, Set, Tuple


DATA_DIR = Path("data/shin_data")
OUT_PATH = Path("data/shin_real_candidates.csv")
SOURCE_SUMMARY_PATH = Path("data/shin_real_selected_sources.csv")
TSS_PATH = DATA_DIR / "Hum_ENSEMBL52.sga"
ORIGINAL_FASTA_DIR = DATA_DIR / "shin_original_fasta"
CPX_DIR = DATA_DIR / "cpxhunter_exports"
CPX_TYPES = ("cpa", "cpc", "cpg", "cpt")

HUNTER_SOURCES = [
    ("hunter_model1_10bp", "model1", "Z-DNA Hunter model 1 size 10", DATA_DIR / "zfs_yes_model1_10bp.csv", DATA_DIR / "zfs_no_model1_10bp.csv", 0.15),
    ("hunter_model2_10bp", "model2", "Z-DNA Hunter model 2 size 10", DATA_DIR / "zfs_yes_model2_10bp.csv", DATA_DIR / "zfs_no_model2_10bp.csv", 0.10),
    ("hunter_model2_8bp", "model2", "Z-DNA Hunter model 2 size 8", DATA_DIR / "zfs_yes_model2_8bp.csv", DATA_DIR / "zfs_no_model2_8bp.csv", 0.55),
    ("hunter_model2_8bp_65perc", "model2", "Z-DNA Hunter model 2 size 8 score 65%", DATA_DIR / "zfs_yes_model2_8bp_65perc.csv", DATA_DIR / "zfs_no_model2_8bp_65perc.csv", 0.10),
    ("hunter_model2_10bp_60perc", "model2", "Z-DNA Hunter model 2 size 10 score 60%", DATA_DIR / "zfs_yes_model2_10bp_60perc.csv", DATA_DIR / "zfs_no_model2_10bp_60perc.csv", 0.10),
]


def clamp(value: float, low: float = 0.0, high: float = 100.0) -> float:
    return max(low, min(high, value))


def read_csv(path: Path) -> List[Dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def base_key(label: str, row: Dict[str, str]) -> Tuple[str, str]:
    return label, row["fasta_id"]


def to_int(value: str) -> int:
    return int(round(float(value)))


def to_float(value: str) -> float:
    return float(value or 0.0)


def normalize_chromosome(value: object) -> str:
    text = str(value).strip()
    return text if text.startswith("chr") else f"chr{text}"


def source_detection(row: Dict[str, str]) -> bool:
    if "ZFS" in row and str(row.get("ZFS", "")).strip() != "":
        return str(row["ZFS"]).strip() in {"1", "1.0", "true", "True"}
    return to_float(row.get("overlap_bp", "0")) > 0.0


def parse_shin_fasta_header(header: str) -> Tuple[str, str, int, int]:
    fields = header.strip().lstrip(">").split("_")
    if len(fields) < 5:
        raise ValueError(f"Unexpected FASTA header: {header}")
    fasta_id = "_".join(fields[:2])
    chrom = "_".join(fields[2:-2])
    start = to_int(fields[-2])
    end = to_int(fields[-1])
    return fasta_id, chrom, start, end


def read_original_shin_intervals(path: Path) -> Dict[Tuple[str, str], Tuple[str, int, int]]:
    intervals: Dict[Tuple[str, str], Tuple[str, int, int]] = {}
    for label, file_name in [("positive", "ZFS_yes.fasta"), ("negative", "ZFS_no.fasta")]:
        fasta_path = path / file_name
        with fasta_path.open(encoding="utf-8") as handle:
            for line in handle:
                if not line.startswith(">"):
                    continue
                fasta_id, chrom, start, end = parse_shin_fasta_header(line)
                intervals[(label, fasta_id)] = (chrom, start, end)
    return intervals


def read_tss_index(path: Path) -> Dict[str, List[int]]:
    index: Dict[str, List[int]] = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            fields = line.rstrip("\n").split("\t")
            if len(fields) < 3 or fields[1] != "TSS":
                continue
            chrom = normalize_chromosome(fields[0])
            index.setdefault(chrom, []).append(to_int(fields[2]))
    for positions in index.values():
        positions.sort()
    return index


def distance_to_interval(position: int, start: int, end: int) -> int:
    if start <= position <= end:
        return 0
    if position < start:
        return start - position
    return position - end


def nearest_tss_distance(tss_index: Dict[str, List[int]], chrom: object, start: int, end: int) -> int:
    positions = tss_index.get(normalize_chromosome(chrom), [])
    if not positions:
        return 50_000
    insertion = bisect_left(positions, start)
    nearby = []
    if insertion < len(positions):
        nearby.append(positions[insertion])
    if insertion > 0:
        nearby.append(positions[insertion - 1])
    if insertion + 1 < len(positions):
        nearby.append(positions[insertion + 1])
    return min(distance_to_interval(position, start, end) for position in nearby)


def read_cpx_index(path: Path, chromosomes: Set[str]) -> Dict[str, Dict[str, Dict[str, object]]]:
    index: Dict[str, Dict[str, Dict[str, object]]] = {}
    for chrom in sorted(chromosomes):
        chrom_index: Dict[str, Dict[str, object]] = {}
        for cpx_type in CPX_TYPES:
            cpx_path = path / chrom / cpx_type / f"hg18_{chrom}_result.csv"
            intervals: List[Tuple[int, int]] = []
            if cpx_path.exists():
                with cpx_path.open(newline="", encoding="utf-8") as handle:
                    reader = csv.DictReader(handle, delimiter="\t")
                    for row in reader:
                        start = to_int(row["POSITION"])
                        end = to_int(row["END"])
                        if end > start:
                            intervals.append((start, end))
            intervals.sort()
            chrom_index[cpx_type] = {
                "intervals": intervals,
                "starts": [start for start, _ in intervals],
            }
        index[chrom] = chrom_index
    return index


def interval_overlap(start_a: int, end_a: int, start_b: int, end_b: int) -> int:
    return max(0, min(end_a, end_b) - max(start_a, start_b))


def cpx_context(
    cpx_index: Dict[str, Dict[str, Dict[str, object]]],
    chrom: object,
    start: int,
    end: int,
) -> Tuple[float, int, str]:
    chrom_key = normalize_chromosome(chrom)
    chrom_index = cpx_index.get(chrom_key, {})
    window_start = start
    window_end = end + 1
    overlapping_segments: List[Tuple[int, int]] = []
    overlapping_types: List[str] = []

    for cpx_type in CPX_TYPES:
        type_index = chrom_index.get(cpx_type)
        if not type_index:
            continue
        intervals = type_index["intervals"]
        starts = type_index["starts"]
        first = max(0, bisect_left(starts, window_start) - 1)
        has_overlap = False
        for interval_start, interval_end in intervals[first:]:
            if interval_start >= window_end:
                break
            overlap = interval_overlap(window_start, window_end, interval_start, interval_end)
            if overlap > 0:
                overlapping_segments.append((max(window_start, interval_start), min(window_end, interval_end)))
                has_overlap = True
        if has_overlap:
            overlapping_types.append(cpx_type)

    if not overlapping_segments:
        return 0.0, 0, ""

    overlapping_segments.sort()
    merged: List[Tuple[int, int]] = []
    for segment_start, segment_end in overlapping_segments:
        if not merged or segment_start > merged[-1][1]:
            merged.append((segment_start, segment_end))
        else:
            merged[-1] = (merged[-1][0], max(merged[-1][1], segment_end))

    overlap_bp = sum(segment_end - segment_start for segment_start, segment_end in merged)
    window_length = max(1, window_end - window_start)
    overlap_pct = clamp(100.0 * overlap_bp / window_length, 0.0, 100.0)
    return overlap_pct, min(3, len(overlapping_types)), ";".join(overlapping_types)


def collect_rows() -> Dict[Tuple[str, str], Dict[str, object]]:
    candidates: Dict[Tuple[str, str], Dict[str, object]] = {}

    def ensure(label: str, row: Dict[str, str]) -> Dict[str, object]:
        key = base_key(label, row)
        if key not in candidates:
            candidates[key] = {
                "label": label,
                "fasta_id": row["fasta_id"],
                "chrom": str(row["chrom"]),
                "start": to_int(row["start"]),
                "end": to_int(row["end"]),
                "window_length": to_int(row["length"]),
                "source_hits": {},
                "source_overlap_bp": {},
                "source_overlap_pct": {},
            }
        return candidates[key]

    for source_id, _, _, yes_path, no_path, _ in HUNTER_SOURCES:
        for label, path in [("positive", yes_path), ("negative", no_path)]:
            for row in read_csv(path):
                candidate = ensure(label, row)
                candidate["source_hits"][source_id] = source_detection(row)
                candidate["source_overlap_bp"][source_id] = to_float(row.get("overlap_bp", "0"))
                candidate["source_overlap_pct"][source_id] = to_float(row.get("overlap_percentage", "0"))

    return candidates


def selected_source_summary(candidates: Dict[Tuple[str, str], Dict[str, object]]) -> List[Dict[str, object]]:
    rows = []
    sources = [(source_id, label) for source_id, _, label, *_ in HUNTER_SOURCES]
    for source_id, label in sources:
        positives = [row for row in candidates.values() if row["label"] == "positive"]
        negatives = [row for row in candidates.values() if row["label"] == "negative"]
        tp = sum(1 for row in positives if row["source_hits"].get(source_id, False))
        fn = len(positives) - tp
        fp = sum(1 for row in negatives if row["source_hits"].get(source_id, False))
        tn = len(negatives) - fp
        recall = tp / len(positives) if positives else 0.0
        specificity = tn / len(negatives) if negatives else 0.0
        rows.append(
            {
                "source_id": source_id,
                "source_label": label,
                "tp": tp,
                "tn": tn,
                "fp": fp,
                "fn": fn,
                "recall": f"{recall:.4f}",
                "specificity": f"{specificity:.4f}",
                "balanced_accuracy": f"{(recall + specificity) / 2.0:.4f}",
                "youden_j": f"{recall + specificity - 1.0:.4f}",
            }
        )
    return rows


def make_model_candidate(
    row: Dict[str, object],
    original_intervals: Dict[Tuple[str, str], Tuple[str, int, int]],
    tss_index: Dict[str, List[int]],
    cpx_index: Dict[str, Dict[str, Dict[str, object]]],
) -> Dict[str, object]:
    source_hits = dict(row["source_hits"])
    overlap_bp = dict(row["source_overlap_bp"])
    hunter_ids = [source[0] for source in HUNTER_SOURCES]
    model1_ids = [source[0] for source in HUNTER_SOURCES if source[1] == "model1"]
    model2_ids = [source[0] for source in HUNTER_SOURCES if source[1] == "model2"]
    total_weight = sum(source[5] for source in HUNTER_SOURCES)

    hunter_fraction = sum(1 for source_id in hunter_ids if source_hits.get(source_id, False)) / len(hunter_ids)
    model1_fraction = sum(1 for source_id in model1_ids if source_hits.get(source_id, False)) / len(model1_ids)
    model2_fraction = sum(1 for source_id in model2_ids if source_hits.get(source_id, False)) / len(model2_ids)
    weighted_signal = sum(source[5] for source in HUNTER_SOURCES if source_hits.get(source[0], False)) / total_weight
    max_hunter_overlap = max((overlap_bp.get(source_id, 0.0) for source_id in hunter_ids), default=0.0)
    max_overlap_pct = max((dict(row["source_overlap_pct"]).get(source_id, 0.0) for source_id in hunter_ids), default=0.0)

    zdna_score = clamp(7.0 + 76.0 * weighted_signal + 17.0 * min(max_hunter_overlap, 90.0) / 90.0) / 100.0

    if model1_fraction > 0 and model2_fraction > 0:
        evidence_type = "cross_tool"
    elif hunter_fraction > 0:
        evidence_type = "predicted_motif"
    else:
        evidence_type = "model_only"

    disagreement = abs(model1_fraction - model2_fraction)
    repeat_overlap = clamp(20.0 * (1.0 - hunter_fraction) + 15.0 * disagreement + 5.0 * (1.0 - min(max_overlap_pct, 100.0) / 100.0))
    primer_uniqueness = clamp(86.0 - 0.35 * repeat_overlap, 35.0, 100.0)
    candidate_length = int(round(clamp(max_hunter_overlap if max_hunter_overlap > 0 else min(float(row["window_length"]), 400.0), 8.0, 400.0)))
    context_chrom, context_start, context_end = original_intervals.get(
        (str(row["label"]), str(row["fasta_id"])),
        (str(row["chrom"]), int(row["start"]), int(row["end"])),
    )
    start = context_start
    end = min(context_end, start + candidate_length)
    if end <= start:
        end = start + candidate_length
    tss_distance = nearest_tss_distance(tss_index, context_chrom, start, end)
    promoter_overlap = int(tss_distance <= 1_000)
    cpx_overlap, regulatory_marks, cpx_types = cpx_context(cpx_index, context_chrom, start, end)

    is_positive = row["label"] == "positive"
    expert_priority = 90.0 if is_positive else 20.0

    return {
        "candidate_id": f"SHIN_REAL_{row['label'].upper()}_{row['fasta_id']}",
        "species": "Homo_sapiens_shin2016",
        "chromosome": normalize_chromosome(context_chrom),
        "start": start,
        "end": end,
        "length_bp": candidate_length,
        "evidence_type": evidence_type,
        "zdna_score": f"{zdna_score:.4f}",
        "z_dnabert_probability": "",
        "cpx_overlap_pct": f"{cpx_overlap:.2f}",
        "tss_distance_bp": tss_distance,
        "promoter_overlap": promoter_overlap,
        "regulatory_marks": regulatory_marks,
        "repeat_overlap_pct": f"{repeat_overlap:.2f}",
        "independent_support": int(model1_fraction > 0 and model2_fraction > 0),
        "primer_uniqueness": f"{primer_uniqueness:.2f}",
        "split": "shin_real",
        "expert_priority_score": f"{expert_priority:.1f}",
        "shin_label": row["label"],
        "hunter_vote_fraction": f"{hunter_fraction:.4f}",
        "model1_vote_fraction": f"{model1_fraction:.4f}",
        "model2_vote_fraction": f"{model2_fraction:.4f}",
        "dnabert_vote_fraction": "",
        "aggregate_vote_fraction": f"{hunter_fraction:.4f}",
        "weighted_hunter_signal": f"{weighted_signal:.4f}",
        "selected_sources": ";".join(source_id for source_id, hit in source_hits.items() if hit),
        "cpx_types": cpx_types,
        "context_chromosome": normalize_chromosome(context_chrom),
        "context_window_start": context_start,
        "context_window_end": context_end,
        "shin_window_start": row["start"],
        "shin_window_end": row["end"],
        "shin_window_length": row["window_length"],
    }


def write_csv(rows: Iterable[Dict[str, object]], path: Path) -> None:
    rows = list(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    candidates = collect_rows()
    original_intervals = read_original_shin_intervals(ORIGINAL_FASTA_DIR)
    tss_index = read_tss_index(TSS_PATH)
    context_chromosomes = {
        normalize_chromosome(original_intervals.get((str(row["label"]), str(row["fasta_id"])), (str(row["chrom"]), 0, 0))[0])
        for row in candidates.values()
    }
    cpx_index = read_cpx_index(CPX_DIR, context_chromosomes)
    prepared = [make_model_candidate(row, original_intervals, tss_index, cpx_index) for row in candidates.values()]
    prepared.sort(key=lambda row: (row["shin_label"], row["candidate_id"]))
    write_csv(prepared, OUT_PATH)
    write_csv(selected_source_summary(candidates), SOURCE_SUMMARY_PATH)
    positives = sum(1 for row in prepared if row["shin_label"] == "positive")
    negatives = sum(1 for row in prepared if row["shin_label"] == "negative")
    print(f"Wrote {len(prepared)} rows to {OUT_PATH} ({positives} positive, {negatives} negative)")
    print(f"Loaded {sum(len(positions) for positions in tss_index.values())} TSS positions from {TSS_PATH}")
    print(f"Loaded CpX intervals for {len(cpx_index)} chromosomes from {CPX_DIR}")
    print(f"Wrote selected source summary to {SOURCE_SUMMARY_PATH}")


if __name__ == "__main__":
    main()
