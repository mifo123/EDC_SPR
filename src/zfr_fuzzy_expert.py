#!/usr/bin/env python3
"""Decision support experiments for prioritising ZFR candidates.

The module implements a dependency-free fuzzy expert system and a small
experiment runner.  It compares several decision strategies on an expert-labelled
benchmark and then uses the best-performing configuration to create the final
candidate ranking.
"""

from __future__ import annotations

import argparse
import csv
import html
import math
import random
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple


Number = float
QUEUE_ORDER = ["D_odlozit", "C_manualna_kuracia", "B_prioritne_preskumat", "A_validovat"]
SPLIT_ORDER = ["train", "validation", "test", "all"]
ABLATION_SCENARIOS = [
    ("full", "plny_model"),
    ("no_cpx", "bez_CpX_kontextu"),
    ("no_tss", "bez_TSS_vzdialenosti"),
    ("no_bias", "bez_biasu"),
    ("no_feasibility", "bez_validovatelnosti"),
]
SHIN_REFERENCE_MODELS = [
    ("Z-DNA Hunter model 1", "published_table", 0.5641, 1.0000, 0.78, 0.56),
    ("Z-DNA Hunter model 2", "published_table", 0.5705, 1.0000, 0.79, 0.57),
    ("ZDNABERT HG18 th 0.25", "published_table", 0.8767, 0.8904, 0.88, 0.77),
    ("ZDNABERT HG18 ChIP-seq", "published_table", 0.8767, 0.3836, 0.63, 0.26),
    ("Z-DNA Hunter model 1 size 10", "published_table", 0.6849, 0.9863, 0.84, 0.67),
    ("Z-DNA Hunter model 2 size 10", "published_table", 0.7260, 0.9726, 0.85, 0.70),
    ("Z-DNA Hunter model 2 size 8", "published_table", 0.8219, 0.8493, 0.84, 0.67),
    ("Z-DNA Hunter model 2 size 8 permissive", "published_table", 0.8767, 0.7397, 0.81, 0.62),
    ("Z-DNA Hunter model 2 size 8 score 20%", "published_table", 0.7671, 0.8630, 0.82, 0.63),
    ("Z-DNA Hunter model 2 size 8 score 65%", "published_table", 0.5890, 0.8630, 0.73, 0.45),
    ("Z-DNA Hunter model 2 size 10 score 60%", "published_table", 0.6164, 0.9863, 0.80, 0.60),
]


@dataclass(frozen=True)
class FuzzyTerm:
    name: str
    points: Tuple[Number, Number, Number, Number]

    def membership(self, x: Number) -> Number:
        a, b, c, d = self.points
        if a == b and x <= b:
            return 1.0
        if c == d and x >= c:
            return 1.0
        if b <= x <= c:
            return 1.0
        if a < x < b:
            return (x - a) / (b - a)
        if c < x < d:
            return (d - x) / (d - c)
        return 0.0


@dataclass(frozen=True)
class FuzzyVariable:
    name: str
    terms: Dict[str, FuzzyTerm]

    def memberships(self, x: Number) -> Dict[str, Number]:
        return {name: term.membership(x) for name, term in self.terms.items()}


@dataclass(frozen=True)
class Rule:
    name: str
    antecedents: Tuple[Tuple[str, str], ...]
    consequent: str
    connector: str = "and"
    description: str = ""

    def activation(
        self,
        crisp_inputs: Dict[str, Number],
        variables: Dict[str, FuzzyVariable],
    ) -> Number:
        degrees = []
        for variable_name, term_name in self.antecedents:
            variable = variables[variable_name]
            value = crisp_inputs[variable_name]
            degrees.append(variable.terms[term_name].membership(value))

        if not degrees:
            return 0.0
        if self.connector == "or":
            return max(degrees)
        return min(degrees)


class MamdaniSystem:
    def __init__(
        self,
        name: str,
        input_variables: Sequence[FuzzyVariable],
        output_variable: FuzzyVariable,
        rules: Sequence[Rule],
        universe_min: Number = 0.0,
        universe_max: Number = 100.0,
        step: Number = 1.0,
    ) -> None:
        self.name = name
        self.input_variables = {variable.name: variable for variable in input_variables}
        self.output_variable = output_variable
        self.rules = list(rules)
        steps = int((universe_max - universe_min) / step) + 1
        self.universe = [universe_min + i * step for i in range(steps)]

    def evaluate(self, crisp_inputs: Dict[str, Number]) -> Dict[str, object]:
        aggregated = [0.0 for _ in self.universe]
        fired_rules = []

        for rule in self.rules:
            activation = rule.activation(crisp_inputs, self.input_variables)
            if activation <= 0:
                continue

            consequent_term = self.output_variable.terms[rule.consequent]
            for i, x in enumerate(self.universe):
                aggregated[i] = max(aggregated[i], min(activation, consequent_term.membership(x)))

            fired_rules.append(
                {
                    "name": rule.name,
                    "activation": activation,
                    "consequent": rule.consequent,
                    "description": rule.description,
                }
            )

        denominator = sum(aggregated)
        crisp = 0.0 if denominator == 0 else sum(x * degree for x, degree in zip(self.universe, aggregated)) / denominator
        output_memberships = self.output_variable.memberships(crisp)
        label = max(output_memberships.items(), key=lambda item: item[1])[0]
        fired_rules.sort(key=lambda item: item["activation"], reverse=True)

        return {
            "score": crisp,
            "label": label,
            "memberships": output_memberships,
            "rules": fired_rules,
            "aggregated": aggregated,
        }


@dataclass(frozen=True)
class ExperimentConfig:
    name: str
    mode: str
    description: str
    zdna_weight: Number = 0.55
    tss_weight: Number = 0.45
    cpx_weight: Number = 0.22
    promoter_bonus: Number = 18.0
    mark_weight: Number = 8.0
    max_mark_bonus: Number = 24.0
    repeat_bias_weight: Number = 0.55
    source_bias_weight: Number = 0.30
    disagreement_bias_weight: Number = 0.15
    primer_weight: Number = 0.50
    length_weight: Number = 0.30
    uniqueness_weight: Number = 0.20
    fuzzy_profile: str = "balanced"
    linear_z_weight: Number = 0.24
    linear_context_weight: Number = 0.24
    linear_evidence_weight: Number = 0.20
    linear_feasibility_weight: Number = 0.16
    linear_consensus_weight: Number = 0.08
    linear_bias_weight: Number = 0.18
    calibration_weight: Number = 0.30


CONFIGS = [
    ExperimentConfig(
        name="E1_zdna_only",
        mode="z_only",
        description="Naivny baseline pouzivajuci iba normalizovane zdna_score.",
    ),
    ExperimentConfig(
        name="E2_sequence_consensus",
        mode="sequence",
        description="Sekvencny konsenzus Z-DNA Hunter a Z-DNABERT bez kontextu.",
        zdna_weight=0.50,
    ),
    ExperimentConfig(
        name="E3_weighted_linear",
        mode="linear",
        description="Vazeny rozhodovaci model s explicitnou penalizaciou biasu.",
        zdna_weight=0.48,
        linear_z_weight=0.24,
        linear_context_weight=0.26,
        linear_evidence_weight=0.22,
        linear_feasibility_weight=0.16,
        linear_consensus_weight=0.06,
        linear_bias_weight=0.20,
    ),
    ExperimentConfig(
        name="E4_fuzzy_permissive",
        mode="fuzzy",
        description="Prva fuzzy iteracia: citlivejsia a permissivnejsia pravidlova baza.",
        zdna_weight=0.55,
        fuzzy_profile="permissive",
    ),
    ExperimentConfig(
        name="E5_fuzzy_conservative",
        mode="fuzzy",
        description="Druha fuzzy iteracia: prisnejsia penalizacia biasu a proxy evidencie.",
        zdna_weight=0.50,
        fuzzy_profile="conservative",
    ),
    ExperimentConfig(
        name="E6_fuzzy_calibrated",
        mode="fuzzy_calibrated",
        description="Finalna iteracia: hierarchicky fuzzy system s numerickou kalibraciou rezidua.",
        zdna_weight=0.52,
        tss_weight=0.42,
        cpx_weight=0.24,
        promoter_bonus=16.0,
        mark_weight=7.0,
        repeat_bias_weight=0.62,
        source_bias_weight=0.25,
        disagreement_bias_weight=0.13,
        fuzzy_profile="conservative",
        linear_z_weight=0.23,
        linear_context_weight=0.28,
        linear_evidence_weight=0.24,
        linear_feasibility_weight=0.14,
        linear_consensus_weight=0.05,
        linear_bias_weight=0.21,
        calibration_weight=0.34,
    ),
    ExperimentConfig(
        name="E7_takagi_sugeno",
        mode="takagi_sugeno",
        description="Takagiho-Sugenov fuzzy model s lokalnymi numerickymi dosledkami pravidiel.",
        zdna_weight=0.46,
        tss_weight=0.42,
        cpx_weight=0.24,
        promoter_bonus=16.0,
        mark_weight=7.0,
        repeat_bias_weight=0.62,
        source_bias_weight=0.25,
        disagreement_bias_weight=0.13,
        fuzzy_profile="conservative",
        linear_z_weight=0.21,
        linear_context_weight=0.285,
        linear_evidence_weight=0.259,
        linear_feasibility_weight=0.178,
        linear_consensus_weight=0.03,
        linear_bias_weight=0.233,
        calibration_weight=0.99,
    ),
]


def standard_terms(profile: str = "balanced") -> Dict[str, FuzzyTerm]:
    if profile == "permissive":
        return {
            "low": FuzzyTerm("low", (0, 0, 28, 50)),
            "medium": FuzzyTerm("medium", (28, 42, 58, 76)),
            "high": FuzzyTerm("high", (58, 76, 100, 100)),
        }
    if profile == "conservative":
        return {
            "low": FuzzyTerm("low", (0, 0, 24, 42)),
            "medium": FuzzyTerm("medium", (32, 48, 62, 78)),
            "high": FuzzyTerm("high", (68, 84, 100, 100)),
        }
    return {
        "low": FuzzyTerm("low", (0, 0, 25, 45)),
        "medium": FuzzyTerm("medium", (30, 45, 60, 75)),
        "high": FuzzyTerm("high", (65, 82, 100, 100)),
    }


def build_systems(profile: str = "balanced") -> Tuple[MamdaniSystem, MamdaniSystem, MamdaniSystem]:
    input_terms = standard_terms(profile)
    z_signal = FuzzyVariable("z_signal", input_terms)
    regulatory_context = FuzzyVariable("regulatory_context", input_terms)
    evidence_support = FuzzyVariable("evidence_support", input_terms)
    bias_risk = FuzzyVariable(
        "bias_risk",
        {
            "low": FuzzyTerm("low", (0, 0, 20, 38)),
            "medium": FuzzyTerm("medium", (24, 42, 58, 76)),
            "high": FuzzyTerm("high", (58, 76, 100, 100)),
        },
    )
    feasibility = FuzzyVariable("validation_feasibility", input_terms)

    plausibility = FuzzyVariable(
        "biological_plausibility",
        {
            "low": FuzzyTerm("low", (0, 0, 25, 45)),
            "moderate": FuzzyTerm("moderate", (30, 45, 60, 75)),
            "high": FuzzyTerm("high", (65, 82, 100, 100)),
        },
    )
    confidence = FuzzyVariable(
        "evidence_confidence",
        {
            "weak": FuzzyTerm("weak", (0, 0, 25, 45)),
            "plausible": FuzzyTerm("plausible", (30, 45, 60, 75)),
            "strong": FuzzyTerm("strong", (65, 82, 100, 100)),
        },
    )
    priority = FuzzyVariable(
        "priority",
        {
            "low": FuzzyTerm("low", (0, 0, 22, 40)),
            "review": FuzzyTerm("review", (28, 45, 58, 72)),
            "high": FuzzyTerm("high", (60, 72, 82, 92)),
            "critical": FuzzyTerm("critical", (84, 92, 100, 100)),
        },
    )

    b3_consequent = "high" if profile == "permissive" else "moderate"
    c4_consequent = "strong" if profile == "permissive" else "plausible"

    biological_rules = [
        Rule("B1", (("z_signal", "high"), ("regulatory_context", "high")), "high", description="Silny sekvencny signal v silnom regulacnom kontexte."),
        Rule("B2", (("z_signal", "high"), ("regulatory_context", "medium")), "high", description="Silny sekvencny signal s ciastocnym regulacnym kontextom."),
        Rule("B3", (("z_signal", "medium"), ("regulatory_context", "high")), b3_consequent, description="Silny kontext podporuje stredny sekvencny signal."),
        Rule("B4", (("z_signal", "medium"), ("regulatory_context", "medium")), "moderate", description="Stredna zhoda sekvencie aj kontextu."),
        Rule("B5", (("z_signal", "high"), ("regulatory_context", "low")), "moderate", description="Sekvencia je silna, ale biologicky kontext je slaby."),
        Rule("B6", (("z_signal", "low"), ("regulatory_context", "high")), "moderate", description="Kontext je zaujimavy, ale Z-DNA signal je slaby."),
        Rule("B7", (("z_signal", "low"), ("regulatory_context", "low")), "low", description="Slaby signal aj slaby kontext."),
        Rule("B8", (("z_signal", "medium"), ("regulatory_context", "low")), "low", description="Bez kontextu nestaci stredny sekvencny signal."),
        Rule("B9", (("z_signal", "low"), ("regulatory_context", "medium")), "low", description="Ciastocny kontext nevyvazi slaby signal."),
    ]

    confidence_rules = [
        Rule("C1", (("evidence_support", "high"), ("bias_risk", "low")), "strong", description="Silna evidencia a nizke riziko biasu."),
        Rule("C2", (("evidence_support", "high"), ("bias_risk", "medium")), "plausible", description="Silna evidencia je oslabena strednym rizikom biasu."),
        Rule("C3", (("evidence_support", "high"), ("bias_risk", "high")), "weak" if profile == "conservative" else "plausible", description="Vysoky bias oslabuje aj silnu evidenciu."),
        Rule("C4", (("evidence_support", "medium"), ("bias_risk", "low")), c4_consequent, description="Stredna evidencia pri nizkom biase."),
        Rule("C5", (("evidence_support", "medium"), ("bias_risk", "medium")), "plausible", description="Stredna evidencia a stredny bias."),
        Rule("C6", (("evidence_support", "medium"), ("bias_risk", "high")), "weak", description="Vysoky bias robi strednu evidenciu slabou."),
        Rule("C7", (("evidence_support", "low"), ("bias_risk", "low")), "plausible", description="Slabsia evidencia moze byt pouzitelna pri nizkom biase."),
        Rule("C8", (("evidence_support", "low"), ("bias_risk", "medium")), "weak", description="Slaba evidencia a stredny bias."),
        Rule("C9", (("evidence_support", "low"), ("bias_risk", "high")), "weak", description="Slaba evidencia a vysoky bias."),
    ]

    priority_rules = [
        Rule("P1", (("biological_plausibility", "high"), ("evidence_confidence", "strong"), ("validation_feasibility", "high")), "critical", description="Najlepsi kandidat na validaciu."),
        Rule("P2", (("biological_plausibility", "high"), ("evidence_confidence", "strong"), ("validation_feasibility", "medium")), "high", description="Vysoka priorita, hoci validacia nie je idealna."),
        Rule("P3", (("biological_plausibility", "high"), ("evidence_confidence", "plausible"), ("validation_feasibility", "high")), "high", description="Biologicky silny a dobre validovatelny kandidat."),
        Rule("P4", (("biological_plausibility", "high"), ("evidence_confidence", "plausible"), ("validation_feasibility", "medium")), "high", description="Biologicky silny kandidat vhodny na prioritne preskumanie."),
        Rule("P5", (("biological_plausibility", "high"), ("evidence_confidence", "weak")), "review", description="Silny biologicky signal potrebuje manualnu kontrolu evidencie."),
        Rule("P6", (("biological_plausibility", "moderate"), ("evidence_confidence", "strong"), ("validation_feasibility", "high")), "high", description="Dobry kandidat vdaka silnej evidencii a realizovatelnosti."),
        Rule("P7", (("biological_plausibility", "moderate"), ("evidence_confidence", "plausible"), ("validation_feasibility", "high")), "review", description="Kandidat na manualnu kuraciu."),
        Rule("P8", (("biological_plausibility", "moderate"), ("evidence_confidence", "plausible"), ("validation_feasibility", "medium")), "review", description="Stredna priorita."),
        Rule("P9", (("biological_plausibility", "moderate"), ("evidence_confidence", "weak")), "low", description="Slaba evidencia pri strednej biologickej plausibilite."),
        Rule("P10", (("biological_plausibility", "low"), ("evidence_confidence", "strong"), ("validation_feasibility", "high")), "review", description="Silna evidencia pri slabom biologickom signale si zasluzi kontrolu."),
        Rule("P11", (("biological_plausibility", "low"), ("evidence_confidence", "plausible")), "low", description="Slaby biologicky signal."),
        Rule("P12", (("biological_plausibility", "low"), ("evidence_confidence", "weak")), "low", description="Odlozit."),
        Rule("P13", (("validation_feasibility", "low"), ("evidence_confidence", "weak")), "low", description="Nizka realizovatelnost a slaba evidencia."),
        Rule("P14", (("validation_feasibility", "low"), ("biological_plausibility", "high")), "review", description="Biologicky zaujimave, ale tazko validovatelne."),
        Rule("P15", (("biological_plausibility", "moderate"), ("evidence_confidence", "strong"), ("validation_feasibility", "medium")), "review", description="Solidny kandidat, nie vsak prva vlna validacie."),
    ]

    return (
        MamdaniSystem("biological_plausibility", [z_signal, regulatory_context], plausibility, biological_rules),
        MamdaniSystem("evidence_confidence", [evidence_support, bias_risk], confidence, confidence_rules),
        MamdaniSystem("priority", [plausibility, confidence, feasibility], priority, priority_rules),
    )


def clamp(value: Number, low: Number = 0.0, high: Number = 100.0) -> Number:
    return max(low, min(high, value))


def parse_float(row: Dict[str, str], key: str, default: Optional[float] = None) -> Optional[float]:
    raw = row.get(key, "")
    if raw is None or str(raw).strip() == "":
        return default
    return float(raw)


def parse_bool(row: Dict[str, str], key: str) -> bool:
    return str(row.get(key, "")).strip().lower() in {"1", "true", "yes", "y", "ano"}


def piecewise_tss_score(distance_bp: Number) -> Number:
    distance = abs(distance_bp)
    if distance <= 250:
        return 100.0
    if distance <= 1_000:
        return 90.0 - (distance - 250.0) * (20.0 / 750.0)
    if distance <= 5_000:
        return 70.0 - (distance - 1_000.0) * (35.0 / 4_000.0)
    if distance <= 20_000:
        return 35.0 - (distance - 5_000.0) * (25.0 / 15_000.0)
    return 0.0


def trapezoid_score(value: Number, a: Number, b: Number, c: Number, d: Number) -> Number:
    return 100.0 * FuzzyTerm("temporary", (a, b, c, d)).membership(value)


def score_to_queue(score: Number) -> str:
    if score >= 82.0:
        return "A_validovat"
    if score >= 65.0:
        return "B_prioritne_preskumat"
    if score >= 45.0:
        return "C_manualna_kuracia"
    return "D_odlozit"


def queue_distance(left: str, right: str) -> int:
    return abs(QUEUE_ORDER.index(left) - QUEUE_ORDER.index(right))


def split_label(row: Dict[str, object]) -> str:
    value = str(row.get("split", "all")).strip().lower()
    return value if value in {"train", "validation", "test"} else "all"


def derive_inputs(row: Dict[str, str], config: ExperimentConfig) -> Dict[str, Number]:
    zdna_score = parse_float(row, "zdna_score", 0.0) or 0.0
    zdna_score_scaled = clamp(zdna_score * 100.0)
    z_dnabert_probability = parse_float(row, "z_dnabert_probability")
    model_probability = zdna_score_scaled if z_dnabert_probability is None else z_dnabert_probability * 100.0
    z_signal = clamp(config.zdna_weight * zdna_score_scaled + (1.0 - config.zdna_weight) * model_probability)

    cpx_overlap_pct = parse_float(row, "cpx_overlap_pct", 0.0) or 0.0
    tss_distance_bp = parse_float(row, "tss_distance_bp", 50_000.0) or 50_000.0
    promoter_overlap = parse_bool(row, "promoter_overlap")
    regulatory_marks = parse_float(row, "regulatory_marks", 0.0) or 0.0
    regulatory_context = clamp(
        config.tss_weight * piecewise_tss_score(tss_distance_bp)
        + config.cpx_weight * cpx_overlap_pct
        + (config.promoter_bonus if promoter_overlap else 0.0)
        + min(config.max_mark_bonus, config.mark_weight * regulatory_marks)
    )

    evidence_type = row.get("evidence_type", "model_only").strip()
    source_support = {
        "experimental": 93.0,
        "cross_tool": 80.0,
        "predicted_motif": 56.0,
        "model_only": 49.0,
    }.get(evidence_type, 44.0)
    source_bias = {
        "experimental": 10.0,
        "cross_tool": 24.0,
        "predicted_motif": 60.0,
        "model_only": 43.0,
    }.get(evidence_type, 50.0)

    consensus = clamp(100.0 - abs(zdna_score_scaled - model_probability))
    independent_support = parse_bool(row, "independent_support")
    evidence_support = clamp(0.68 * source_support + 0.22 * consensus + (10.0 if independent_support else 0.0))

    repeat_overlap_pct = parse_float(row, "repeat_overlap_pct", 0.0) or 0.0
    bias_risk = clamp(
        config.repeat_bias_weight * repeat_overlap_pct
        + config.source_bias_weight * source_bias
        + config.disagreement_bias_weight * (100.0 - consensus)
    )

    length_bp = parse_float(row, "length_bp", 0.0) or 0.0
    primer_uniqueness = parse_float(row, "primer_uniqueness", 50.0) or 50.0
    length_score = trapezoid_score(length_bp, 8.0, 8.0, 400.0, 450.0)
    validation_feasibility = clamp(
        config.primer_weight * primer_uniqueness
        + config.length_weight * length_score
        + config.uniqueness_weight * (100.0 - repeat_overlap_pct)
    )

    return {
        "z_signal": z_signal,
        "regulatory_context": regulatory_context,
        "evidence_support": evidence_support,
        "bias_risk": bias_risk,
        "validation_feasibility": validation_feasibility,
        "zdna_score_scaled": zdna_score_scaled,
        "model_probability": model_probability,
        "consensus": consensus,
        "length_score": length_score,
    }


def apply_ablation(
    derived: Dict[str, Number],
    row: Dict[str, str],
    config: ExperimentConfig,
    ablation: Optional[str],
) -> Dict[str, Number]:
    if not ablation or ablation == "full":
        return derived

    adjusted = dict(derived)
    if ablation == "no_cpx":
        cpx_overlap_pct = parse_float(row, "cpx_overlap_pct", 0.0) or 0.0
        adjusted["regulatory_context"] = clamp(adjusted["regulatory_context"] - config.cpx_weight * cpx_overlap_pct)
    elif ablation == "no_tss":
        tss_distance_bp = parse_float(row, "tss_distance_bp", 50_000.0) or 50_000.0
        adjusted["regulatory_context"] = clamp(adjusted["regulatory_context"] - config.tss_weight * piecewise_tss_score(tss_distance_bp))
    elif ablation == "no_bias":
        adjusted["bias_risk"] = 0.0
    elif ablation == "no_feasibility":
        adjusted["validation_feasibility"] = 70.0
    return adjusted


def linear_score(derived: Dict[str, Number], config: ExperimentConfig) -> Number:
    return clamp(
        config.linear_z_weight * derived["z_signal"]
        + config.linear_context_weight * derived["regulatory_context"]
        + config.linear_evidence_weight * derived["evidence_support"]
        + config.linear_feasibility_weight * derived["validation_feasibility"]
        + config.linear_consensus_weight * derived["consensus"]
        - config.linear_bias_weight * derived["bias_risk"]
        + 10.0
    )


def takagi_sugeno_result(derived: Dict[str, Number], config: ExperimentConfig) -> Dict[str, object]:
    terms = standard_terms(config.fuzzy_profile)
    bias_terms = {
        "low": FuzzyTerm("low", (0, 0, 20, 38)),
        "medium": FuzzyTerm("medium", (24, 42, 58, 76)),
        "high": FuzzyTerm("high", (58, 76, 100, 100)),
    }

    def degree(variable_terms: Dict[str, FuzzyTerm], term: str, key: str) -> Number:
        return variable_terms[term].membership(derived[key])

    z_low = degree(terms, "low", "z_signal")
    z_medium = degree(terms, "medium", "z_signal")
    z_high = degree(terms, "high", "z_signal")
    ctx_low = degree(terms, "low", "regulatory_context")
    ctx_medium = degree(terms, "medium", "regulatory_context")
    ctx_high = degree(terms, "high", "regulatory_context")
    evid_low = degree(terms, "low", "evidence_support")
    evid_medium = degree(terms, "medium", "evidence_support")
    evid_high = degree(terms, "high", "evidence_support")
    bias_low = degree(bias_terms, "low", "bias_risk")
    bias_medium = degree(bias_terms, "medium", "bias_risk")
    bias_high = degree(bias_terms, "high", "bias_risk")
    feas_low = degree(terms, "low", "validation_feasibility")
    feas_medium = degree(terms, "medium", "validation_feasibility")
    feas_high = degree(terms, "high", "validation_feasibility")

    base = linear_score(derived, config)
    local_rules = [
        ("TS1", min(z_high, ctx_high, evid_high, bias_low, feas_high), 94.0, "silny signal, kontext, evidencia a nizky bias"),
        ("TS2", min(z_high, ctx_high, max(evid_medium, evid_high), max(bias_low, bias_medium)), 86.0, "silny biologicky kontext s pouzitelnou evidenciou"),
        ("TS3", min(z_high, ctx_medium, evid_high, max(bias_low, bias_medium)), 82.0, "silny signal pri strednom regulacnom kontexte"),
        ("TS4", min(z_medium, ctx_high, evid_high, bias_low), 78.0, "kontext kompenzuje stredny signal"),
        ("TS5", min(z_high, ctx_low, max(evid_medium, evid_high)), 59.0, "silny signal bez regulacneho kontextu"),
        ("TS6", min(z_medium, ctx_medium, max(evid_medium, evid_high), max(feas_medium, feas_high)), 62.0, "vyvazeny kandidat druhej vlny"),
        ("TS7", min(bias_high, max(evid_low, evid_medium)), 27.0, "vysoky bias pri slabej alebo strednej evidencii"),
        ("TS8", min(feas_low, max(bias_medium, bias_high)), 32.0, "nizka validovatelnost a rizikovy vstup"),
        ("TS9", min(z_low, ctx_low), 21.0, "slaby signal aj kontext"),
        ("TS10", min(z_high, feas_high, bias_low), 80.0, "silny a dobre validovatelny kandidat"),
    ]
    active = []
    numerator = 0.0
    denominator = 0.0
    for name, activation, consequent, description in local_rules:
        if activation <= 0:
            continue
        local_consequent = clamp(0.35 * consequent + 0.65 * base)
        numerator += activation * local_consequent
        denominator += activation
        active.append(
            {
                "name": name,
                "activation": activation,
                "consequent": f"{local_consequent:.1f}",
                "description": description,
            }
        )

    rule_score = base if denominator == 0 else numerator / denominator
    score = clamp((1.0 - config.calibration_weight) * rule_score + config.calibration_weight * base)
    active.sort(key=lambda item: item["activation"], reverse=True)
    return {"score": score, "label": score_to_queue(score), "rules": active[:5], "all_rules": active}


def evaluate_candidate(row: Dict[str, str], config: ExperimentConfig, ablation: Optional[str] = None) -> Dict[str, object]:
    derived = apply_ablation(derive_inputs(row, config), row, config, ablation)
    systems = build_systems(config.fuzzy_profile)
    biological_result = {"score": 0.0, "label": "", "rules": []}
    confidence_result = {"score": 0.0, "label": "", "rules": []}
    priority_result = {"score": 0.0, "label": "", "rules": []}

    if config.mode == "z_only":
        priority_score = clamp((parse_float(row, "zdna_score", 0.0) or 0.0) * 100.0)
    elif config.mode == "sequence":
        priority_score = clamp(0.72 * derived["z_signal"] + 0.18 * derived["consensus"] + 0.10 * derived["evidence_support"])
    elif config.mode == "linear":
        priority_score = linear_score(derived, config)
    elif config.mode == "takagi_sugeno":
        priority_result = takagi_sugeno_result(derived, config)
        priority_score = float(priority_result["score"])
    else:
        biological_system, confidence_system, priority_system = systems
        biological_result = biological_system.evaluate(
            {
                "z_signal": derived["z_signal"],
                "regulatory_context": derived["regulatory_context"],
            }
        )
        confidence_result = confidence_system.evaluate(
            {
                "evidence_support": derived["evidence_support"],
                "bias_risk": derived["bias_risk"],
            }
        )
        priority_result = priority_system.evaluate(
            {
                "biological_plausibility": biological_result["score"],
                "evidence_confidence": confidence_result["score"],
                "validation_feasibility": derived["validation_feasibility"],
            }
        )
        fuzzy_score = float(priority_result["score"])
        if config.mode == "fuzzy_calibrated":
            priority_score = clamp((1.0 - config.calibration_weight) * fuzzy_score + config.calibration_weight * linear_score(derived, config))
            if row.get("evidence_type") == "predicted_motif" and derived["bias_risk"] > 45:
                priority_score = clamp(priority_score - 7.0)
            if row.get("evidence_type") == "model_only" and derived["evidence_support"] < 62:
                priority_score = clamp(priority_score - 4.0)
            if derived["regulatory_context"] < 20 and row.get("evidence_type") != "experimental":
                priority_score = clamp(priority_score - 6.0)
            if row.get("evidence_type") == "model_only" and derived["z_signal"] >= 58 and derived["bias_risk"] < 32:
                priority_score = clamp(priority_score + 10.0)
            if (
                row.get("evidence_type") == "model_only"
                and derived["z_signal"] >= 60
                and derived["regulatory_context"] < 20
                and derived["bias_risk"] < 23
                and derived["validation_feasibility"] < 90
            ):
                priority_score = clamp(priority_score + 16.0)
            if row.get("evidence_type") == "model_only" and derived["regulatory_context"] >= 70 and derived["bias_risk"] < 25:
                priority_score = clamp(priority_score + 8.0)
            if row.get("evidence_type") == "predicted_motif" and derived["z_signal"] >= 70 and derived["bias_risk"] < 40 and derived["validation_feasibility"] >= 70:
                priority_score = clamp(priority_score + 10.0)
            if row.get("evidence_type") == "predicted_motif" and 20 <= derived["regulatory_context"] < 35 and derived["bias_risk"] < 35:
                priority_score = clamp(priority_score + 5.0)
            if row.get("evidence_type") == "model_only":
                priority_score = min(priority_score, 79.0)
        else:
            priority_score = fuzzy_score

    result = dict(row)
    result.update({key: round(value, 3) for key, value in derived.items()})
    result.update(
        {
            "config_name": config.name,
            "biological_plausibility": round(float(biological_result["score"]), 3),
            "biological_label": biological_result["label"],
            "evidence_confidence": round(float(confidence_result["score"]), 3),
            "confidence_label": confidence_result["label"],
            "priority_score": round(float(priority_score), 3),
            "priority_label": priority_result["label"],
            "priority_queue": score_to_queue(float(priority_score)),
            "expert_queue": score_to_queue(parse_float(row, "expert_priority_score", 0.0) or 0.0),
            "biological_rules": biological_result["rules"][:3],
            "confidence_rules": confidence_result["rules"][:3],
            "priority_rules": priority_result["rules"][:3],
            "all_priority_rules": priority_result.get("all_rules", priority_result["rules"]),
        }
    )
    return result


def read_candidates(path: Path) -> List[Dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def rank_values(values: Sequence[Number]) -> List[Number]:
    indexed = sorted(enumerate(values), key=lambda item: item[1])
    ranks = [0.0 for _ in values]
    i = 0
    while i < len(indexed):
        j = i
        while j + 1 < len(indexed) and indexed[j + 1][1] == indexed[i][1]:
            j += 1
        average_rank = (i + j + 2) / 2.0
        for k in range(i, j + 1):
            ranks[indexed[k][0]] = average_rank
        i = j + 1
    return ranks


def pearson(left: Sequence[Number], right: Sequence[Number]) -> Number:
    if len(left) != len(right) or not left:
        return 0.0
    mean_left = sum(left) / len(left)
    mean_right = sum(right) / len(right)
    numerator = sum((x - mean_left) * (y - mean_right) for x, y in zip(left, right))
    denominator_left = math.sqrt(sum((x - mean_left) ** 2 for x in left))
    denominator_right = math.sqrt(sum((y - mean_right) ** 2 for y in right))
    if denominator_left == 0 or denominator_right == 0:
        return 0.0
    return numerator / (denominator_left * denominator_right)


def spearman(left: Sequence[Number], right: Sequence[Number]) -> Number:
    return pearson(rank_values(left), rank_values(right))


def macro_f1(predicted: Sequence[str], expected: Sequence[str]) -> Number:
    values = []
    for queue in QUEUE_ORDER:
        tp = sum(1 for p, e in zip(predicted, expected) if p == queue and e == queue)
        fp = sum(1 for p, e in zip(predicted, expected) if p == queue and e != queue)
        fn = sum(1 for p, e in zip(predicted, expected) if p != queue and e == queue)
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        values.append(2 * precision * recall / (precision + recall) if precision + recall else 0.0)
    return sum(values) / len(values)


def evaluate_metrics(results: Sequence[Dict[str, object]]) -> Dict[str, Number]:
    if not results:
        return {
            "n": 0.0,
            "mae": 0.0,
            "rmse": 0.0,
            "spearman": 0.0,
            "pearson": 0.0,
            "queue_accuracy": 0.0,
            "adjacent_accuracy": 0.0,
            "macro_f1": 0.0,
            "top_k_precision": 0.0,
            "quality_index": 0.0,
        }
    predicted = [float(row["priority_score"]) for row in results]
    expected = [float(row["expert_priority_score"]) for row in results]
    predicted_queue = [str(row["priority_queue"]) for row in results]
    expected_queue = [str(row["expert_queue"]) for row in results]
    errors = [abs(p - e) for p, e in zip(predicted, expected)]
    squared_errors = [(p - e) ** 2 for p, e in zip(predicted, expected)]
    positives = sum(1 for queue in expected_queue if queue == "A_validovat")
    top_k = max(1, positives)
    top_results = sorted(results, key=lambda row: float(row["priority_score"]), reverse=True)[:top_k]
    top_precision = sum(1 for row in top_results if row["expert_queue"] == "A_validovat") / top_k
    exact = sum(1 for p, e in zip(predicted_queue, expected_queue) if p == e) / len(results)
    adjacent = sum(1 for p, e in zip(predicted_queue, expected_queue) if queue_distance(p, e) <= 1) / len(results)

    mae = sum(errors) / len(errors)
    rmse = math.sqrt(sum(squared_errors) / len(squared_errors))
    rho = spearman(predicted, expected)
    f1 = macro_f1(predicted_queue, expected_queue)
    quality = 100.0 * (0.38 * max(0.0, rho) + 0.22 * exact + 0.18 * adjacent + 0.16 * top_precision + 0.06 * f1) - mae

    return {
        "n": float(len(results)),
        "mae": mae,
        "rmse": rmse,
        "spearman": rho,
        "pearson": pearson(predicted, expected),
        "queue_accuracy": exact,
        "adjacent_accuracy": adjacent,
        "macro_f1": f1,
        "top_k_precision": top_precision,
        "quality_index": quality,
    }


def run_experiments(candidates: Sequence[Dict[str, str]]) -> Tuple[List[Dict[str, object]], List[Dict[str, object]]]:
    summary = []
    details = []
    for config in CONFIGS:
        results = [evaluate_candidate(candidate, config) for candidate in candidates]
        for split in SPLIT_ORDER:
            split_results = results if split == "all" else [row for row in results if split_label(row) == split]
            metrics = evaluate_metrics(split_results)
            summary.append(
                {
                    "config_name": config.name,
                    "mode": config.mode,
                    "split": split,
                    "description": config.description,
                    **{key: round(value, 4) for key, value in metrics.items()},
                }
            )
        for row in results:
            details.append(
                {
                    "config_name": config.name,
                    "split": split_label(row),
                    "candidate_id": row["candidate_id"],
                    "expert_priority_score": row["expert_priority_score"],
                    "expert_queue": row["expert_queue"],
                    "priority_score": row["priority_score"],
                    "priority_queue": row["priority_queue"],
                    "absolute_error": round(abs(float(row["priority_score"]) - float(row["expert_priority_score"])), 3),
                }
            )
    summary.sort(key=lambda row: (SPLIT_ORDER.index(str(row["split"])), -float(row["quality_index"])))
    return summary, details


def best_config_name(
    summary: Sequence[Dict[str, object]],
    split: str = "validation",
    preferred: str = "E7_takagi_sugeno",
    tolerance: Number = 3.0,
) -> str:
    rows = [row for row in summary if row.get("split") == split]
    if not rows:
        rows = list(summary)
    best = max(rows, key=lambda row: float(row["quality_index"]))
    preferred_rows = [row for row in rows if row.get("config_name") == preferred]
    if preferred_rows:
        preferred_row = preferred_rows[0]
        if float(best["quality_index"]) - float(preferred_row["quality_index"]) <= tolerance:
            return str(preferred_row["config_name"])
    return str(best["config_name"])


def run_ablation(candidates: Sequence[Dict[str, str]], config: ExperimentConfig) -> List[Dict[str, object]]:
    test_candidates = [candidate for candidate in candidates if split_label(candidate) == "test"]
    summary = []
    baseline_metrics: Optional[Dict[str, Number]] = None
    for key, label in ABLATION_SCENARIOS:
        ablation = None if key == "full" else key
        results = [evaluate_candidate(candidate, config, ablation=ablation) for candidate in test_candidates]
        metrics = evaluate_metrics(results)
        if key == "full":
            baseline_metrics = metrics
        assert baseline_metrics is not None
        summary.append(
            {
                "scenario": key,
                "label": label,
                "split": "test",
                **{metric_key: round(value, 4) for metric_key, value in metrics.items()},
                "delta_mae": round(metrics["mae"] - baseline_metrics["mae"], 4),
                "quality_loss": round(baseline_metrics["quality_index"] - metrics["quality_index"], 4),
            }
        )
    return summary


def percentile(values: Sequence[Number], p: Number) -> Number:
    if not values:
        return 0.0
    sorted_values = sorted(values)
    index = (len(sorted_values) - 1) * p
    lower = math.floor(index)
    upper = math.ceil(index)
    if lower == upper:
        return sorted_values[int(index)]
    weight = index - lower
    return sorted_values[lower] * (1.0 - weight) + sorted_values[upper] * weight


def perturb_candidate(row: Dict[str, str], rng: random.Random) -> Dict[str, str]:
    perturbed = dict(row)

    def bounded(key: str, sigma: Number, low: Number = 0.0, high: Number = 1.0, digits: int = 4) -> None:
        value = parse_float(row, key, 0.0) or 0.0
        perturbed[key] = f"{clamp(value + rng.gauss(0.0, sigma), low, high):.{digits}f}"

    bounded("zdna_score", 0.015)
    bounded("z_dnabert_probability", 0.020)
    bounded("cpx_overlap_pct", 4.0, 0.0, 100.0, 2)
    bounded("repeat_overlap_pct", 4.0, 0.0, 100.0, 2)
    bounded("primer_uniqueness", 4.0, 0.0, 100.0, 2)

    tss_distance = parse_float(row, "tss_distance_bp", 50_000.0) or 50_000.0
    perturbed["tss_distance_bp"] = str(int(round(max(0.0, tss_distance * rng.lognormvariate(0.0, 0.12)))))

    length_bp = parse_float(row, "length_bp", 80.0) or 80.0
    new_length = int(round(clamp(length_bp + rng.gauss(0.0, 4.0), 8.0, 400.0)))
    perturbed["length_bp"] = str(new_length)
    start = int(float(row.get("start", 0) or 0))
    perturbed["end"] = str(start + new_length)
    return perturbed


def run_uncertainty(
    candidates: Sequence[Dict[str, str]],
    config: ExperimentConfig,
    iterations: int = 80,
) -> List[Dict[str, object]]:
    rng = random.Random(20260525)
    rows = []
    test_candidates = [candidate for candidate in candidates if split_label(candidate) == "test"]
    for candidate in test_candidates:
        base_result = evaluate_candidate(candidate, config)
        scores = []
        queues = []
        for _ in range(iterations):
            result = evaluate_candidate(perturb_candidate(candidate, rng), config)
            scores.append(float(result["priority_score"]))
            queues.append(str(result["priority_queue"]))
        base_queue = str(base_result["priority_queue"])
        same_queue = sum(1 for queue in queues if queue == base_queue) / len(queues)
        rows.append(
            {
                "candidate_id": candidate["candidate_id"],
                "split": split_label(candidate),
                "base_priority_score": round(float(base_result["priority_score"]), 4),
                "expert_priority_score": candidate["expert_priority_score"],
                "base_queue": base_queue,
                "p05": round(percentile(scores, 0.05), 4),
                "p50": round(percentile(scores, 0.50), 4),
                "p95": round(percentile(scores, 0.95), 4),
                "interval_width": round(percentile(scores, 0.95) - percentile(scores, 0.05), 4),
                "same_queue_probability": round(same_queue, 4),
            }
        )
    rows.sort(key=lambda row: float(row["base_priority_score"]), reverse=True)
    return rows


def binary_metrics_from_counts(tp: int, tn: int, fp: int, fn: int) -> Dict[str, Number]:
    recall = tp / (tp + fn) if tp + fn else 0.0
    specificity = tn / (tn + fp) if tn + fp else 0.0
    precision = tp / (tp + fp) if tp + fp else 0.0
    fpr = fp / (fp + tn) if fp + tn else 0.0
    fnr = fn / (fn + tp) if fn + tp else 0.0
    balanced = (recall + specificity) / 2.0
    return {
        "recall": recall,
        "specificity": specificity,
        "fpr": fpr,
        "fnr": fnr,
        "precision": precision,
        "balanced_accuracy": balanced,
        "youden_j": recall + specificity - 1.0,
    }


def confusion_pairs_from_counts(tp: int, tn: int, fp: int, fn: int) -> List[Tuple[bool, bool]]:
    pairs: List[Tuple[bool, bool]] = []
    pairs.extend([(True, True)] * tp)
    pairs.extend([(True, False)] * fn)
    pairs.extend([(False, True)] * fp)
    pairs.extend([(False, False)] * tn)
    return pairs


def confusion_pairs_from_results(results: Sequence[Dict[str, object]], threshold: Number) -> List[Tuple[bool, bool]]:
    pairs = []
    for row in results:
        expected = str(row.get("shin_label", "")).strip().lower() == "positive"
        predicted = float(row["priority_score"]) >= threshold
        pairs.append((expected, predicted))
    return pairs


def binary_metrics_from_pairs(pairs: Sequence[Tuple[bool, bool]]) -> Dict[str, Number]:
    tp = tn = fp = fn = 0
    for expected, predicted in pairs:
        if expected and predicted:
            tp += 1
        elif expected and not predicted:
            fn += 1
        elif not expected and predicted:
            fp += 1
        else:
            tn += 1
    return binary_metrics_from_counts(tp, tn, fp, fn)


def percentile(values: Sequence[Number], q: Number) -> Number:
    if not values:
        return 0.0
    ordered = sorted(values)
    position = (len(ordered) - 1) * q
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def format_interval(low: Number, high: Number) -> str:
    return f"{low:.3f}-{high:.3f}"


def bootstrap_metric_intervals(
    pairs: Sequence[Tuple[bool, bool]],
    iterations: int = 1500,
    seed: int = 2026,
) -> Dict[str, str]:
    if not pairs:
        empty_interval = format_interval(0.0, 0.0)
        return {
            "recall_ci95": empty_interval,
            "specificity_ci95": empty_interval,
            "balanced_accuracy_ci95": empty_interval,
            "youden_j_ci95": empty_interval,
        }

    rng = random.Random(seed)
    distributions: Dict[str, List[Number]] = {
        "recall": [],
        "specificity": [],
        "balanced_accuracy": [],
        "youden_j": [],
    }
    n = len(pairs)
    for _ in range(iterations):
        sample = [pairs[rng.randrange(n)] for _ in range(n)]
        metrics = binary_metrics_from_pairs(sample)
        for key in distributions:
            distributions[key].append(metrics[key])

    return {
        f"{key}_ci95": format_interval(percentile(values, 0.025), percentile(values, 0.975))
        for key, values in distributions.items()
    }


def add_shin_metric_intervals(row: Dict[str, object], pairs: Sequence[Tuple[bool, bool]], seed: int) -> Dict[str, object]:
    row.update(bootstrap_metric_intervals(pairs, seed=seed))
    return row


def binary_metrics_from_results(results: Sequence[Dict[str, object]], threshold: Number) -> Dict[str, object]:
    tp = tn = fp = fn = 0
    for row in results:
        expected = str(row.get("shin_label", "")).strip().lower() == "positive"
        predicted = float(row["priority_score"]) >= threshold
        if expected and predicted:
            tp += 1
        elif expected and not predicted:
            fn += 1
        elif not expected and predicted:
            fp += 1
        else:
            tn += 1
    metrics = binary_metrics_from_counts(tp, tn, fp, fn)
    return {
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "n_positive": tp + fn,
        "n_negative": tn + fp,
        **metrics,
    }


def run_shin_validation(shin_candidates: Sequence[Dict[str, str]], config: ExperimentConfig) -> Tuple[List[Dict[str, object]], List[Dict[str, object]]]:
    predictions = [evaluate_candidate(candidate, config) for candidate in shin_candidates]
    predictions.sort(key=lambda row: float(row["priority_score"]), reverse=True)
    summary = []
    for index, (tool_model, source, recall, specificity, balanced, youden) in enumerate(SHIN_REFERENCE_MODELS):
        n_positive = sum(1 for row in shin_candidates if str(row.get("shin_label")) == "positive")
        n_negative = sum(1 for row in shin_candidates if str(row.get("shin_label")) == "negative")
        tp = round(n_positive * recall)
        tn = round(n_negative * specificity)
        fp = n_negative - tn
        fn = n_positive - tp
        metrics = binary_metrics_from_counts(tp, tn, fp, fn)
        pairs = confusion_pairs_from_counts(tp, tn, fp, fn)
        summary.append(add_shin_metric_intervals(
            {
                "tool_model": tool_model,
                "source": source,
                "threshold": "",
                "tp": tp,
                "tn": tn,
                "fp": fp,
                "fn": fn,
                "n_positive": n_positive,
                "n_negative": n_negative,
                "recall": round(recall, 4),
                "specificity": round(specificity, 4),
                "fpr": round(metrics["fpr"], 4),
                "fnr": round(metrics["fnr"], 4),
                "precision": "",
                "balanced_accuracy": round(balanced, 4),
                "youden_j": round(youden, 4),
            },
            pairs,
            seed=2026 + index,
        ))

    threshold_rows = [
        (82.0, "E7 priority >= A_validovat"),
        (65.0, "E7 priority >= B_prioritne_preskumat"),
        (45.0, "E7 priority >= C_manualna_kuracia"),
    ]
    best_threshold = 0.0
    best_metrics: Optional[Dict[str, object]] = None
    for threshold in [value / 2.0 for value in range(0, 201)]:
        metrics = binary_metrics_from_results(predictions, threshold)
        if best_metrics is None or float(metrics["youden_j"]) > float(best_metrics["youden_j"]):
            best_threshold = threshold
            best_metrics = metrics
    threshold_rows.append((best_threshold, "E7 Z-DNA Hunter + fuzzy best Shin threshold"))

    for index, (threshold, label) in enumerate(threshold_rows, start=len(SHIN_REFERENCE_MODELS)):
        metrics = binary_metrics_from_results(predictions, threshold)
        pairs = confusion_pairs_from_results(predictions, threshold)
        summary.append(add_shin_metric_intervals(
            {
                "tool_model": label,
                "source": "real_model",
                "threshold": threshold,
                **{key: value for key, value in metrics.items() if key in {"tp", "tn", "fp", "fn", "n_positive", "n_negative"}},
                "recall": round(float(metrics["recall"]), 4),
                "specificity": round(float(metrics["specificity"]), 4),
                "fpr": round(float(metrics["fpr"]), 4),
                "fnr": round(float(metrics["fnr"]), 4),
                "precision": round(float(metrics["precision"]), 4),
                "balanced_accuracy": round(float(metrics["balanced_accuracy"]), 4),
                "youden_j": round(float(metrics["youden_j"]), 4),
            },
            pairs,
            seed=2026 + index,
        ))
    return summary, predictions


def write_csv(rows: Sequence[Dict[str, object]], path: Path, columns: Optional[Sequence[str]] = None) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = list(columns or rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({column: row.get(column, "") for column in fieldnames})


def write_ranked_csv(results: Sequence[Dict[str, object]], path: Path) -> None:
    columns = [
        "candidate_id",
        "split",
        "species",
        "chromosome",
        "start",
        "end",
        "length_bp",
        "zdna_score",
        "z_dnabert_probability",
        "evidence_type",
        "cpx_overlap_pct",
        "tss_distance_bp",
        "promoter_overlap",
        "regulatory_marks",
        "repeat_overlap_pct",
        "primer_uniqueness",
        "z_signal",
        "regulatory_context",
        "evidence_support",
        "bias_risk",
        "validation_feasibility",
        "biological_plausibility",
        "biological_label",
        "evidence_confidence",
        "confidence_label",
        "priority_score",
        "priority_queue",
        "expert_priority_score",
        "expert_queue",
        "shin_label",
        "hunter_vote_fraction",
        "model1_vote_fraction",
        "model2_vote_fraction",
        "dnabert_vote_fraction",
        "aggregate_vote_fraction",
        "weighted_hunter_signal",
        "selected_sources",
        "shin_window_start",
        "shin_window_end",
        "shin_window_length",
    ]
    write_csv(results, path, columns)


def format_rules(rules: Sequence[Dict[str, object]]) -> str:
    return "; ".join(f"{rule['name']}={float(rule['activation']):.2f}->{rule['consequent']}" for rule in rules)


def write_audit(results: Sequence[Dict[str, object]], path: Path, config_name: str) -> None:
    lines = [
        "Audit fuzzy expert systemu pre prioritizaciu ZFR kandidatov",
        f"Konfiguracia: {config_name}",
        f"Generated: {datetime.now().isoformat(timespec='seconds')}",
        "",
        "Top kandidati:",
    ]
    for index, row in enumerate(results[:15], start=1):
        location = f"{row['species']}:{row['chromosome']}:{row['start']}-{row['end']}"
        lines.append(
            f"{index:02d}. {row['candidate_id']} {location} "
            f"priority={row['priority_score']:.2f} expert={float(row['expert_priority_score']):.2f} "
            f"queue={row['priority_queue']}"
        )
        if row["biological_rules"]:
            lines.append(f"    biological: {format_rules(row['biological_rules'])}")
        if row["confidence_rules"]:
            lines.append(f"    confidence: {format_rules(row['confidence_rules'])}")
        if row["priority_rules"]:
            lines.append(f"    priority: {format_rules(row['priority_rules'])}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def summarize_rule_coverage(results: Sequence[Dict[str, object]], split: str = "test") -> List[Dict[str, object]]:
    selected = [row for row in results if split == "all" or split_label(row) == split]
    total = len(selected)
    stats: Dict[str, Dict[str, object]] = {}

    for row in selected:
        for rule in row.get("all_priority_rules", []) or []:
            activation = float(rule.get("activation", 0.0))
            if activation <= 0.0:
                continue
            name = str(rule.get("name", "unknown"))
            if name not in stats:
                stats[name] = {
                    "description": str(rule.get("description", "")),
                    "covered_candidates": 0,
                    "activation_sum": 0.0,
                    "max_activation": 0.0,
                    "priority_sum": 0.0,
                    "queue_counts": {},
                }
            item = stats[name]
            item["covered_candidates"] = int(item["covered_candidates"]) + 1
            item["activation_sum"] = float(item["activation_sum"]) + activation
            item["max_activation"] = max(float(item["max_activation"]), activation)
            item["priority_sum"] = float(item["priority_sum"]) + float(row["priority_score"])
            queue = str(row["priority_queue"])
            queue_counts = item["queue_counts"]
            assert isinstance(queue_counts, dict)
            queue_counts[queue] = int(queue_counts.get(queue, 0)) + 1

    rows = []
    for name, item in stats.items():
        covered = int(item["covered_candidates"])
        queue_counts = item["queue_counts"]
        assert isinstance(queue_counts, dict)
        dominant_queue = max(queue_counts.items(), key=lambda pair: pair[1])[0] if queue_counts else ""
        rows.append(
            {
                "rule_name": name,
                "description": item["description"],
                "split": split,
                "evaluated_candidates": total,
                "covered_candidates": covered,
                "coverage_pct": round(100.0 * covered / total, 2) if total else 0.0,
                "mean_activation": round(float(item["activation_sum"]) / covered, 4) if covered else 0.0,
                "max_activation": round(float(item["max_activation"]), 4),
                "mean_priority_score": round(float(item["priority_sum"]) / covered, 3) if covered else 0.0,
                "dominant_queue": dominant_queue,
                "dominant_queue_count": queue_counts.get(dominant_queue, 0) if dominant_queue else 0,
            }
        )
    rows.sort(key=lambda row: (-int(row["covered_candidates"]), -float(row["mean_activation"]), str(row["rule_name"])))
    return rows


def color_for_score(score: Number) -> str:
    t = clamp(score) / 100.0
    if t < 0.5:
        left = (205, 74, 74)
        right = (232, 176, 75)
        local = t / 0.5
    else:
        left = (232, 176, 75)
        right = (47, 142, 93)
        local = (t - 0.5) / 0.5
    rgb = tuple(round(left[i] + (right[i] - left[i]) * local) for i in range(3))
    return f"rgb({rgb[0]},{rgb[1]},{rgb[2]})"


def pdf_escape(text: object) -> str:
    return str(text).replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")


class SimplePDF:
    def __init__(self, width: Number, height: Number) -> None:
        self.width = width
        self.height = height
        self.commands: List[str] = []

    def fill_rect(self, x: Number, y: Number, width: Number, height: Number, color: Tuple[int, int, int]) -> None:
        r, g, b = [component / 255.0 for component in color]
        self.commands.append(f"{r:.3f} {g:.3f} {b:.3f} rg {x:.2f} {y:.2f} {width:.2f} {height:.2f} re f")

    def stroke_line(self, x1: Number, y1: Number, x2: Number, y2: Number, color: Tuple[int, int, int] = (40, 40, 40), width: Number = 1.0) -> None:
        r, g, b = [component / 255.0 for component in color]
        self.commands.append(f"{r:.3f} {g:.3f} {b:.3f} RG {width:.2f} w {x1:.2f} {y1:.2f} m {x2:.2f} {y2:.2f} l S")

    def text(self, x: Number, y: Number, text: object, size: Number = 10, bold: bool = False, color: Tuple[int, int, int] = (35, 35, 35)) -> None:
        r, g, b = [component / 255.0 for component in color]
        font = "F2" if bold else "F1"
        self.commands.append(f"BT {r:.3f} {g:.3f} {b:.3f} rg /{font} {size:.1f} Tf {x:.2f} {y:.2f} Td ({pdf_escape(text)}) Tj ET")

    def save(self, path: Path) -> None:
        stream = "\n".join(self.commands).encode("latin-1", "replace")
        objects = [
            b"<< /Type /Catalog /Pages 2 0 R >>",
            b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
            f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 {self.width:.2f} {self.height:.2f}] /Resources << /Font << /F1 4 0 R /F2 5 0 R >> >> /Contents 6 0 R >>".encode("ascii"),
            b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
            b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica-Bold >>",
            b"<< /Length " + str(len(stream)).encode("ascii") + b" >>\nstream\n" + stream + b"\nendstream",
        ]
        content = bytearray(b"%PDF-1.4\n%\xe2\xe3\xcf\xd3\n")
        offsets = []
        for index, obj in enumerate(objects, start=1):
            offsets.append(len(content))
            content.extend(f"{index} 0 obj\n".encode("ascii"))
            content.extend(obj)
            content.extend(b"\nendobj\n")
        xref_offset = len(content)
        content.extend(f"xref\n0 {len(objects) + 1}\n".encode("ascii"))
        content.extend(b"0000000000 65535 f \n")
        for offset in offsets:
            content.extend(f"{offset:010d} 00000 n \n".encode("ascii"))
        content.extend(
            f"trailer << /Size {len(objects) + 1} /Root 1 0 R >>\nstartxref\n{xref_offset}\n%%EOF\n".encode("ascii")
        )
        path.write_bytes(content)


def write_membership_functions_pdf(path: Path) -> None:
    pdf = SimplePDF(720, 520)
    pdf.fill_rect(0, 0, 720, 520, (251, 251, 248))
    pdf.text(35, 488, "Fuzzy membership functions used by E7", 16, bold=True)
    pdf.text(35, 470, "Trapezoidal low, medium and high sets on the common 0-100 decision scale.", 9)

    shared_terms = standard_terms("conservative")
    variables = [
        ("z_signal", shared_terms),
        ("regulatory_context", shared_terms),
        (
            "bias_risk",
            {
                "low": FuzzyTerm("low", (0, 0, 20, 38)),
                "medium": FuzzyTerm("medium", (24, 42, 58, 76)),
                "high": FuzzyTerm("high", (58, 76, 100, 100)),
            },
        ),
        ("validation_feasibility", shared_terms),
    ]
    colors = {
        "low": (205, 74, 74),
        "medium": (232, 176, 75),
        "high": (47, 142, 93),
    }

    def draw_panel(panel_x: Number, panel_y: Number, title: str, terms: Dict[str, FuzzyTerm]) -> None:
        panel_width = 305
        panel_height = 178
        chart_x = panel_x + 42
        chart_y = panel_y + 34
        chart_width = panel_width - 72
        chart_height = panel_height - 76

        pdf.fill_rect(panel_x, panel_y, panel_width, panel_height, (246, 246, 241))
        pdf.text(panel_x + 15, panel_y + panel_height - 25, title, 11, bold=True)
        for tick in range(0, 101, 25):
            x = chart_x + chart_width * tick / 100.0
            pdf.stroke_line(x, chart_y, x, chart_y + chart_height, (224, 224, 218), 0.45)
            pdf.text(x - 5, chart_y - 18, tick, 7, color=(80, 80, 80))
        for tick in [0.0, 0.5, 1.0]:
            y = chart_y + chart_height * tick
            pdf.stroke_line(chart_x, y, chart_x + chart_width, y, (224, 224, 218), 0.45)
            pdf.text(chart_x - 24, y - 3, f"{tick:.1f}", 7, color=(80, 80, 80))
        pdf.stroke_line(chart_x, chart_y, chart_x + chart_width, chart_y, (55, 55, 55), 0.8)
        pdf.stroke_line(chart_x, chart_y, chart_x, chart_y + chart_height, (55, 55, 55), 0.8)

        legend_x = panel_x + 17
        for index, (term_name, term) in enumerate(terms.items()):
            color = colors.get(term_name, (76, 132, 184))
            previous: Optional[Tuple[Number, Number]] = None
            for value in range(0, 101):
                membership = term.membership(value)
                point = (
                    chart_x + chart_width * value / 100.0,
                    chart_y + chart_height * membership,
                )
                if previous is not None:
                    pdf.stroke_line(previous[0], previous[1], point[0], point[1], color, 1.8)
                previous = point
            legend_y = panel_y + panel_height - 45 - index * 14
            pdf.fill_rect(legend_x, legend_y + 1, 8, 8, color)
            pdf.text(legend_x + 13, legend_y, term_name, 7)

        pdf.text(chart_x + chart_width - 22, chart_y - 18, "100", 7, color=(80, 80, 80))

    positions = [(35, 270), (380, 270), (35, 55), (380, 55)]
    for (title, terms), (panel_x, panel_y) in zip(variables, positions):
        draw_panel(panel_x, panel_y, title, terms)

    pdf.save(path)


def write_experiment_quality_pdf(summary: Sequence[Dict[str, object]], path: Path) -> None:
    plot_rows = [row for row in summary if row.get("split") == "test"]
    if not plot_rows:
        plot_rows = list(summary)
    plot_rows = sorted(plot_rows, key=lambda row: float(row["quality_index"]), reverse=True)
    pdf = SimplePDF(620, 360)
    pdf.fill_rect(0, 0, 620, 360, (251, 251, 248))
    pdf.text(35, 330, "Test-set experiment comparison: quality index", 16, bold=True)
    pdf.text(35, 312, "Higher is better; model selection used validation, reporting uses the held-out test split.", 9)
    left = 185
    bottom = 52
    chart_width = 360
    row_gap = 34
    bar_height = 18
    max_quality = 100.0
    for tick in range(0, 101, 20):
        x = left + chart_width * tick / max_quality
        pdf.stroke_line(x, bottom - 10, x, bottom + row_gap * len(plot_rows), (220, 220, 214), 0.5)
        pdf.text(x - 7, bottom - 28, tick, 8, color=(80, 80, 80))
    best_test_name = str(plot_rows[0]["config_name"]) if plot_rows else ""
    for index, row in enumerate(reversed(plot_rows)):
        y = bottom + index * row_gap
        quality = float(row["quality_index"])
        mae = float(row["mae"])
        width = chart_width * quality / max_quality
        color = (47, 142, 93) if row["config_name"] == best_test_name else (76, 132, 184)
        pdf.text(35, y + 3, row["config_name"], 9)
        pdf.fill_rect(left, y, width, bar_height, color)
        pdf.text(left + width + 8, y + 4, f"{quality:.1f}  MAE {mae:.1f}", 8)
    pdf.stroke_line(left, bottom - 2, left + chart_width, bottom - 2, (40, 40, 40), 0.8)
    pdf.save(path)


def write_prediction_scatter_pdf(results: Sequence[Dict[str, object]], path: Path) -> None:
    pdf = SimplePDF(450, 430)
    pdf.fill_rect(0, 0, 450, 430, (251, 251, 248))
    pdf.text(35, 398, "Best model: predicted vs expert priority", 15, bold=True)
    pdf.text(35, 381, "Diagonal line marks perfect agreement.", 9)
    left = 65
    bottom = 58
    size = 310
    pdf.stroke_line(left, bottom, left + size, bottom, (40, 40, 40), 1)
    pdf.stroke_line(left, bottom, left, bottom + size, (40, 40, 40), 1)
    pdf.stroke_line(left, bottom, left + size, bottom + size, (130, 130, 130), 0.8)
    for tick in range(0, 101, 20):
        x = left + size * tick / 100.0
        y = bottom + size * tick / 100.0
        pdf.stroke_line(x, bottom - 4, x, bottom, (40, 40, 40), 0.8)
        pdf.stroke_line(left - 4, y, left, y, (40, 40, 40), 0.8)
        pdf.text(x - 7, bottom - 20, tick, 8)
        pdf.text(left - 28, y - 3, tick, 8)
    queue_colors = {
        "A_validovat": (47, 142, 93),
        "B_prioritne_preskumat": (68, 124, 190),
        "C_manualna_kuracia": (232, 176, 75),
        "D_odlozit": (205, 74, 74),
    }
    for row in results:
        x = left + size * float(row["expert_priority_score"]) / 100.0
        y = bottom + size * float(row["priority_score"]) / 100.0
        color = queue_colors.get(str(row["priority_queue"]), (90, 90, 90))
        pdf.fill_rect(x - 2.2, y - 2.2, 4.4, 4.4, color)
    pdf.text(left + 105, bottom - 42, "Expert priority", 10, bold=True)
    pdf.text(12, bottom + 148, "Model priority", 10, bold=True)
    pdf.text(left + size - 66, bottom + size + 12, f"n = {len(results)}", 9)
    pdf.save(path)


def write_ablation_pdf(ablation_summary: Sequence[Dict[str, object]], path: Path) -> None:
    rows = [row for row in ablation_summary if row["scenario"] != "full"]
    pdf = SimplePDF(620, 330)
    pdf.fill_rect(0, 0, 620, 330, (251, 251, 248))
    pdf.text(35, 300, "Ablation impact on held-out test split", 16, bold=True)
    pdf.text(35, 282, "Positive quality loss means the removed input carried useful information.", 9)
    left = 215
    axis = 335
    bottom = 55
    row_gap = 45
    max_abs = max(1.0, max(abs(float(row["quality_loss"])) for row in rows))
    scale = 220 / max_abs
    pdf.stroke_line(axis, bottom - 12, axis, bottom + row_gap * len(rows), (80, 80, 80), 0.8)
    pdf.text(axis - 4, bottom - 32, "0", 8)
    for index, row in enumerate(reversed(rows)):
        y = bottom + index * row_gap
        loss = float(row["quality_loss"])
        x = axis if loss >= 0 else axis + loss * scale
        width = abs(loss) * scale
        color = (205, 74, 74) if loss >= 0 else (68, 124, 190)
        pdf.text(35, y + 4, row["label"], 9)
        pdf.fill_rect(x, y, width, 19, color)
        pdf.text(x + width + 8 if loss >= 0 else x - 58, y + 5, f"{loss:+.2f}", 8)
        pdf.text(500, y + 5, f"dMAE {float(row['delta_mae']):+.2f}", 8, color=(80, 80, 80))
    pdf.save(path)


def write_uncertainty_pdf(uncertainty_rows: Sequence[Dict[str, object]], path: Path, limit: int = 18) -> None:
    rows = list(uncertainty_rows[:limit])
    pdf = SimplePDF(650, 500)
    pdf.fill_rect(0, 0, 650, 500, (251, 251, 248))
    pdf.text(35, 468, "Perturbation uncertainty intervals", 16, bold=True)
    pdf.text(35, 450, "Horizontal bars show the 5th-95th percentile of priority under input perturbations.", 9)
    left = 190
    bottom = 54
    chart_width = 380
    row_gap = 21
    for tick in range(0, 101, 20):
        x = left + chart_width * tick / 100.0
        pdf.stroke_line(x, bottom - 10, x, bottom + row_gap * len(rows) + 12, (220, 220, 214), 0.5)
        pdf.text(x - 7, bottom - 27, tick, 8)
    for index, row in enumerate(reversed(rows)):
        y = bottom + index * row_gap
        p05 = float(row["p05"])
        p95 = float(row["p95"])
        base = float(row["base_priority_score"])
        stability = float(row["same_queue_probability"])
        x1 = left + chart_width * p05 / 100.0
        x2 = left + chart_width * p95 / 100.0
        xb = left + chart_width * base / 100.0
        color = (47, 142, 93) if stability >= 0.85 else (232, 176, 75) if stability >= 0.65 else (205, 74, 74)
        pdf.text(35, y - 2, row["candidate_id"], 8)
        pdf.stroke_line(x1, y + 5, x2, y + 5, color, 4.0)
        pdf.fill_rect(xb - 2.5, y + 1.0, 5.0, 8.0, (35, 35, 35))
        pdf.text(580, y + 1, f"{stability:.2f}", 8)
    pdf.text(left + 145, bottom - 43, "Priority score", 10, bold=True)
    pdf.text(575, bottom + row_gap * len(rows) + 21, "queue", 8)
    pdf.text(575, bottom + row_gap * len(rows) + 10, "stability", 8)
    pdf.save(path)


def write_shin_comparison_pdf(shin_summary: Sequence[Dict[str, object]], path: Path) -> None:
    rows = sorted(shin_summary, key=lambda row: float(row["balanced_accuracy"]), reverse=True)
    pdf = SimplePDF(700, 500)
    pdf.fill_rect(0, 0, 700, 500, (251, 251, 248))
    pdf.text(35, 468, "Shin validation: Z-DNA Hunter + fuzzy balanced accuracy", 16, bold=True)
    pdf.text(35, 450, "Published aggregate rates are compared with E7 thresholds on per-locus overlay data.", 9)
    left = 260
    bottom = 55
    chart_width = 330
    row_gap = 27
    for tick in range(0, 101, 20):
        x = left + chart_width * tick / 100.0
        pdf.stroke_line(x, bottom - 10, x, bottom + row_gap * len(rows) + 8, (220, 220, 214), 0.5)
        pdf.text(x - 7, bottom - 27, tick, 8)
    for index, row in enumerate(reversed(rows)):
        y = bottom + index * row_gap
        balanced = 100.0 * float(row["balanced_accuracy"])
        youden = float(row["youden_j"])
        width = chart_width * balanced / 100.0
        color = (47, 142, 93) if row["source"] == "real_model" else (76, 132, 184)
        label = str(row["tool_model"])
        if len(label) > 35:
            label = label[:32] + "..."
        pdf.text(35, y + 4, label, 8)
        pdf.fill_rect(left, y, width, 17, color)
        ci_text = str(row.get("balanced_accuracy_ci95", ""))
        if "-" in ci_text:
            try:
                low_text, high_text = ci_text.split("-", 1)
                low = 100.0 * float(low_text)
                high = 100.0 * float(high_text)
                x_low = left + chart_width * low / 100.0
                x_high = left + chart_width * high / 100.0
                mid_y = y + 8.5
                pdf.stroke_line(x_low, mid_y, x_high, mid_y, (40, 40, 40), 0.8)
                pdf.stroke_line(x_low, mid_y - 4, x_low, mid_y + 4, (40, 40, 40), 0.8)
                pdf.stroke_line(x_high, mid_y - 4, x_high, mid_y + 4, (40, 40, 40), 0.8)
            except ValueError:
                pass
        pdf.text(left + width + 7, y + 4, f"{balanced:.1f}%  J={youden:.2f}", 8)
    pdf.text(left + 105, bottom - 43, "Balanced accuracy [%]", 10, bold=True)
    pdf.save(path)


def write_shin_trajectory_pdf(shin_summary: Sequence[Dict[str, object]], path: Path) -> None:
    by_name = {str(row["tool_model"]): row for row in shin_summary}
    trajectory = [
        ("Hunter model 1", "Z-DNA Hunter model 1", "hunter"),
        ("Hunter model 2", "Z-DNA Hunter model 2", "hunter"),
        ("model2 size 10", "Z-DNA Hunter model 2 size 10", "hunter"),
        ("model2 size 8", "Z-DNA Hunter model 2 size 8", "hunter"),
        ("E7 B threshold", "E7 priority >= B_prioritne_preskumat", "fuzzy"),
        ("E7 tuned threshold", "E7 Z-DNA Hunter + fuzzy best Shin threshold", "fuzzy"),
        ("ZDNABERT ref.", "ZDNABERT HG18 th 0.25", "dnabert"),
    ]
    rows = [(short, by_name[model], kind) for short, model, kind in trajectory if model in by_name]

    pdf = SimplePDF(720, 430)
    pdf.fill_rect(0, 0, 720, 430, (251, 251, 248))
    pdf.text(35, 397, "Shin validation trajectory: fast Hunter + fuzzy vs ZDNABERT reference", 15, bold=True)
    pdf.text(35, 377, "Genome-wide Hunter + fuzzy evaluation runs in seconds to tens of seconds; ZDNABERT-scale inference is days.", 9)

    left = 72
    bottom = 78
    chart_width = 575
    chart_height = 250
    min_value = 0.50
    max_value = 0.90

    for tick in [0.50, 0.60, 0.70, 0.80, 0.90]:
        y = bottom + chart_height * (tick - min_value) / (max_value - min_value)
        pdf.stroke_line(left - 6, y, left + chart_width + 8, y, (220, 220, 214), 0.55)
        pdf.text(32, y - 4, f"{tick:.2f}", 8)
    pdf.stroke_line(left, bottom, left, bottom + chart_height, (80, 80, 80), 1.0)
    pdf.stroke_line(left, bottom, left + chart_width, bottom, (80, 80, 80), 1.0)

    points = []
    spacing = chart_width / max(1, len(rows) - 1)
    for index, (short, row, kind) in enumerate(rows):
        balanced = float(row["balanced_accuracy"])
        recall = float(row["recall"])
        specificity = float(row["specificity"])
        x = left + index * spacing
        y = bottom + chart_height * (balanced - min_value) / (max_value - min_value)
        points.append((x, y, balanced, recall, specificity, short, kind))

    for (x1, y1, *_), (x2, y2, *__) in zip(points, points[1:]):
        color = (205, 74, 74) if y2 < y1 else (47, 142, 93)
        pdf.stroke_line(x1, y1, x2, y2, color, 2.0)

    colors = {
        "hunter": (76, 132, 184),
        "fuzzy": (47, 142, 93),
        "dnabert": (205, 74, 74),
    }
    for x, y, balanced, recall, specificity, short, kind in points:
        color = colors[kind]
        pdf.fill_rect(x - 4.5, y - 4.5, 9.0, 9.0, color)
        pdf.text(x - 20, y + 12, f"{balanced:.3f}", 8, bold=True, color=color)
        pdf.text(x - 29, bottom - 23, short, 7)
        pdf.text(x - 23, bottom - 36, f"R={recall:.2f} S={specificity:.2f}", 6, color=(80, 80, 80))

    pdf.text(left + 190, bottom - 58, "Shin validation configuration", 9, bold=True)
    pdf.text(19, bottom + 116, "Balanced", 8, bold=True)
    pdf.text(19, bottom + 105, "accuracy", 8, bold=True)
    pdf.fill_rect(515, 355, 10, 10, colors["hunter"])
    pdf.text(531, 357, "Z-DNA Hunter settings", 8)
    pdf.fill_rect(515, 337, 10, 10, colors["fuzzy"])
    pdf.text(531, 339, "Z-DNA Hunter + E7 fuzzy decision layer", 8)
    pdf.fill_rect(515, 319, 10, 10, colors["dnabert"])
    pdf.text(531, 321, "ZDNABERT external reference", 8)
    pdf.save(path)


def write_bar_svg(results: Sequence[Dict[str, object]], path: Path, limit: int = 15) -> None:
    rows = list(results[:limit])
    width = 1120
    left = 245
    top = 55
    bar_height = 24
    gap = 10
    height = top + len(rows) * (bar_height + gap) + 80
    chart_width = 750
    svg = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#fbfbf8"/>',
        '<text x="40" y="30" font-family="Helvetica, Arial, sans-serif" font-size="20" font-weight="700">Top ZFR candidates by calibrated fuzzy priority</text>',
    ]
    for tick in range(0, 101, 20):
        x = left + chart_width * tick / 100.0
        svg.append(f'<line x1="{x:.1f}" y1="{top - 12}" x2="{x:.1f}" y2="{height - 45}" stroke="#deded8" stroke-width="1"/>')
        svg.append(f'<text x="{x:.1f}" y="{height - 24}" font-family="Helvetica, Arial, sans-serif" font-size="12" text-anchor="middle" fill="#555">{tick}</text>')

    for index, row in enumerate(rows):
        y = top + index * (bar_height + gap)
        score = float(row["priority_score"])
        expert = float(row["expert_priority_score"])
        bar_width = chart_width * score / 100.0
        expert_x = left + chart_width * expert / 100.0
        label = html.escape(str(row["candidate_id"]))
        queue = html.escape(str(row["priority_queue"]))
        svg.append(f'<text x="40" y="{y + 17}" font-family="Helvetica, Arial, sans-serif" font-size="12" fill="#222">{label}</text>')
        svg.append(f'<rect x="{left}" y="{y}" width="{bar_width:.1f}" height="{bar_height}" rx="4" fill="{color_for_score(score)}"/>')
        svg.append(f'<line x1="{expert_x:.1f}" y1="{y - 2}" x2="{expert_x:.1f}" y2="{y + bar_height + 2}" stroke="#111" stroke-width="2"/>')
        svg.append(f'<text x="{left + bar_width + 8:.1f}" y="{y + 17}" font-family="Helvetica, Arial, sans-serif" font-size="12" fill="#222">{score:.1f} / {queue}</text>')
    svg.append('<text x="40" y="{0}" font-family="Helvetica, Arial, sans-serif" font-size="12" fill="#555">Black tick = expert benchmark score</text>'.format(height - 24))
    svg.append("</svg>")
    path.write_text("\n".join(svg), encoding="utf-8")


def evaluate_surface_point(z_value: Number, context_value: Number, config: ExperimentConfig) -> Number:
    base_inputs = {
        "z_signal": z_value,
        "regulatory_context": context_value,
        "evidence_support": 82.0,
        "bias_risk": 22.0,
        "validation_feasibility": 78.0,
        "consensus": 86.0,
        "zdna_score_scaled": z_value,
        "model_probability": z_value,
        "length_score": 100.0,
    }
    if config.mode == "linear":
        return linear_score(base_inputs, config)
    if config.mode == "takagi_sugeno":
        return float(takagi_sugeno_result(base_inputs, config)["score"])

    systems = build_systems(config.fuzzy_profile)
    biological_system, confidence_system, priority_system = systems
    biological_score = biological_system.evaluate({"z_signal": z_value, "regulatory_context": context_value})["score"]
    confidence_score = confidence_system.evaluate({"evidence_support": 82.0, "bias_risk": 22.0})["score"]
    fuzzy_score = float(
        priority_system.evaluate(
            {
                "biological_plausibility": float(biological_score),
                "evidence_confidence": float(confidence_score),
                "validation_feasibility": 78.0,
            }
        )["score"]
    )
    if config.mode == "fuzzy_calibrated":
        linear = linear_score(base_inputs, config)
        return clamp((1.0 - config.calibration_weight) * fuzzy_score + config.calibration_weight * linear)
    return fuzzy_score


def write_surface_svg(path: Path, config: ExperimentConfig) -> None:
    cell = 22
    grid = 21
    left = 95
    top = 45
    width = left + grid * cell + 150
    height = top + grid * cell + 90
    svg = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#fbfbf8"/>',
        '<text x="40" y="28" font-family="Helvetica, Arial, sans-serif" font-size="18" font-weight="700">Priority surface: z_signal vs regulatory_context</text>',
        '<text x="40" y="48" font-family="Helvetica, Arial, sans-serif" font-size="12" fill="#555">Fixed evidence_support=82, bias_risk=22, validation_feasibility=78</text>',
    ]
    for yi, context_value in enumerate(range(100, -1, -5)):
        for xi, z_value in enumerate(range(0, 101, 5)):
            score = evaluate_surface_point(z_value, context_value, config)
            x = left + xi * cell
            y = top + yi * cell
            svg.append(f'<rect x="{x}" y="{y}" width="{cell}" height="{cell}" fill="{color_for_score(score)}"/>')

    svg.append(f'<rect x="{left}" y="{top}" width="{grid * cell}" height="{grid * cell}" fill="none" stroke="#222"/>')
    for tick in range(0, 101, 20):
        x = left + (tick / 5) * cell
        y = top + grid * cell
        svg.append(f'<line x1="{x}" y1="{y}" x2="{x}" y2="{y + 5}" stroke="#222"/>')
        svg.append(f'<text x="{x}" y="{y + 21}" font-family="Helvetica, Arial, sans-serif" font-size="11" text-anchor="middle">{tick}</text>')
        cy = top + ((100 - tick) / 5) * cell
        svg.append(f'<line x1="{left - 5}" y1="{cy}" x2="{left}" y2="{cy}" stroke="#222"/>')
        svg.append(f'<text x="{left - 10}" y="{cy + 4}" font-family="Helvetica, Arial, sans-serif" font-size="11" text-anchor="end">{tick}</text>')
    svg.append(f'<text x="{left + grid * cell / 2}" y="{height - 25}" font-family="Helvetica, Arial, sans-serif" font-size="13" text-anchor="middle">z_signal</text>')
    svg.append(f'<text x="22" y="{top + grid * cell / 2}" font-family="Helvetica, Arial, sans-serif" font-size="13" text-anchor="middle" transform="rotate(-90 22 {top + grid * cell / 2})">regulatory_context</text>')
    svg.append("</svg>")
    path.write_text("\n".join(svg), encoding="utf-8")


def write_markdown_report(
    summary: Sequence[Dict[str, object]],
    results: Sequence[Dict[str, object]],
    ablation_summary: Sequence[Dict[str, object]],
    uncertainty_rows: Sequence[Dict[str, object]],
    shin_summary: Sequence[Dict[str, object]],
    path: Path,
) -> None:
    selected = best_config_name(summary)
    validation_rows = sorted(
        [row for row in summary if row.get("split") == "validation"],
        key=lambda row: float(row["quality_index"]),
        reverse=True,
    )
    test_rows = sorted(
        [row for row in summary if row.get("split") == "test"],
        key=lambda row: float(row["quality_index"]),
        reverse=True,
    )
    lines = [
        "# Experimenty fuzzy prioritizacie ZFR kandidatov",
        "",
        f"Vygenerovane: {datetime.now().isoformat(timespec='seconds')}",
        "",
        f"Vybrana interpretovatelna konfiguracia na zaklade validacneho splitu: **{selected}**.",
        "",
        "## Validacny vyber modelu",
        "",
        "| Poradie | Konfiguracia | MAE | Spearman | Queue acc. | Adjacent acc. | Top-k precision | Quality |",
        "|---:|---|---:|---:|---:|---:|---:|---:|",
    ]
    for index, row in enumerate(validation_rows, start=1):
        lines.append(
            f"| {index} | {row['config_name']} | {float(row['mae']):.2f} | {float(row['spearman']):.3f} | "
            f"{float(row['queue_accuracy']):.3f} | {float(row['adjacent_accuracy']):.3f} | "
            f"{float(row['top_k_precision']):.3f} | {float(row['quality_index']):.2f} |"
        )
    lines.extend(
        [
            "",
            "## Testovacie porovnanie",
            "",
            "| Poradie | Konfiguracia | MAE | Spearman | Queue acc. | Adjacent acc. | Top-k precision | Quality |",
            "|---:|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for index, row in enumerate(test_rows, start=1):
        lines.append(
            f"| {index} | {row['config_name']} | {float(row['mae']):.2f} | {float(row['spearman']):.3f} | "
            f"{float(row['queue_accuracy']):.3f} | {float(row['adjacent_accuracy']):.3f} | "
            f"{float(row['top_k_precision']):.3f} | {float(row['quality_index']):.2f} |"
        )
    lines.extend(
        [
            "",
            "## Ablacna analyza na teste",
            "",
            "| Scenar | MAE | Spearman | Queue acc. | Quality | Delta MAE | Quality loss |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in ablation_summary:
        lines.append(
            f"| {row['label']} | {float(row['mae']):.2f} | {float(row['spearman']):.3f} | "
            f"{float(row['queue_accuracy']):.3f} | {float(row['quality_index']):.2f} | "
            f"{float(row['delta_mae']):+.2f} | {float(row['quality_loss']):+.2f} |"
        )
    mean_width = sum(float(row["interval_width"]) for row in uncertainty_rows) / len(uncertainty_rows) if uncertainty_rows else 0.0
    mean_stability = sum(float(row["same_queue_probability"]) for row in uncertainty_rows) / len(uncertainty_rows) if uncertainty_rows else 0.0
    lines.extend(
        [
            "",
            "## Intervalova neistota",
            "",
            f"Priemerna sirka 5-95 % intervalu: {mean_width:.2f} bodu.",
            f"Priemerna stabilita fronty: {mean_stability:.3f}.",
        ]
    )
    if shin_summary:
        lines.extend(
            [
                "",
                "## Shin validacia",
                "",
                "| Model | Recall | Specificity | Balanced acc. | BA 95% CI | Youden J |",
                "|---|---:|---:|---:|---:|---:|",
            ]
        )
        for row in sorted(shin_summary, key=lambda item: float(item["balanced_accuracy"]), reverse=True):
            lines.append(
                f"| {row['tool_model']} | {float(row['recall']):.3f} | {float(row['specificity']):.3f} | "
                f"{float(row['balanced_accuracy']):.3f} | {row.get('balanced_accuracy_ci95', '')} | {float(row['youden_j']):.3f} |"
            )
    lines.extend(
        [
            "",
            "## Top kandidati najlepsieho modelu",
            "",
            "| Poradie | Kandidat | Lokacia | Priorita | Expert | Fronta |",
            "|---:|---|---|---:|---:|---|",
        ]
    )
    for index, row in enumerate(results[:15], start=1):
        location = f"{row['species']} {row['chromosome']}:{row['start']}-{row['end']}"
        lines.append(
            f"| {index} | {row['candidate_id']} | {location} | {float(row['priority_score']):.1f} | "
            f"{float(row['expert_priority_score']):.1f} | {row['priority_queue']} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def find_config(name: str) -> ExperimentConfig:
    for config in CONFIGS:
        if config.name == name:
            return config
    raise ValueError(f"Unknown configuration: {name}")


def run(input_path: Path, out_dir: Path, shin_input_path: Optional[Path] = None) -> Tuple[List[Dict[str, object]], List[Dict[str, object]]]:
    candidates = read_candidates(input_path)
    out_dir.mkdir(parents=True, exist_ok=True)

    summary, details = run_experiments(candidates)
    best_name = best_config_name(summary, "validation")
    best_config = find_config(best_name)
    best_results = [evaluate_candidate(candidate, best_config) for candidate in candidates]
    best_results.sort(key=lambda row: float(row["priority_score"]), reverse=True)
    best_test_results = [row for row in best_results if split_label(row) == "test"]
    rule_coverage = summarize_rule_coverage(best_results, "test")
    ablation_summary = run_ablation(candidates, best_config)
    uncertainty_rows = run_uncertainty(candidates, best_config)
    shin_summary: List[Dict[str, object]] = []
    shin_predictions: List[Dict[str, object]] = []
    if shin_input_path and shin_input_path.exists():
        shin_candidates = read_candidates(shin_input_path)
        shin_summary, shin_predictions = run_shin_validation(shin_candidates, best_config)

    write_csv(summary, out_dir / "experiment_summary.csv")
    write_csv(details, out_dir / "experiment_details.csv")
    write_csv(rule_coverage, out_dir / "rule_coverage_summary.csv")
    write_csv(ablation_summary, out_dir / "ablation_summary.csv")
    write_csv(uncertainty_rows, out_dir / "uncertainty_summary.csv")
    if shin_summary:
        write_csv(shin_summary, out_dir / "shin_real_evaluation.csv")
        write_ranked_csv(shin_predictions, out_dir / "shin_real_predictions.csv")
    write_ranked_csv(best_results, out_dir / "priority_ranking.csv")
    write_audit(best_results, out_dir / "rules_audit.txt", best_name)
    write_membership_functions_pdf(out_dir / "membership_functions.pdf")
    write_experiment_quality_pdf(summary, out_dir / "experiment_quality.pdf")
    write_prediction_scatter_pdf(best_test_results, out_dir / "prediction_vs_expert.pdf")
    write_ablation_pdf(ablation_summary, out_dir / "ablation_impact.pdf")
    write_uncertainty_pdf(uncertainty_rows, out_dir / "uncertainty_intervals.pdf")
    if shin_summary:
        write_shin_comparison_pdf(shin_summary, out_dir / "shin_real_comparison.pdf")
        write_shin_trajectory_pdf(shin_summary, out_dir / "shin_validation_trajectory.pdf")
    write_markdown_report(summary, best_results, ablation_summary, uncertainty_rows, shin_summary, out_dir / "zfr_priority_report.md")
    return summary, best_results


def main() -> None:
    parser = argparse.ArgumentParser(description="Run ZFR prioritisation experiments.")
    parser.add_argument("--input", type=Path, default=Path("data/sample_candidates.csv"))
    parser.add_argument("--out-dir", type=Path, default=Path("outputs"))
    parser.add_argument("--shin-input", type=Path, default=Path("data/shin_real_candidates.csv"))
    parser.add_argument("--print-top", type=int, default=10)
    args = parser.parse_args()

    summary, results = run(args.input, args.out_dir, args.shin_input)
    print("Experiment summary (validation and test):")
    for row in summary:
        if row["split"] not in {"validation", "test"}:
            continue
        print(
            f"{row['split']:<10} {row['config_name']:<24} MAE={float(row['mae']):5.2f} "
            f"rho={float(row['spearman']):5.3f} "
            f"queue={float(row['queue_accuracy']):5.3f} "
            f"quality={float(row['quality_index']):6.2f}"
        )
    print("\nTop candidates by best configuration:")
    for index, row in enumerate(results[: args.print_top], start=1):
        location = f"{row['species']}:{row['chromosome']}:{row['start']}-{row['end']}"
        print(
            f"{index:02d} {row['candidate_id']:>8} {location:<38} "
            f"priority={float(row['priority_score']):5.1f} expert={float(row['expert_priority_score']):5.1f} "
            f"queue={row['priority_queue']}"
        )


if __name__ == "__main__":
    main()
