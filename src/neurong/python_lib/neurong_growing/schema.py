"""Translate Growing Neuron random-tree state into NeuronG celltemplate input.

This layer intentionally stops at morphology.  Spatial contact generation
remains owned by Growing Neuron; mechanisms, synapses, and packed trial events
are added only after morphology/cable parity is validated against NEURON.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np

from biophys_prepare import BioValue, BiophysPrepare


@dataclass(frozen=True)
class GrowingMorphologyConfig:
    template_name: str = "growing_column"
    soma_length_um: float = 20.0
    soma_diam_um: float = 20.0
    soma_nseg: int = 1
    dend_diam_um: float = 1.0
    level_diam_decay: float = 0.9
    dend_nseg: int = 5
    num_cells: int = 1

    def __post_init__(self) -> None:
        if not self.template_name:
            raise ValueError("template_name must not be empty")
        for name in ("soma_length_um", "soma_diam_um", "dend_diam_um"):
            if not np.isfinite(float(getattr(self, name))) or float(getattr(self, name)) <= 0.0:
                raise ValueError(f"{name} must be finite and positive")
        if not np.isfinite(float(self.level_diam_decay)) or float(self.level_diam_decay) <= 0.0:
            raise ValueError("level_diam_decay must be finite and positive")
        if int(self.soma_nseg) <= 0 or int(self.dend_nseg) <= 0:
            raise ValueError("nseg values must be positive")
        if int(self.num_cells) <= 0:
            raise ValueError("num_cells must be positive")


@dataclass(frozen=True)
class GrowingMorphologySpec:
    """NeuronG morphology template plus stable Growing-section name mapping."""

    template: Mapping[str, Any]
    section_name_by_index: Mapping[int, str]


@dataclass(frozen=True)
class GrowingBiophysicsConfig:
    """Forward-only HH settings matching the current Stage617/625 model."""

    v_init_mv: float = -65.0
    ra_ohm_cm: float = 100.0
    cm_uf_cm2: float = 1.0
    gnabar_s_cm2: float = 0.12
    gkbar_s_cm2: float = 0.036
    gl_s_cm2: float = 0.00003
    el_mv: float = -65.0
    ena_mv: float = 50.0
    ek_mv: float = -77.0
    dend_only: bool = False

    def __post_init__(self) -> None:
        positive = ("ra_ohm_cm", "cm_uf_cm2")
        nonnegative = ("gnabar_s_cm2", "gkbar_s_cm2", "gl_s_cm2")
        for name in positive:
            if not np.isfinite(float(getattr(self, name))) or float(getattr(self, name)) <= 0.0:
                raise ValueError(f"{name} must be finite and positive")
        for name in nonnegative:
            if not np.isfinite(float(getattr(self, name))) or float(getattr(self, name)) < 0.0:
                raise ValueError(f"{name} must be finite and non-negative")
        for name in ("v_init_mv", "el_mv", "ena_mv", "ek_mv"):
            if not np.isfinite(float(getattr(self, name))):
                raise ValueError(f"{name} must be finite")


@dataclass(frozen=True)
class GrowingSynapseConfig:
    tau1_ms: float = 0.1
    tau2_ms: float = 2.0
    reversal_mv: float = 0.0

    def __post_init__(self) -> None:
        if not np.isfinite(self.tau1_ms) or not np.isfinite(self.tau2_ms):
            raise ValueError("synaptic time constants must be finite")
        if self.tau1_ms <= 0.0 or self.tau2_ms <= 0.0:
            raise ValueError("synaptic time constants must be positive")
        if not np.isfinite(self.reversal_mv):
            raise ValueError("reversal_mv must be finite")


@dataclass(frozen=True)
class GrowingContactMechanismMaps:
    instances: Mapping[str, Sequence[float]]
    tau1_ms: Mapping[str, Sequence[float]]
    tau2_ms: Mapping[str, Sequence[float]]
    reversal_mv: Mapping[str, Sequence[float]]


def build_contact_instance_map(
    contacts: Sequence[Mapping[str, Any]],
    section_name_by_index: Mapping[int, str],
    *,
    nseg: int,
) -> dict[str, list[float]]:
    """Count independent point-process instances required on each segment."""

    if int(nseg) <= 0:
        raise ValueError("nseg must be positive")
    counts: dict[str, np.ndarray] = {}
    for contact in contacts:
        section_index = int(contact["section_index"])
        if section_index not in section_name_by_index:
            raise ValueError(f"contact references missing section {section_index}")
        loc = float(contact["x"])
        if not np.isfinite(loc) or loc < 0.0 or loc > 1.0:
            raise ValueError(f"contact x must be finite and in [0, 1], got {loc}")
        segment = min(int(np.floor(loc * int(nseg))), int(nseg) - 1)
        section_name = section_name_by_index[section_index]
        per_segment = counts.setdefault(section_name, np.zeros(int(nseg), dtype=np.int64))
        per_segment[segment] += 1
    return {name: values.astype(float).tolist() for name, values in counts.items()}


def build_contact_mechanism_maps(
    contacts: Sequence[Mapping[str, Any]],
    section_name_by_index: Mapping[int, str],
    *,
    nseg: int,
    defaults: GrowingSynapseConfig = GrowingSynapseConfig(),
) -> GrowingContactMechanismMaps:
    """Build multiplicity and stable per-instance Exp2Syn parameter maps."""

    instances = build_contact_instance_map(contacts, section_name_by_index, nseg=nseg)
    buckets: dict[str, list[list[Mapping[str, Any]]]] = {
        name: [[] for _ in range(int(nseg))] for name in instances
    }
    for contact in contacts:
        section_index = int(contact["section_index"])
        section_name = section_name_by_index[section_index]
        loc = float(contact["x"])
        segment = min(int(np.floor(loc * int(nseg))), int(nseg) - 1)
        buckets[section_name][segment].append(contact)

    def values_for(key: str, fallback: float) -> dict[str, list[float]]:
        output: dict[str, list[float]] = {}
        for section_name, per_segment in buckets.items():
            values = [float(row.get(key, fallback)) for rows in per_segment for row in rows]
            if not np.all(np.isfinite(values)):
                raise ValueError(f"contact {key} values must be finite")
            output[section_name] = values
        return output

    tau1 = values_for("tau1_ms", defaults.tau1_ms)
    tau2 = values_for("tau2_ms", defaults.tau2_ms)
    if any(value <= 0.0 for values in tau1.values() for value in values):
        raise ValueError("contact tau1_ms values must be positive")
    if any(value <= 0.0 for values in tau2.values() for value in values):
        raise ValueError("contact tau2_ms values must be positive")
    return GrowingContactMechanismMaps(
        instances=instances,
        tau1_ms=tau1,
        tau2_ms=tau2,
        reversal_mv=values_for("reversal_mv", defaults.reversal_mv),
    )


def _validate_records(records: Sequence[Mapping[str, Any]], lengths_um: np.ndarray) -> None:
    if lengths_um.ndim != 1:
        raise ValueError(f"lengths_um must be one-dimensional, got {lengths_um.shape}")
    if len(records) != lengths_um.size:
        raise ValueError(f"records/lengths mismatch: {len(records)} != {lengths_um.size}")
    if not np.all(np.isfinite(lengths_um)) or np.any(lengths_um <= 0.0):
        raise ValueError("all physical section lengths must be finite and positive")

    indices = [int(record["index"]) for record in records]
    if sorted(indices) != list(range(len(records))):
        raise ValueError("Growing section indices must be dense and zero-based")
    known = set(indices)
    for record in records:
        index = int(record["index"])
        parent = int(record["parent_index"])
        if parent >= 0 and parent not in known:
            raise ValueError(f"section {index} references missing parent {parent}")
        if parent >= index:
            raise ValueError(
                f"section records must be parent-before-child; section {index} has parent {parent}"
            )


def build_morphology_spec(
    neurong_module: Any,
    records: Sequence[Mapping[str, Any]],
    lengths_um: Sequence[float] | np.ndarray,
    config: GrowingMorphologyConfig,
    *,
    section_diam_scales: Mapping[int | str, float] | None = None,
) -> GrowingMorphologySpec:
    """Build a NeuronG celltemplate matching ``RandomTreeTaskCell`` geometry."""

    lengths = np.ascontiguousarray(np.asarray(lengths_um, dtype=np.float64))
    _validate_records(records, lengths)
    scales = dict(section_diam_scales or {})

    soma = neurong_module.section("soma_0", "soma")
    soma.nseg = int(config.soma_nseg)
    soma.L_um = float(config.soma_length_um)
    soma.diam_um = float(config.soma_diam_um)

    by_index: dict[int, Any] = {}
    names: dict[int, str] = {}
    dendrites = []
    for record in records:
        index = int(record["index"])
        level = int(record.get("level", 1))
        name = f"dend_{index}"
        section = neurong_module.section(name, "dend")
        section.nseg = int(config.dend_nseg)
        section.L_um = float(lengths[index])
        scale = float(scales.get(str(index), scales.get(index, 1.0)))
        if not np.isfinite(scale) or scale <= 0.0:
            raise ValueError(f"section_diam_scales[{index}] must be finite and positive")
        section.diam_um = (
            float(config.dend_diam_um)
            * float(config.level_diam_decay) ** max(level - 1, 0)
            * scale
        )
        parent_index = int(record["parent_index"])
        section.connect(soma if parent_index < 0 else by_index[parent_index], 1.0)
        by_index[index] = section
        names[index] = name
        dendrites.append(section)

    return GrowingMorphologySpec(
        template={
            "name": str(config.template_name),
            "num_cells": int(config.num_cells),
            "sections": [soma, *dendrites],
        },
        section_name_by_index=names,
    )


def build_biophysics_template(
    morphology: GrowingMorphologySpec,
    config: GrowingBiophysicsConfig,
    *,
    exp2syn_instances: Mapping[str, Sequence[float]] | None = None,
    exp2syn_parameter_maps: GrowingContactMechanismMaps | None = None,
    vecstim_instances_per_cell: int = 0,
    synapse: GrowingSynapseConfig = GrowingSynapseConfig(),
) -> Mapping[str, Any]:
    """Build standard-HH biophysics for forward-only Growing Neuron simulation.

    Growing Neuron's ``hh_grad`` adds analytic-gradient bookkeeping that the
    current finite-difference training path never reads.  NeuronG therefore
    uses its standard ``hh`` forward mechanism and validates numerical parity
    at the trace, feature, FD, and proposal-decision levels.
    """

    prepare = BiophysPrepare()
    all_sections = prepare.for_label("all")
    all_sections.resistivity(float(config.ra_ohm_cm))
    all_sections.specificCapacitance(float(config.cm_uf_cm2))

    hh_label = "dend" if bool(config.dend_only) else "all"
    hh = prepare.for_label(hh_label)
    # The native builder requires one explicit ion declaration per insert.
    # Duplicate label/mechanism operations are merged into one mechanism
    # population while contributing the Na and K reversal specifications.
    hh.insert("hh", ion="na", erev=float(config.ena_mv))
    hh.insert(
        "hh",
        ion="k",
        erev=float(config.ek_mv),
        gnabar=float(config.gnabar_s_cm2),
        gkbar=float(config.gkbar_s_cm2),
        gl=float(config.gl_s_cm2),
        el=float(config.el_mv),
    )

    # Growing Neuron inserts pas everywhere with g=0 solely to satisfy its
    # historical matrix tooling.  Preserve the forward semantics explicitly.
    all_sections.insert("pas", ion="non_specific", g_pas=0.0, e_pas=float(config.v_init_mv))

    if exp2syn_parameter_maps is not None:
        if exp2syn_instances is not None and dict(exp2syn_instances) != dict(exp2syn_parameter_maps.instances):
            raise ValueError("conflicting Exp2Syn instance maps")
        exp2syn_instances = exp2syn_parameter_maps.instances
    if exp2syn_instances:
        all_sections.insert(
            "Exp2Syn",
            ion="non_specific",
            instances=dict(exp2syn_instances),
            tau1=(dict(exp2syn_parameter_maps.tau1_ms) if exp2syn_parameter_maps else float(synapse.tau1_ms)),
            tau2=(dict(exp2syn_parameter_maps.tau2_ms) if exp2syn_parameter_maps else float(synapse.tau2_ms)),
            e=(dict(exp2syn_parameter_maps.reversal_mv) if exp2syn_parameter_maps else float(synapse.reversal_mv)),
        )
    if int(vecstim_instances_per_cell) < 0:
        raise ValueError("vecstim_instances_per_cell must be non-negative")
    if int(vecstim_instances_per_cell) > 0:
        prepare.for_label("soma").insert(
            "VecStim", ion="non_specific", instances=float(vecstim_instances_per_cell))

    return {
        "template_name": str(morphology.template["name"]),
        "inserts": prepare.prepare_inserts(),
        "ion_species": prepare.prepare_ion_species(),
        "specificCapacitance": prepare.prepare_specificCapacitance(),
        "resistivity": prepare.prepare_resistivity(),
        "v_init": BioValue.fixed(config.v_init_mv),
    }
