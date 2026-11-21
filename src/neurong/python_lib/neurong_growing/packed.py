"""Size-independent input contracts for Growing Neuron GPU simulation."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def _int_vector(name: str, value: object) -> np.ndarray:
    array = np.ascontiguousarray(np.asarray(value, dtype=np.int64))
    if array.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional")
    return array


def _float_vector(name: str, value: object) -> np.ndarray:
    array = np.ascontiguousarray(np.asarray(value, dtype=np.float64))
    if array.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values")
    return array


def _validate_offsets(name: str, offsets: np.ndarray, terminal: int) -> None:
    if offsets.size == 0 or offsets[0] != 0:
        raise ValueError(f"{name} must start at zero")
    if np.any(offsets[1:] < offsets[:-1]):
        raise ValueError(f"{name} must be monotonic")
    if offsets[-1] != terminal:
        raise ValueError(f"{name} must end at {terminal}, got {offsets[-1]}")


@dataclass(frozen=True)
class PackedEventSources:
    """Arbitrary event trains grouped first by replica, then by source.

    The CSR-style offsets permit different source counts per replica and any
    number of events per source. Source labels and scientific roles stay in
    Growing Neuron rather than becoming backend constants.
    """

    replica_source_offsets: np.ndarray
    source_event_offsets: np.ndarray
    event_times_ms: np.ndarray

    def __post_init__(self) -> None:
        replica_offsets = _int_vector("replica_source_offsets", self.replica_source_offsets)
        source_offsets = _int_vector("source_event_offsets", self.source_event_offsets)
        event_times = _float_vector("event_times_ms", self.event_times_ms)
        source_count = source_offsets.size - 1
        _validate_offsets("source_event_offsets", source_offsets, event_times.size)
        _validate_offsets("replica_source_offsets", replica_offsets, source_count)
        for source in range(source_count):
            begin, end = source_offsets[source : source + 2]
            if np.any(event_times[begin + 1 : end] < event_times[begin : end - 1]):
                raise ValueError(f"events for source {source} must be sorted")
        object.__setattr__(self, "replica_source_offsets", replica_offsets)
        object.__setattr__(self, "source_event_offsets", source_offsets)
        object.__setattr__(self, "event_times_ms", event_times)

    @property
    def replica_count(self) -> int:
        return self.replica_source_offsets.size - 1

    @property
    def source_count(self) -> int:
        return self.source_event_offsets.size - 1


@dataclass(frozen=True)
class PackedContacts:
    """One independent postsynaptic state instance per contact row.

    Repeated target nodes are valid: contacts are never merged or assigned to
    a finite set of mechanism aliases.
    """

    replica_index: np.ndarray
    source_index: np.ndarray
    target_node_index: np.ndarray
    weight: np.ndarray
    delay_ms: np.ndarray
    tau1_ms: np.ndarray
    tau2_ms: np.ndarray
    reversal_mv: np.ndarray

    def __post_init__(self) -> None:
        replicas = _int_vector("replica_index", self.replica_index)
        sources = _int_vector("source_index", self.source_index)
        targets = _int_vector("target_node_index", self.target_node_index)
        fields = {name: _float_vector(name, getattr(self, name)) for name in
                  ("weight", "delay_ms", "tau1_ms", "tau2_ms", "reversal_mv")}
        count = sources.size
        if (replicas.size != count or targets.size != count or
                any(array.size != count for array in fields.values())):
            raise ValueError("all contact arrays must have equal length")
        if np.any(replicas < 0) or np.any(sources < 0) or np.any(targets < 0):
            raise ValueError("replica, source, and target indices must be non-negative")
        if np.any(fields["delay_ms"] < 0.0):
            raise ValueError("delay_ms must be non-negative")
        if np.any(fields["tau1_ms"] <= 0.0) or np.any(fields["tau2_ms"] <= 0.0):
            raise ValueError("synaptic time constants must be positive")
        object.__setattr__(self, "replica_index", replicas)
        object.__setattr__(self, "source_index", sources)
        object.__setattr__(self, "target_node_index", targets)
        for name, array in fields.items():
            object.__setattr__(self, name, array)

    def validate_against(
        self,
        events: PackedEventSources,
        replica_node_offsets: np.ndarray,
    ) -> None:
        node_offsets = _int_vector("replica_node_offsets", replica_node_offsets)
        if node_offsets.size != events.replica_count + 1:
            raise ValueError("replica_node_offsets must have one range per replica")
        _validate_offsets("replica_node_offsets", node_offsets, int(node_offsets[-1]))
        if np.any(self.replica_index >= events.replica_count):
            raise ValueError("contact references a missing replica")
        if np.any(self.source_index >= events.source_count):
            raise ValueError("contact references a missing source")
        if np.any(self.target_node_index >= node_offsets[-1]):
            raise ValueError("contact references a missing target node")
        for replica, source, target in zip(
            self.replica_index, self.source_index, self.target_node_index
        ):
            if not (events.replica_source_offsets[replica] <= source <
                    events.replica_source_offsets[replica + 1]):
                raise ValueError("contact source belongs to a different replica")
            if not (node_offsets[replica] <= target < node_offsets[replica + 1]):
                raise ValueError("contact target belongs to a different replica")

    @property
    def contact_count(self) -> int:
        return self.source_index.size
