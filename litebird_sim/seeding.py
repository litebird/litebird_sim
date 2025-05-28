# -*- encoding: utf-8 -*-

from copy import deepcopy
import pickle
from datetime import datetime, timezone
from typing import List, Optional, Union

from numpy.random import PCG64, Generator, SeedSequence


def get_derived_random_generators(
    source_sequence: Union[SeedSequence, List[SeedSequence]], num_to_spawn: int
):
    if isinstance(source_sequence, SeedSequence):
        source_sequence = [source_sequence]

    derived_sequences: List[SeedSequence] = []
    derived_generators: List[Generator] = []

    for seq in source_sequence:
        if not isinstance(seq, SeedSequence):
            raise ValueError

        new_seqs = seq.spawn(num_to_spawn)
        derived_sequences += new_seqs
        derived_generators += [Generator(PCG64(seq)) for seq in new_seqs]

    return derived_sequences, derived_generators


def get_generator_from_hierarchy(hierarchy: dict, *indices) -> Generator:
    """
    Navigate the hierarchy using a sequence of keys (e.g., "Rank0", "Det1", ...)
    or plain integers, which are auto-converted to expected labels.

    Parameters:
    - hierarchy: dict representing the RNG hierarchy.
    - indices: path to follow to reach a specific generator.

    Returns:
    - numpy.random.Generator instance at the specified location.

    Raises:
    - KeyError if path is invalid.
    """
    node = deepcopy(hierarchy)
    for depth, idx in enumerate(indices):
        # Convert integer index to labeled key
        if isinstance(idx, int):
            idx = f"rank{idx}" if depth == 0 else f"det{idx}"

        if idx not in node:
            raise KeyError(f"Index '{idx}' not found in hierarchy at depth {depth}")

        node = node[idx]

    if "generator" in node:
        return node["generator"]
    else:
        raise KeyError(f"No generator found at path {indices}")


def get_detector_level_generators_from_hierarchy(
    hierarchy: dict, rank: Union[int, str]
) -> List[Generator]:
    """
    Return only the detector-level generators (i.e., direct children of the given MPI rank).

    Parameters:
    - hierarchy: dict representing the RNG hierarchy.
    - rank: integer or string (e.g., 0 or "Rank0")

    Returns:
    - List of numpy.random.Generator instances at detector level.

    Raises:
    - KeyError if the rank is not found or has no detector children.
    """
    # Normalize rank label
    if isinstance(rank, int):
        rank = f"rank{rank}"

    if rank not in hierarchy:
        raise KeyError(f"Rank '{rank}' not found in hierarchy")

    rank_node = hierarchy[rank]
    children = rank_node.get("children", {})

    return [child["generator"] for child in children.values()]


class RNGHierarchy:
    SAVE_FORMAT_VERSION = 1

    def __init__(
        self, base_seed: int, num_ranks: int = None, num_detectors_per_rank: int = None
    ):
        self.base_seed = base_seed
        self.root_seq = SeedSequence(base_seed)
        self.metadata = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "save_format_version": self.SAVE_FORMAT_VERSION,
        }
        self.hierarchy = {}
        if num_ranks is not None:
            self.build_mpi_layer(num_ranks)
            if num_detectors_per_rank is not None:
                self.build_detector_layer(num_detectors_per_rank)

    def __repr__(self):
        return f"RNGHierarchy(base_seed={self.base_seed})"

    def build_mpi_layer(self, num_ranks: int):
        # Spawn MPI rank seed sequences and generators from root
        spawned = self.root_seq.spawn(num_ranks)
        for rank, seq in enumerate(spawned):
            self.hierarchy[f"rank{rank}"] = {
                "seed_seq": seq,
                "generator": Generator(PCG64(seq)),
                "children": {},
            }

    def build_detector_layer(self, num_detectors_per_rank: int):
        # For each MPI rank, spawn detectors from the rank's seed_seq
        for _, rank_node in self.hierarchy.items():
            spawned = rank_node["seed_seq"].spawn(num_detectors_per_rank)
            for det, seq in enumerate(spawned):
                rank_node["children"][f"det{det}"] = {
                    "seed_seq": seq,
                    "generator": Generator(PCG64(seq)),
                    "children": {},
                }

    def add_extra_layer(self, num_children: int, layer_name: Optional[str] = None):
        def recurse_add(node, layer_depth):
            children = node["children"]
            for _, child_node in children.items():
                spawned = child_node["seed_seq"].spawn(num_children)
                new_children = {}
                for i, seq in enumerate(spawned):
                    label = f"{layer_name or f'L{layer_depth}'}_{i}"
                    new_children[label] = {
                        "seed_seq": seq,
                        "generator": Generator(PCG64(seq)),
                        "children": {},
                    }
                child_node["children"] = new_children
                recurse_add(child_node, layer_depth + 1)

        for rank_node in self.hierarchy.values():
            recurse_add(rank_node, 1)

    def build_hierarchy(self, ranks: int, detectors_per_rank: int):
        self.build_mpi_layer(num_ranks=ranks)

        self.build_detector_layer(num_detectors_per_rank=detectors_per_rank)

    def get_generator(self, *indices):
        return get_generator_from_hierarchy(self.hierarchy, *indices)

    def get_detector_level_generators_on_rank(self, rank: Union[int, str]):
        return get_detector_level_generators_from_hierarchy(self.hierarchy, rank)

    def save(self, filename):
        # Flatten the hierarchy to save seed sequences and generator states
        flat_states = {}

        def recurse(node, key_path):
            seed_seq = node["seed_seq"]
            gen = node["generator"]
            key = "-".join(map(str, key_path))
            flat_states[key] = {
                "entropy": seed_seq.entropy,
                "spawn_key": seed_seq.spawn_key,
                "rng_state": gen.bit_generator.state,
            }
            for child_idx, child_node in node.get("children", {}).items():
                recurse(child_node, key_path + [child_idx])

        for rank, rank_node in self.hierarchy.items():
            recurse(rank_node, [rank])

        data = {
            "metadata": {
                "base_seed": self.base_seed,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "save_format_version": self.SAVE_FORMAT_VERSION,
            },
            "states": flat_states,
        }

        with open(filename, "wb") as f:
            pickle.dump(data, f)

    @classmethod
    def from_hierarchy(cls, hierarchy: dict, base_seed: int):
        obj = cls.__new__(cls)
        obj.base_seed = base_seed
        obj.hierarchy = hierarchy

        return obj

    @classmethod
    def from_saved_hierarchy(cls, filename: str) -> "RNGHierarchy":
        with open(filename, "rb") as f:
            data = pickle.load(f)

        save_format_version = data.get("metadata", {}).get("save_format_version", 0)
        if save_format_version != cls.SAVE_FORMAT_VERSION:
            raise ValueError(f"Unsupported save file version {save_format_version}")

        base_seed = data["metadata"]["base_seed"]
        hierarchy = {}

        # Rebuild hierarchy from flat states
        flat_states = data["states"]

        # Helper to recursively build nodes
        def recurse_build(keys):
            node_key = "-".join(str(k) for k in keys)
            state = flat_states[node_key]
            seq = SeedSequence(
                entropy=state["entropy"], spawn_key=tuple(state["spawn_key"])
            )
            gen = Generator(PCG64())
            gen.bit_generator.state = state["rng_state"]
            children = {}
            # Find children keys by looking for longer keys starting with current node_key + '-'
            prefix = node_key + "-"
            child_keys = [
                k for k in flat_states if k.startswith(prefix) and k != node_key
            ]
            immediate_children_indices = set(
                int(k.split("-")[len(keys)])
                for k in child_keys
                if len(k.split("-")) == len(keys) + 1
            )
            for child_idx in immediate_children_indices:
                children[child_idx] = recurse_build(keys + [child_idx])
            return {"seed_seq": seq, "generator": gen, "children": children}

        # Find top-level keys (length 1)
        top_keys = sorted(
            set(int(k.split("-")[0]) for k in flat_states if len(k.split("-")) == 1)
        )
        for tk in top_keys:
            hierarchy[tk] = recurse_build([tk])

        return cls.from_hierarchy(hierarchy, base_seed)
