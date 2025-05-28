# -*- encoding: utf-8 -*-

from copy import deepcopy
import pickle
from datetime import datetime, timezone
from typing import List, Optional, Union

import numpy as np
from numpy.random import PCG64, Generator, SeedSequence

from .observations import Observation


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


def regenerate_or_check_detector_generators(
    observations: List[Observation],
    user_seed: Union[int, None] = None,
    dets_random: List[Generator] = None,
):
    comm = observations[0].comm
    if comm is not None:
        rank = comm.rank
        num_ranks = comm.size
    else:
        rank = 0
        num_ranks = 1

    if user_seed is not None:
        RNG_hierarchy = RNGHierarchy(user_seed, num_ranks, observations[0].n_detectors)
        dets_random = RNG_hierarchy.get_detector_level_generators_on_rank(rank=rank)
    if user_seed is None and dets_random is None:
        raise ValueError("You should pass either `user_seed` or `dets_random`.")
    assert len(dets_random) == observations[0].n_detectors, (
        "The number of random generators must match the number of detectors"
    )
    return dets_random


class RNGHierarchy:
    # If breaking changes are introduced, this ensures compatibility.
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

    def __eq__(self, other):
        if not isinstance(other, RNGHierarchy):
            return False

        if self.base_seed != other.base_seed:
            return False

        return self._compare_hierarchy(self.hierarchy, other.hierarchy)

    @staticmethod
    def _compare_hierarchy(h1, h2):
        if set(h1.keys()) != set(h2.keys()):
            return False

        for key in h1:
            node1 = h1[key]
            node2 = h2[key]

            # Check seed_seq state
            if "seed_seq" in node1 or "seed_seq" in node2:
                if type(node1.get("seed_seq")) is not type(node2.get("seed_seq")):
                    return False
                if node1["seed_seq"].state != node2["seed_seq"].state:
                    return False

            # Check generator state
            if "generator" in node1 or "generator" in node2:
                gen1 = node1.get("generator")
                gen2 = node2.get("generator")
                if type(gen1) is not type(gen2):
                    return False

                state1 = gen1.bit_generator.state
                state2 = gen2.bit_generator.state

                if state1["bit_generator"] != state2["bit_generator"]:
                    return False

                if set(state1["state"].keys()) != set(state2["state"].keys()):
                    return False

                for k in state1["state"]:
                    v1 = state1["state"][k]
                    v2 = state2["state"][k]

                    if isinstance(v1, np.ndarray):
                        if not np.array_equal(v1, v2):
                            return False
                    else:
                        if v1 != v2:
                            return False

            # Recurse into children
            children1 = node1.get("children", {})
            children2 = node2.get("children", {})
            if not RNGHierarchy._compare_hierarchy(children1, children2):
                return False

        return True

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
        with open(filename, "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def from_saved_hierarchy(cls, filename) -> "RNGHierarchy":
        with open(filename, "rb") as f:
            rng_hierarchy = pickle.load(f)

        if rng_hierarchy.SAVE_FORMAT_VERSION != cls.SAVE_FORMAT_VERSION:
            msg = "The loaded `RNGHierarchy` is incompatible with the current implementation."
            raise ValueError(msg)

        return rng_hierarchy
