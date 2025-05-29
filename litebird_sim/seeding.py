# -*- encoding: utf-8 -*-

import pickle
from copy import deepcopy
from datetime import datetime, timezone
from typing import List, Optional, Tuple, Union

import numpy as np
from numpy.random import PCG64, Generator, SeedSequence

from .observations import Observation


def get_derived_random_generators(
    source_sequence: Union[SeedSequence, List[SeedSequence]], num_to_spawn: int
) -> Tuple[List[SeedSequence], List[Generator]]:
    """
    Generate multiple derived `SeedSequence` and corresponding RNGs from one or more sources.

    This utility function spawns `num_to_spawn` child `SeedSequence` objects from each source
    sequence and uses them to create `numpy.random.Generator` instances using the PCG64 bit generator.

    Parameters
    ----------
    source_sequence : Union[SeedSequence, List[SeedSequence]]
        The source seed sequence(s) from which to derive new sequences.
    num_to_spawn : int
        Number of new sequences to spawn from each source sequence.

    Returns
    -------
    Tuple[List[SeedSequence], List[Generator]]
        Returns a tuple where the first element is the list of newly spawned `SeedSequence`
        objects, and the second is the corresponding list of RNG generators.

    Raises
    ------
    ValueError
        Raised if any of the provided source sequences are not valid `SeedSequence` instances.
    """
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


def get_generator_from_hierarchy(
    hierarchy: dict, *indices: Union[int, str]
) -> Generator:
    """
    Retrieve a specific RNG generator from a nested hierarchy using a path of indices.

    The indices are interpreted as hierarchical labels (e.g., "rank0", "det3"). If integers are
    passed, they are automatically converted to appropriate labels (e.g., (0, 1) → ("rank0", "det1")).

    Parameters
    ----------
    hierarchy : dict
        A nested dictionary representing the RNG hierarchy.
    indices : sequence of str or int
        Path to follow in the hierarchy, e.g., (0, 2) or ("rank0", "det2").

    Returns
    -------
    numpy.random.Generator
        Generator located at the given path in the hierarchy.

    Raises
    ------
    KeyError
        If any index along the path does not exist, or if no generator is found at the final node.
    """
    node = deepcopy(hierarchy)
    for depth, idx in enumerate(indices):
        if isinstance(idx, int):
            idx = f"rank{idx}" if depth == 0 else f"det{idx}"

        try:
            node = node[idx] if depth == 0 else node["children"][idx]
        except KeyError:
            raise KeyError(f"Index '{idx}' not found in hierarchy at depth {depth}")

    if "generator" in node:
        return node["generator"]
    else:
        raise KeyError(f"No generator found at path {indices}")


def get_detector_level_generators_from_hierarchy(
    hierarchy: dict, rank: Union[int, str]
) -> List[Generator]:
    """
    Retrieve the list of detector-level RNGs under a given MPI rank node.

    Parameters
    ----------
    hierarchy : dict
        The top-level RNG hierarchy dictionary.
    rank : int or str
        Rank identifier (e.g., 0 or "rank0").

    Returns
    -------
    List[numpy.random.Generator]
        A list of RNG generators corresponding to the detectors for that rank.

    Raises
    ------
    KeyError
        If the specified rank is not found or has no detector children.
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
) -> List[Generator]:
    """
    Check or regenerate detector-level RNGs for a given set of observations.

    This function ensures that a list of `numpy.random.Generator` objects is provided for all detectors
    in the observation. If a `user_seed` is provided, a new `RNGHierarchy` is built and the corresponding
    generators are extracted for the current MPI rank.

    Parameters
    ----------
    observations : List[Observation]
        A list of observations, assumed to have consistent number of detectors and communicators.
    user_seed : int, optional
        Optional base seed used to regenerate the RNG hierarchy.
    dets_random : List[Generator], optional
        List of pre-constructed RNGs to be validated.

    Returns
    -------
    List[Generator]
        A list of RNG generators, one for each detector.

    Raises
    ------
    ValueError
        If neither `user_seed` nor `dets_random` is provided.
    AssertionError
        If the number of generators does not match the number of detectors.
    """
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
        """
        Construct the MPI rank layer of the RNG hierarchy.

        This method spawns a seed and corresponding RNG generator for each MPI rank, starting from
        the root seed sequence. Each rank node includes an empty 'children' field for downstream levels.

        Parameters
        ----------
        num_ranks : int
            The number of MPI ranks to include in the hierarchy.
        """
        spawned = self.root_seq.spawn(num_ranks)
        for rank, seq in enumerate(spawned):
            self.hierarchy[f"rank{rank}"] = {
                "seed_seq": seq,
                "generator": Generator(PCG64(seq)),
                "children": {},
            }

    def build_detector_layer(self, num_detectors_per_rank: int):
        """
        Build the detector layer beneath each MPI rank in the RNG hierarchy.

        Each MPI rank node spawns a fixed number of detector-level seed sequences and generators.
        These are added under the "children" dictionary of each rank node.

        Parameters
        ----------
        num_detectors_per_rank : int
            Number of detectors (i.e., child nodes) to create under each MPI rank.
        """
        for _, rank_node in self.hierarchy.items():
            spawned = rank_node["seed_seq"].spawn(num_detectors_per_rank)
            for det, seq in enumerate(spawned):
                rank_node["children"][f"det{det}"] = {
                    "seed_seq": seq,
                    "generator": Generator(PCG64(seq)),
                    "children": {},
                }

    def add_extra_layer(self, num_children: int, layer_name: Optional[str] = None):
        """
        Recursively add an additional layer to all leaves of the hierarchy.

        This method adds a new generation of children nodes to all existing detector-level nodes.
        The added layer can be optionally named (e.g., "subdet") or will be auto-labeled using the depth.

        Parameters
        ----------
        num_children : int
            Number of child nodes to spawn under each leaf node.
        layer_name : str, optional
            Prefix used to label the new layer's children, by default None.

        Notes
        -----
        The new layer is recursively inserted below the current deepest level (detectors by default).
        """

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
        """
        Convenience function to construct a two-level hierarchy with ranks and detectors.

        Equivalent to calling `build_mpi_layer` followed by `build_detector_layer`.

        Parameters
        ----------
        ranks : int
            Number of MPI ranks.
        detectors_per_rank : int
            Number of detectors per MPI rank.
        """
        self.build_mpi_layer(num_ranks=ranks)

        self.build_detector_layer(num_detectors_per_rank=detectors_per_rank)

    def get_generator(self, *indices: Union[int, str]) -> Generator:
        """
        Retrieve a generator from the hierarchy using a sequence of indices.

        Wrapper around the `get_generator_from_hierarchy` utility function.
        """
        return get_generator_from_hierarchy(self.hierarchy, *indices)

    def get_detector_level_generators_on_rank(
        self, rank: Union[int, str]
    ) -> List[Generator]:
        """
        Retrieve the list of RNG generators for all detectors under a given MPI rank.

        Wrapper around the `get_detector_level_generators_from_hierarchy` utility.
        """
        return get_detector_level_generators_from_hierarchy(self.hierarchy, rank)

    def save(self, path: str):
        """
        Serialize and save the RNGHierarchy to a file.

        This method uses `pickle` to serialize the current instance and save it to a binary file.
        Metadata, including the format version and timestamp, are preserved.

        Parameters
        ----------
        path : str
            Path to the file where the hierarchy should be saved.
        """
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, path: str) -> "RNGHierarchy":
        """
        Load a saved RNGHierarchy instance from file.

        This method loads a previously saved RNGHierarchy object and validates its format version.

        Parameters
        ----------
        path : str
            Path to the saved hierarchy pickle file.

        Returns
        -------
        RNGHierarchy
            The loaded hierarchy instance.

        Raises
        ------
        ValueError
            If the saved format version is incompatible with the current implementation.
        """
        with open(path, "rb") as f:
            rng_hierarchy = pickle.load(f)

        if rng_hierarchy.SAVE_FORMAT_VERSION != cls.SAVE_FORMAT_VERSION:
            msg = "The loaded `RNGHierarchy` is incompatible with the current implementation."
            raise ValueError(msg)

        return rng_hierarchy
