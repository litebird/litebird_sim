# -*- encoding: utf-8 -*-

from collections import namedtuple
from typing import List

Span = namedtuple("Span", ["start_idx", "num_of_elements"])
"""A sub-range in a sequence of elements.

It has two fields: `start_idx` and `num_of_elements`.

"""


def distribute_evenly(num_of_elements, num_of_groups):
    """Evenly distribute a set of equal elements between groups.

    Assuming that we have `num_of_elements` objects that we want to
    assign to `num_of_groups` separate groups, this function proposes
    a distribution that tries to assign the most uniform number of
    elements to each group.

    This function works even if ``num_of_elements < num_of_groups``.

    .. doctest::

      >>> distribute_evenly(5, 2)
      [Span(start_idx=0, num_of_elements=3), Span(start_idx=3, num_of_elements=2)]

      >>> distribute_evenly(1, 2)
      [Span(start_idx=0, num_of_elements=1), Span(start_idx=1, num_of_elements=0)]

    Args:
        num_of_elements (int): The number of elements to distribute
        num_of_groups (int): The number of groups to use in the result

    Returns:
        a list of 2-element tuples containing `num_of_groups`
        elements, each of them being a 2-element tuple containing (1)
        the index of the element in the group and (2) the number of
        elements in the group. Being named tuples, you can access the
        index using the field name ``start_idx``, and the number of
        elements using the name ``num_of_elements``.

    """
    assert num_of_elements > 0
    assert num_of_groups > 0

    base_length = num_of_elements // num_of_groups
    leftovers = num_of_elements % num_of_groups

    # If leftovers == 0, then the number of elements is divided evenly
    # by num_of_groups, and the solution is trivial. If it's not, then
    # each of the "leftoverss" is placed in one of the first groups.
    #
    # Example: let's split 8 elements in 3 groups. In this case,
    # base_length=2 and leftovers=2 (elements #7 and #8):
    #
    # +----+----+----+  +----+----+----+  +----+----+
    # | #1 | #2 | #3 |  | #4 | #5 | #6 |  | #7 | #8 |
    # +----+----+----+  +----+----+----+  +----+----+
    #
    #

    result = []
    for i in range(num_of_groups):
        if i < leftovers:
            # Make place for one of the leftovers
            cur_length = base_length + 1
            cur_pos = cur_length * i
        else:
            # No need to accomodate for leftovers, but consider their
            # presence in fixing the starting position for this group
            cur_length = base_length
            cur_pos = base_length * i + leftovers

        result.append(Span(start_idx=cur_pos, num_of_elements=cur_length))

    assert len(result) == num_of_groups, (
        f"wrong result(len(result)={len(result)}) in "
        + f"distribute_evenly(num_of_elements={num_of_elements}, "
        + f"num_of_groups={num_of_groups})"
    )
    assert sum([pair.num_of_elements for pair in result]) == num_of_elements
    return result


# The following implementation of the painter's partition problem is
# heavily inspired by the code at
# https://www.geeksforgeeks.org/painters-partition-problem-set-2/?ref=rp


def _num_of_workers(arr, n, maxLen, weight_fn):
    total = 0
    num_of_workers = 1

    for cur_element in arr:
        cur_weight = weight_fn(cur_element)
        total += cur_weight

        if total > maxLen:
            num_of_workers += 1
            # Reset "total" for the next worker
            total = cur_weight

    return num_of_workers


def _find_max_and_sum(arr, weight_fn):
    weights = list(map(weight_fn, arr))
    return (max(weights), sum(weights))


def _partition(arr, n, k, weight_fn):
    lo, hi = _find_max_and_sum(arr, weight_fn)

    while lo < hi:
        mid = lo + (hi - lo) / 2
        required_workers = _num_of_workers(arr, n, mid, weight_fn)

        # find better optimum in lower half
        # here mid is included because we
        # may not get anything better
        if required_workers <= k:
            hi = mid

        # find better optimum in upper half
        # here mid is excluded because it gives
        # required Painters > k, which is invalid
        else:
            lo = mid + 1

    # required
    return lo


def __identity_fn(x):
    return x


def distribute_optimally(elements, num_of_groups, weight_fn=None) -> List[Span]:
    """Evenly distribute a set of equal elements between groups.

    Assuming that we have a set of elements, each having its own
    weight that is computed using `weight_fn` (i.e., for any element
    ``x`` in ``elements``, its weight is ``weight_fn(x)``), this
    function proposes a distribution that tries to assign the elements
    to each group so that the weight in each group is more or less the
    same.

    This function works even if ``num_of_elements < num_of_groups``.

    .. doctest::

      >>> distribute_optimally([10, 10, 10, 10], 2)
      [Span(start_idx=0, num_of_elements=2), Span(start_idx=2, num_of_elements=2)]

      >>> distribute_optimally([10, 10, 10, 20], 2)
      [Span(start_idx=0, num_of_elements=3), Span(start_idx=3, num_of_elements=1)]

      >>> distribute_optimally([40, 10, 10, 10], 2)
      [Span(start_idx=0, num_of_elements=1), Span(start_idx=1, num_of_elements=3)]

      >>> distribute_optimally([('a', 10), ('b', 10), ('c', 10)], 2, lambda x: x[1])
      [Span(start_idx=0, num_of_elements=2), Span(start_idx=2, num_of_elements=1)]

    This function is a generalization of :meth:`.distribute_evenly`;
    in that case, the function assumes that each element weights
    equally.

    Args:
        elements (list): A list of objects to be split in groups
        num_of_groups (int): The number of groups to use in the result
        weight_fn: A function-like object that computes the weight
            given one of the objects in `elements`. If unspecified, the
            identity function will be used.

    Returns:
        a list of 2-element tuples containing `num_of_groups`
        elements, each of them being a 2-element tuple containing (1)
        the index of the element in the group and (2) the number of
        elements in the group. Being named tuples, you can access the
        index using the field name ``start_idx``, and the number of
        elements using the name ``num_of_elements``.

    """
    if not weight_fn:
        weight_fn = __identity_fn

    max_weight = _partition(elements, len(elements), num_of_groups, weight_fn)

    result = []  # type: List[Span]
    start_idx = 0
    weight = 0
    cur_num = 0
    for cur_idx, cur_element in enumerate(elements):
        cur_weight = weight_fn(cur_element)
        if weight + cur_weight > max_weight:
            result.append(Span(start_idx=start_idx, num_of_elements=cur_num))
            cur_num = 1
            weight = cur_weight
            start_idx = cur_idx
        else:
            weight += cur_weight
            cur_num += 1

    result.append(Span(start_idx=start_idx, num_of_elements=cur_num))

    # The way we implemented this implies the possibility that not every processor
    # is being used. We just fill with empty elements the end of `result`
    result += [Span(start_idx=0, num_of_elements=0)] * (num_of_groups - len(result))

    assert len(result) == num_of_groups, (
        f"wrong result(len(result)={len(result)}) in "
        + f"distribute_optimally(len(elements)={len(elements)}, "
        + f"num_of_groups={num_of_groups})"
    )
    assert sum([r.num_of_elements for r in result]) == len(elements)
    return result
