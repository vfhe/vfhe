from __future__ import annotations

from collections import deque
from math import ceil, log2, sqrt


def it_prod(lst):
    result = 1
    for x in lst:
        result *= x
    return result


def mod_ratios(chain):
    return [chain[i] / chain[i + 1] for i in range(len(chain) - 1)]


def scaling_ratios(chain):
    return [chain[i] ** 2 / chain[i + 1] for i in range(len(chain) - 1)]


def moduli_chain_from_scaling_factors(scaling_chain, q0):
    ratios = scaling_ratios(scaling_chain)
    ratios.reverse()

    moduli_chain = [q0]
    for ratio in ratios:
        next_modulus = ceil(ratio * moduli_chain[-1])
        moduli_chain.append(next_modulus)

    moduli_chain.reverse()
    return moduli_chain


def scaling_factors_from_moduli(moduli_chain, delta=2**30):
    ratios = mod_ratios(moduli_chain)
    ratios.reverse()

    scaling_factors = [delta]
    for ratio in ratios:
        next_scaling_factor = ceil(sqrt(scaling_factors[-1] * ratio))
        scaling_factors.append(next_scaling_factor)

    scaling_factors.reverse()
    return scaling_factors


def width_search_modulus(moduli, residues_choices, max_modulus):
    min_residue = min(residues_choices)
    queue = deque()
    queue.append((moduli, [], 0))
    results = []
    while queue:
        current_m, current_residues, current_modulus = queue.pop()
        if current_modulus > max_modulus:
            continue
        if current_m == 0:
            results.append(current_residues)
            continue
        if current_m < 0 or current_m < min_residue:
            continue
        for r in residues_choices:
            next_m = current_m - r
            if next_m < 0:
                continue
            next_residues = current_residues + [r]
            next_modulus = current_modulus + r
            if next_modulus > max_modulus:
                continue
            queue.append((next_m, next_residues, next_modulus))
    results = [tuple(sorted(residues)) for residues in results]
    results = sorted(set(results))
    return [list(residues) for residues in results]


def union_residues(residues1, residues2):
    values = set(residues1).union(set(residues2))
    result = []
    for x in values:
        result.extend([x] * max(residues1.count(x), residues2.count(x)))
    return sorted(result)


def residues_fit_in_top(residues, top_residues):
    for r in set(residues):
        if residues.count(r) > top_residues.count(r):
            return False
    return True


def residues_to_mask(residues, top_residues):
    used = [False] * len(top_residues)
    mask = 0
    for r in residues:
        for i, top_r in enumerate(top_residues):
            if not used[i] and top_r == r:
                used[i] = True
                mask |= 1 << i
                break
        else:
            return None
    return mask


def select_moduli_and_residues(moduli_chain, residues_choices, max_modulus):
    residues_per_moduli = []
    for m in moduli_chain:
        residues_per_moduli.append(
            width_search_modulus(m, residues_choices, max_modulus)
        )

    results = []

    def rec_agg_residues_search(index, agg_residues):
        if index == len(moduli_chain):
            results.append((agg_residues, sum(agg_residues)))
            return

        for residues in residues_per_moduli[index]:
            next_agg_residues = union_residues(agg_residues, residues)
            next_agg_modulus = sum(next_agg_residues)
            if next_agg_modulus > max_modulus:
                continue
            rec_agg_residues_search(index + 1, next_agg_residues)

    rec_agg_residues_search(0, [])
    if not results:
        return None, None

    top_residues = sorted(min(results, key=lambda x: x[1])[0])
    residue_indices_chain = []
    for residues_list in residues_per_moduli:
        compatible_residues = [
            residues
            for residues in residues_list
            if residues_fit_in_top(residues, top_residues)
        ]
        if not compatible_residues:
            return None, None
        selected_residues = [False for _ in top_residues]
        for r in compatible_residues[0]:
            for i, top_r in enumerate(top_residues):
                if not selected_residues[i] and top_r == r:
                    selected_residues[i] = True
                    break
        residue_indices = [
            i for i, selected in enumerate(selected_residues) if selected
        ]
        residue_indices_chain.append(residue_indices)
    return top_residues, residue_indices_chain


def search_log_residues(log_moduli_chain, logr_min, logr_max, max_modulus):
    log_search_range = list(range(logr_min, logr_max + 1))

    log_top_residues, residue_indices_chain = select_moduli_and_residues(
        log_moduli_chain, log_search_range, max_modulus
    )
    if log_top_residues is None:
        raise ValueError(
            "No residue decomposition found. Try increasing the search range or max_modulus."
        )

    return log_top_residues, residue_indices_chain


def search_log_residues_minq0_full_infos(
    log_scaling_factor_chain, logr_min, logr_max, max_modulus, log_q0_max=3000
):
    log_q0 = logr_min
    log_search_range = list(range(logr_min, logr_max + 1))
    scaling_factor_chain = [2**x for x in log_scaling_factor_chain]

    while True:
        q0 = 2**log_q0
        moduli_chain = moduli_chain_from_scaling_factors(scaling_factor_chain, q0)
        log_moduli_chain = [ceil(log2(q)) for q in moduli_chain]

        log_top_residues, residue_indices_chain = select_moduli_and_residues(
            log_moduli_chain,
            log_search_range,
            max_modulus,
        )

        if log_top_residues is not None:
            return {
                "log_q0": log_q0,
                "moduli_chain": moduli_chain,
                "log_moduli_chain": log_moduli_chain,
                "log_top_residues": log_top_residues,
                "residue_indices_chain": residue_indices_chain,
            }
        log_q0 += 1
        if log_q0 > log_q0_max:
            raise ValueError("Cannot find suitable log_q0 within the specified range.")


def search_log_residues_minq0(
    log_scaling_factor_chain, logr_min, logr_max, max_modulus, log_q0_max=1000
):
    """
    Search for the smallest log_q0 such that the moduli chain
    derived from the scaling factor chain can be decomposed into
    residues that fit within the specified range and max_modulus.
    """
    log_q0 = logr_min
    log_search_range = list(range(logr_min, logr_max + 1))
    scaling_factor_chain = [2**x for x in log_scaling_factor_chain]

    while True:
        q0 = 2**log_q0
        moduli_chain = moduli_chain_from_scaling_factors(scaling_factor_chain, q0)
        log_moduli_chain = [int(log2(q)) for q in moduli_chain]
        log_top_residues, residue_indices_chain = select_moduli_and_residues(
            log_moduli_chain, log_search_range, max_modulus
        )
        if log_top_residues is not None:
            return log_top_residues, residue_indices_chain
        log_q0 += 1
        if log_q0 > log_q0_max:
            raise ValueError("Cannot find suitable log_q0 within the specified range.")


if __name__ == "__main__":
    log_scaling_factor_chain = [29, 29, 29]
    logr_min = 40
    logr_max = 64
    max_modulus = 200

    res = search_log_residues_minq0_full_infos(
        log_scaling_factor_chain=log_scaling_factor_chain,
        logr_min=logr_min,
        logr_max=logr_max,
        max_modulus=max_modulus,
    )

    print("log_q0:", res["log_q0"])
    print("moduli_chain:", res["moduli_chain"])
    print("log_moduli_chain:", res["log_moduli_chain"])
    print("log_top_residues:", res["log_top_residues"])
    print("residue_indices_chain:", res["residue_indices_chain"])
