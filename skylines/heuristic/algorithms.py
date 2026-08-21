from itertools import permutations
import numpy as np
from skylines.dataset.dataset import Dataset


def rank_by_top_priority(ref):
    n = len(ref)
    # item importance based on reference (top = most important)
    item_value = {item: n - i for i, item in enumerate(ref)}
    # position weights (top position = highest weight)
    pos_weight = {pos: n - pos + 1 for pos in range(1, n+1)}

    def score(perm):
        return sum(
            item_value[item] * pos_weight[pos]
            for pos, item in enumerate(perm, start=1)
        )

    perms = list(permutations(ref))
    perms.sort(key=lambda p: -score(p))

    return perms


def min_corr(dataset: Dataset):
    data, variates, preferences = dataset.to_numpy()
    corr = np.corrcoef(data, rowvar=False)

    scores = []

    for pref_i, i in preferences.items():
        score = 0
        for pref_j, j in preferences.items():
            if i != j:
                score += corr[i, j]
        scores.append((pref_i, score))

    preferences = [name for name, _ in sorted(scores, key=lambda x: x[1])]

    return preferences


def min_abs_corr(dataset: Dataset):
    data, variates, preferences = dataset.to_numpy()
    corr = np.corrcoef(data, rowvar=False)

    scores = []

    for pref_i, i in preferences.items():
        score = 0
        for pref_j, j in preferences.items():
            if i != j:
                score += abs(corr[i, j])
        scores.append((pref_i, score))

    preferences = [name for name, _ in sorted(scores, key=lambda x: x[1])]

    return preferences


def max_corr(dataset: Dataset):
    data, variates, preferences = dataset.to_numpy()
    corr = np.corrcoef(data, rowvar=False)

    scores = []

    for pref_i, i in preferences.items():
        score = 0
        for pref_j, j in preferences.items():
            if i != j:
                score += corr[i, j]
        scores.append((pref_i, score))

    preferences = [name for name, _ in sorted(scores, key=lambda x: x[1], reverse=True)]

    return preferences


def incremental_min_corr_first(dataset: Dataset):
    data, variates, preferences = dataset.to_numpy()
    corr = np.corrcoef(data, rowvar=False)
    
    pref_reverse = {v: k for k, v in preferences.items()}
    remaining = list(preferences.values())

    # Start with an empty chain
    chain = []

    # Greedily add least correlated variable w.r.t. remaining variables
    while remaining:
        next_var = min(remaining, key=lambda j: np.sum([corr[j, k] for k in remaining if k != j]))
        chain.append(next_var)
        remaining.remove(next_var)

    # Convert back to variable names
    ordered_names = [pref_reverse[i] for i in chain]

    return ordered_names


def incremental_max_corr_last(dataset: Dataset):
    data, variates, preferences = dataset.to_numpy()
    corr = np.corrcoef(data, rowvar=False)

    pref_reverse = {v: k for k, v in preferences.items()}
    remaining = list(preferences.values())

    # Start with an empty chain
    chain = []

    # Greedily add most correlated variable w.r.t. remaining variables
    while remaining:
        prev_var = max(remaining, key=lambda j: np.sum([corr[j, k] for k in remaining if k != j]))
        chain.insert(0, prev_var)
        remaining.remove(prev_var)

    # Convert back to variable names
    ordered_names = [pref_reverse[i] for i in chain]

    return ordered_names


def min_corr_pairwise(dataset: Dataset):
    data, variates, preferences = dataset.to_numpy()
    corr = np.corrcoef(data, rowvar=False)

    perms = list(permutations(preferences.items()))

    perm_score = dict()
    for perm in perms:
        score = 0
        for i in range(len(perm) - 1):
            score += corr[perm[i][1], perm[i + 1][1]] * (len(perm) - i)
        perm_score[perm] = score

    order = min(perm_score, key=perm_score.get)
    order = [k for k, _ in order]

    return order
