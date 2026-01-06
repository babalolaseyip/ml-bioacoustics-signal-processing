from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np
from sklearn.mixture import GaussianMixture

# Optional dependency for HMM
try:
    from hmmlearn.hmm import GaussianHMM  # type: ignore
except Exception:
    GaussianHMM = None


def gmm_unsupervised_predict(X: np.ndarray, n_components: int, seed: int) -> np.ndarray:
    if len(X) == 0:
        return np.zeros((0,), dtype=int)
    gmm = GaussianMixture(n_components=n_components, covariance_type="full", random_state=seed)
    gmm.fit(X)
    return gmm.predict(X).astype(int)


def map_clusters_to_binary(y_cluster: np.ndarray, y_true: np.ndarray) -> np.ndarray:
    """
    Majority-vote mapping per cluster to avoid label inversion.
    """
    y_out = np.zeros_like(y_cluster, dtype=int)
    for k in np.unique(y_cluster):
        idx = np.where(y_cluster == k)[0]
        if len(idx) == 0:
            continue
        maj = int(np.mean(y_true[idx]) >= 0.5)
        y_out[idx] = maj
    return y_out


def hmm_supervised_binary(
    sequences: List[np.ndarray],
    y_true: np.ndarray,
    seed: int,
    n_states: int = 6,
    n_iter: int = 50,
) -> np.ndarray:
    """
    Two-model supervised HMM:
      - Fit one HMM on noise sequences
      - Fit one HMM on call sequences
      - Predict by higher log-likelihood
    """
    if GaussianHMM is None:
        raise ImportError("hmmlearn is required for WT-HMM. Install: pip install hmmlearn")

    idx0 = np.where(y_true == 0)[0]
    idx1 = np.where(y_true == 1)[0]

    if len(idx0) < 2 or len(idx1) < 2:
        return _fallback_unsupervised_energy(sequences, seed=seed)

    def _stack(idxs: np.ndarray) -> Tuple[np.ndarray, List[int]]:
        seqs = [sequences[i] for i in idxs if len(sequences[i]) > 0]
        if not seqs:
            return np.zeros((0, 3), dtype=np.float32), []
        lengths = [len(s) for s in seqs]
        X = np.vstack(seqs).astype(np.float32)
        return X, lengths

    X0, L0 = _stack(idx0)
    X1, L1 = _stack(idx1)

    m0 = GaussianHMM(n_components=n_states, covariance_type="full", n_iter=n_iter, random_state=seed)
    m1 = GaussianHMM(n_components=n_states, covariance_type="full", n_iter=n_iter, random_state=seed)
    m0.fit(X0, lengths=L0)
    m1.fit(X1, lengths=L1)

    y_pred = np.zeros((len(sequences),), dtype=int)
    for i, seq in enumerate(sequences):
        if len(seq) == 0:
            y_pred[i] = 0
            continue
        s0 = m0.score(seq)
        s1 = m1.score(seq)
        y_pred[i] = 1 if s1 > s0 else 0
    return y_pred


def _fallback_unsupervised_energy(sequences: List[np.ndarray], seed: int) -> np.ndarray:
    summaries = np.array(
        [seq.mean(axis=0) if len(seq) else np.zeros((3,), dtype=np.float32) for seq in sequences],
        dtype=np.float32,
    )
    cl = gmm_unsupervised_predict(summaries, n_components=2, seed=seed)
    if len(summaries) == 0:
        return np.zeros((0,), dtype=int)

    e0 = summaries[cl == 0, 0].mean() if np.any(cl == 0) else -np.inf
    e1 = summaries[cl == 1, 0].mean() if np.any(cl == 1) else -np.inf
    call_cluster = 0 if e0 >= e1 else 1
    return (cl == call_cluster).astype(int)
