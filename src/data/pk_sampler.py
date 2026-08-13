"""PK batch sampling: every batch is P identities x K images (plan §4, poster panel 2).

One sampler serves all three losses, which is the point -- it makes the loss the
only thing that differs between experiments in E1:

  * contrastive: same-identity pairs within the batch are the y=1 pairs, and
    cross-identity pairs supply an equal number of y=0 pairs.
  * triplet: every image is an anchor; K > 1 guarantees each anchor has at least
    one positive present, which is what makes "hardest positive" in batch-hard
    mining a meaningful quantity at all rather than a no-op.
  * InfoNCE: each anchor draws its in-batch positive from its own K-group, and
    every one of the other P-1 identity groups contributes negatives.

With K = 1 a batch would be P singleton identities and there would be no
positives at all -- the sampler is what makes an in-batch positive exist.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence

import numpy as np
from torch.utils.data import Sampler


class PKSampler(Sampler[list[int]]):
    """Yields batches of exactly P identities x K images as lists of dataset indices.

    Use as `DataLoader(dataset, batch_sampler=PKSampler(...))`; it is a *batch*
    sampler, so do not also pass `batch_size` or `shuffle`.

    Identities are reshuffled every epoch, and each identity's image pool is
    reshuffled independently, so repeated epochs do not replay the same P-groups
    or the same K-subsets.

    Identities with fewer than K images cannot contribute K *distinct* images.
    The default is to drop them: a duplicated image is a positive pair at
    distance exactly 0, which is a free, uninformative term for contrastive and
    triplet and a trivially-solved row for InfoNCE. `allow_replacement=True`
    keeps them by sampling with replacement, which is only advisable if dropping
    would cost a large fraction of the identities -- check `n_dropped_identities`.

    Attributes:
        n_identities: usable identities after the < K filter.
        n_dropped_identities: identities excluded for having fewer than K images.
        batch_size: P * K, the exact size of every yielded batch.
    """

    def __init__(
        self,
        labels: Sequence[int] | np.ndarray,
        p: int = 64,
        k: int = 4,
        *,
        seed: int = 0,
        allow_replacement: bool = False,
        num_batches: int | None = None,
        rank: int = 0,
        world_size: int = 1,
    ) -> None:
        """
        Args:
            labels: per-sample identity label, `labels[i]` for dataset index `i`.
            p: identities per batch (default 64, plan §4).
            k: images per identity per batch (default 4 -> N = 256).
            seed: base RNG seed; the epoch index is mixed in by `set_epoch`.
            allow_replacement: keep identities with < K images by sampling their
                images with replacement instead of dropping them.
            num_batches: batches per epoch. Default `n_identities // p`, i.e. one
                pass in which each usable identity appears exactly once.
            rank: this process's rank under DDP.
            world_size: number of DDP processes. When > 1, every rank draws from
                the SAME seeded identity permutation but takes a different slice
                of it, so the P identities on rank 0 are disjoint from those on
                ranks 1 and 2 within the same step. That disjointness is what
                makes global negatives mean what E5 claims: with independent
                per-rank shuffles, the same identity could land on two ranks and
                some of the "extra negatives" gained by gathering would actually
                be positives.

        Raises:
            ValueError: if p or k < 1, k < 2 (no positives possible), or fewer
                than p usable identities remain after filtering.
        """
        if p < 1 or k < 1:
            raise ValueError(f"p and k must be >= 1, got p={p}, k={k}")
        if k < 2:
            raise ValueError(
                f"k={k} gives no in-batch positives; PK sampling needs k >= 2 (plan §4)"
            )

        self.p = p
        self.k = k
        self.seed = seed
        self.rank = rank
        self.world_size = world_size
        self.allow_replacement = allow_replacement
        self._epoch = 0

        labels_arr = np.asarray(labels)
        if labels_arr.ndim != 1:
            raise ValueError(f"labels must be 1-D, got shape {labels_arr.shape}")

        # identity -> dataset indices carrying that identity
        order = np.argsort(labels_arr, kind="stable")
        sorted_labels = labels_arr[order]
        uniq, starts = np.unique(sorted_labels, return_index=True)
        groups = np.split(order, starts[1:])

        keep = [(int(i), g) for i, g in zip(uniq, groups) if allow_replacement or len(g) >= k]
        self.n_dropped_identities = len(uniq) - len(keep)

        if len(keep) < p:
            raise ValueError(
                f"only {len(keep)} identities have >= k={k} images, need at least p={p}. "
                f"Lower p/k, or pass allow_replacement=True "
                f"({self.n_dropped_identities} identities were dropped)."
            )

        self._identities = np.array([i for i, _ in keep], dtype=np.int64)
        self._pools: dict[int, np.ndarray] = {i: g for i, g in keep}

        self.n_identities = len(self._identities)
        self.batch_size = p * k
        # Each step consumes P identities per rank, so a full pass over
        # identities yields correspondingly fewer steps under DDP.
        self._num_batches = (
            self.n_identities // (p * world_size) if num_batches is None else int(num_batches)
        )

        if self.n_identities < p * world_size:
            raise ValueError(
                f"{self.n_identities} usable identities cannot fill p={p} x "
                f"world_size={world_size} = {p * world_size} disjoint slots per step"
            )

    def set_epoch(self, epoch: int) -> None:
        """Reseed the shuffle for `epoch`.

        Call once per epoch before iterating. Mirrors `DistributedSampler.set_epoch`,
        and matters for the same reason: without it every epoch replays the
        identical batch composition, so the model sees far fewer distinct
        (anchor, positive, negative) combinations than the epoch count suggests.
        """
        self._epoch = int(epoch)

    def __len__(self) -> int:
        """Number of batches per epoch."""
        return self._num_batches

    def __iter__(self) -> Iterator[list[int]]:
        """Yield `len(self)` batches, each exactly P*K dataset indices.

        Identity order is reshuffled per epoch and consumed in contiguous chunks
        of P, so within one epoch an identity appears in at most one batch (until
        the identity list is exhausted and reshuffled, when `num_batches` is set
        higher than the default).
        """
        # The identity permutation is seeded identically on every rank; only the
        # slice each rank takes differs. Image choice within an identity uses a
        # rank-specific stream, which is harmless since the identity sets are
        # already disjoint.
        order_rng = np.random.default_rng([self.seed, self._epoch])
        rng = np.random.default_rng([self.seed, self._epoch, self.rank])

        identity_order = order_rng.permutation(self._identities)
        stride = self.p * self.world_size
        cursor = 0

        for _ in range(self._num_batches):
            # Refill and reshuffle when fewer than one full step remains.
            if cursor + stride > len(identity_order):
                identity_order = order_rng.permutation(self._identities)
                cursor = 0

            lo = cursor + self.rank * self.p
            batch: list[int] = []
            for identity in identity_order[lo : lo + self.p]:
                pool = self._pools[int(identity)]
                replace = len(pool) < self.k
                chosen = rng.choice(pool, size=self.k, replace=replace)
                batch.extend(int(i) for i in chosen)

            cursor += stride
            yield batch
