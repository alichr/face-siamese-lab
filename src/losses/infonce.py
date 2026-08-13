"""InfoNCE, van den Oord et al. 2018 (plan §6c, poster panel 4c).

For anchor i with positive i+:

    L_i = -log [ exp(s(z_i, z_i+) / tau) / sum_{j in B, j != i} exp(s(z_i, z_j) / tau) ]

Default tau = 0.07, fixed.

Two implementation details are load-bearing and easy to get subtly wrong.

**The denominator includes the positive.** It runs over every j != i, the
positive among them -- not over negatives only. This is what the poster writes
and it is what makes the expression a proper softmax cross-entropy: the ratio is
then bounded in (0, 1] and the loss in [0, inf). Excluding the positive would let
the ratio exceed 1 and the loss go negative, and a loss unbounded below is a loss
the optimizer can chase forever. Note this changes the loss *value* but not its
argmin: driving s_pos up and s_neg down minimizes either form. The lower bound is
the useful part -- at tau=1 with a single negative at s=0 and the positive at
s=1, the floor is -log(e/(e+1)) = 0.31326, exactly the plan §12 vector.

**Only the diagonal is masked, not all same-identity entries.** With PK sampling
a batch holds K images per identity, so anchor i has K-1 same-identity images
present. Standard InfoNCE treats the K-2 that are not the chosen positive as
*negatives* -- the objective is "identify the true match among candidates", and
those are genuine competing candidates. `--supcon` (Khosla et al. 2020) changes
this, promoting every same-identity sample to a positive; it is off by default
because the poster describes plain InfoNCE.

Temperature: tau scales the logits, so small tau sharpens the softmax and
concentrates gradient on the hardest negatives (those with the highest
similarity). Too small and training chases a handful of near-duplicates; too
large and the distribution flattens until every negative contributes equally and
the signal washes out. E4 measures the resulting U-shape.
"""

from __future__ import annotations

import torch
from torch import nn

from src.losses.geometry import cosine_similarity_matrix

DEFAULT_TEMPERATURE: float = 0.07


class InfoNCELoss(nn.Module):
    """InfoNCE over a PK batch, with an optional SupCon variant.

    Logged per step:
        mean_s_pos / mean_s_neg   mean similarity to the positive / to negatives
        gap                       mean_s_pos - mean_s_neg, the quantity the loss
                                  is really maximizing
        pos_rank_frac             fraction of anchors whose positive is already
                                  the single most similar sample in the batch,
                                  i.e. top-1 in-batch retrieval accuracy
    """

    def __init__(
        self,
        temperature: float = DEFAULT_TEMPERATURE,
        supcon: bool = False,
        seed: int = 0,
    ) -> None:
        """
        Args:
            temperature: tau, default 0.07 fixed (E4 sweeps it).
            supcon: treat every same-identity sample as a positive (Khosla et al.).
            seed: RNG seed for positive selection.

        Raises:
            ValueError: if temperature <= 0.
        """
        super().__init__()
        if temperature <= 0:
            raise ValueError(f"temperature must be > 0, got {temperature}")
        self.temperature = temperature
        self.supcon = supcon
        self._generator = torch.Generator().manual_seed(seed)

    def forward(
        self,
        embeddings: torch.Tensor,
        labels: torch.Tensor,
        global_embeddings: torch.Tensor | None = None,
        global_labels: torch.Tensor | None = None,
        rank_offset: int = 0,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """Compute InfoNCE for the local anchors against the (possibly global) pool.

        Args:
            embeddings: (N, d) local anchors, L2-normalized.
            labels: (N,) local identity labels.
            global_embeddings: (M, d) gathered pool for DDP global negatives
                (plan §6c). Defaults to `embeddings` (single-process / local mode).
            global_labels: (M,) labels for the pool. Required with `global_embeddings`.
            rank_offset: row index of this rank's first anchor within the pool.
                **Critical for DDP**: self-similarity must be masked at the
                anchor's position in the GLOBAL matrix, not at its local index.
                Off-by-rank here leaves each anchor's own s=1 in the denominator
                and removes a real negative, silently changing the objective.

        Returns:
            `(loss, logs)`.
        """
        pool = embeddings if global_embeddings is None else global_embeddings
        pool_labels = labels if global_embeddings is None else global_labels
        if pool_labels is None:
            raise ValueError("global_labels is required when global_embeddings is given")

        n_local = embeddings.shape[0]
        device = embeddings.device

        # (N, M) similarity of every local anchor against the whole pool.
        similarity = cosine_similarity_matrix(embeddings, pool)
        logits = similarity / self.temperature

        # Mask each anchor's own entry -- offset into the global matrix.
        rows = torch.arange(n_local, device=device)
        self_cols = rows + rank_offset
        logits[rows, self_cols] = float("-inf")

        same_identity = labels[:, None] == pool_labels[None, :]
        positive_mask = same_identity.clone()
        positive_mask[rows, self_cols] = False

        valid = positive_mask.any(dim=1)
        if not valid.any():
            zero = embeddings.sum() * 0.0
            return zero, {"mean_s_pos": 0.0, "mean_s_neg": 0.0, "gap": 0.0, "pos_rank_frac": 0.0}

        log_denominator = torch.logsumexp(logits, dim=1)

        if self.supcon:
            # SupCon: average the loss over every positive of each anchor.
            log_probs = logits - log_denominator[:, None]
            n_positives = positive_mask.sum(dim=1).clamp_min(1)
            # masked_fill, not multiplication: `logits` carries -inf on the masked
            # diagonal, and `-inf * 0` is NaN, not 0. Multiplying by a boolean mask
            # is only safe when the tensor is finite everywhere.
            per_anchor = (
                -log_probs.masked_fill(~positive_mask, 0.0).sum(dim=1) / n_positives
            )
        else:
            # Plain InfoNCE: one randomly chosen in-batch positive per anchor.
            noise = torch.rand(
                positive_mask.shape, generator=self._generator, device="cpu"
            ).to(device)
            positive_idx = (noise * positive_mask).argmax(dim=1)
            per_anchor = -(logits[rows, positive_idx] - log_denominator)

        loss = per_anchor[valid].mean()

        with torch.no_grad():
            negative_mask = ~same_identity
            s_pos = (similarity * positive_mask).sum(1) / positive_mask.sum(1).clamp_min(1)
            s_neg = (similarity * negative_mask).sum(1) / negative_mask.sum(1).clamp_min(1)
            # Top-1: is the positive already the most similar entry in the pool?
            masked = similarity.clone()
            masked[rows, self_cols] = float("-inf")
            top1 = masked.argmax(dim=1)
            correct = positive_mask[rows, top1]
            logs = {
                "mean_s_pos": s_pos[valid].mean().item(),
                "mean_s_neg": s_neg[valid].mean().item(),
                "gap": (s_pos - s_neg)[valid].mean().item(),
                "pos_rank_frac": correct[valid].float().mean().item(),
            }

        return loss, logs


@torch.no_grad()
def similarity_heatmap_data(
    embeddings: torch.Tensor, labels: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Identity-ordered similarity matrix for the V3 heatmap (poster panel 5.4).

    Sorting rows and columns by identity makes the K x K same-identity blocks
    land on the diagonal, so a working model shows a visibly block-diagonal S.

    Returns:
        `(S_ordered, sorted_labels)`.
    """
    order = torch.argsort(labels)
    z = embeddings[order]
    return cosine_similarity_matrix(z), labels[order]
