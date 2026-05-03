"""Lightweight feature adapter for OpenScene fused features.

Architecture: 2-layer residual MLP (g_theta) + linear classifier head (h_psi).
  g_theta: Linear(768,768) ? LayerNorm ? ReLU ? Linear(768,768) + residual skip
  h_psi:   Linear(768, num_classes)

The fc2 weights are zero-initialized so the adapter starts as identity,
preserving the OpenScene baseline at epoch 0.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class FeatureAdapter(nn.Module):
    """Lightweight adapter: 2-layer residual MLP + linear classifier head."""

    def __init__(self, feat_dim=768, num_classes=21):
        super().__init__()
        self.fc1 = nn.Linear(feat_dim, feat_dim)
        self.norm = nn.LayerNorm(feat_dim)
        self.fc2 = nn.Linear(feat_dim, feat_dim)
        self.classifier = nn.Linear(feat_dim, num_classes)
        # Zero init fc2: adapter is identity at initialization
        nn.init.zeros_(self.fc2.weight)
        nn.init.zeros_(self.fc2.bias)

    def get_adapted_features(self, x):
        """768-D features after g_theta (before the classifier head)."""
        residual = x
        out = F.relu(self.norm(self.fc1(x)))
        out = self.fc2(out) + residual
        return out

    def forward(self, x):
        logits = self.classifier(self.get_adapted_features(x))
        return logits


def predictive_entropy(logits):
    """Per-row softmax entropy H(pi), shape (N,)."""
    probs = F.softmax(logits, dim=-1)
    return -(probs * probs.clamp(min=1e-8).log()).sum(dim=-1)


def vanilla_entropy_minimization(logits):
    """TENT-style mean entropy over all points (Run 3 ablation; no mask)."""
    return predictive_entropy(logits).mean()


def confidence_masked_entropy(logits, tau):
    """Confidence-masked entropy loss (L_ent).

    Computes mean entropy over unlabeled points, but only for those whose
    predictive entropy H(y_i) < tau (the confident subset).

    The confidence mask is .detach()-ed: no gradient flows through the gating
    decision, which would otherwise create pathological incentives to drive
    high-entropy points toward *higher* entropy to escape the mask.

    Returns 0.0 (no gradient) when no points pass the confidence threshold,
    which is the expected curriculum behavior early in training.
    """
    H = predictive_entropy(logits)
    conf_mask = (H < tau).detach()  # CRITICAL: no grad through mask
    if conf_mask.sum() == 0:
        return torch.tensor(0.0, device=logits.device, requires_grad=True)
    return (conf_mask.float() * H).sum() / conf_mask.float().sum()


def inverted_masked_entropy(logits, tau):
    """Entropy minimization on the *uncertain* subset (inverse H2 mask).

    Same detached hard mask as ``confidence_masked_entropy``, but selects
    points with H(y_i) > tau. Mean entropy is averaged only over that subset.
    """
    H = predictive_entropy(logits)
    inv_mask = (H > tau).detach()
    if inv_mask.sum() == 0:
        return torch.tensor(0.0, device=logits.device, requires_grad=True)
    return (inv_mask.float() * H).sum() / inv_mask.float().sum()


def soft_weighted_entropy(logits, temperature=0.5):
    """Confidence-weighted entropy with continuous detached weights."""
    H = predictive_entropy(logits)
    weights = torch.exp(-H / float(temperature)).detach()
    denom = weights.sum().clamp(min=1e-8)
    return (weights * H).sum() / denom


def pseudo_label_loss(logits, threshold=0.9):
    """Pseudo-label CE on confident points; returns (loss, fraction_confident)."""
    probs = F.softmax(logits, dim=-1)
    max_probs, pseudo_labels = probs.max(dim=-1)
    conf_mask = (max_probs > float(threshold)).detach()
    if conf_mask.sum() == 0:
        return torch.tensor(0.0, device=logits.device, requires_grad=True), 0.0
    loss = F.cross_entropy(logits[conf_mask], pseudo_labels[conf_mask].detach())
    frac = float(conf_mask.float().mean().item())
    return loss, frac


def temperature_sharpening_loss(logits, sharpening_temp=0.5):
    """Soft CE against detached temperature-sharpened targets."""
    with torch.no_grad():
        probs = F.softmax(logits, dim=-1)
        sharp = probs.pow(1.0 / float(sharpening_temp))
        sharp = sharp / sharp.sum(dim=-1, keepdim=True).clamp(min=1e-8)
    log_probs = F.log_softmax(logits, dim=-1)
    return -(sharp * log_probs).sum(dim=-1).mean()
