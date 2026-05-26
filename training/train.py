"""
Curriculum-based multi-phase training script for ForgeLens.

Phases
------
1. authentic vs ai_generated   (mask_class=2,  10 epochs, lr=1e-4)
2. authentic vs manipulated     (mask_class=1,  10 epochs, lr=1e-4)
3. all three classes             (mask_class=None, 15 epochs, lr=5e-5)
4. fine-tune with unfrozen last 3 backbone layers (5 epochs, lr=1e-5)
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader

from backend.app.forensics.models.forgelens_model import ForgeLensModel
from training.datasets.loaders import ForensicDataset, get_weighted_sampler

# Limit CPU parallelism — prevents thread-contention on machines with many
# cores when running alongside other processes.
torch.set_num_threads(3)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Masked loss
# ---------------------------------------------------------------------------

class MaskedCrossEntropy(nn.Module):
    """Cross-entropy loss that optionally ignores samples of a masked class.

    This is critical for curriculum learning where early phases intentionally
    exclude a class from the training set — yet the model head still has
    ``num_classes=3`` logits.  By masking the absent class we avoid
    back-propagating from labels the model has never seen in the current
    phase.

    Parameters
    ----------
    mask_class : int | None
        If not *None*, any sample whose target equals *mask_class* is
        filtered out before the loss is computed.
    """

    def __init__(self, mask_class: int | None = None) -> None:
        super().__init__()
        self.mask_class = mask_class

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        if self.mask_class is not None:
            keep = targets != self.mask_class
            logits = logits[keep]
            targets = targets[keep]

            if targets.numel() == 0:
                # Nothing to learn — return a zero loss that still
                # participates in the computation graph.
                return torch.tensor(0.0, requires_grad=True, device=logits.device)

        return nn.functional.cross_entropy(logits, targets)


# ---------------------------------------------------------------------------
# Phase config
# ---------------------------------------------------------------------------

_PHASE_DEFAULTS: dict[int, dict] = {
    1: {"mask_class": 2, "epochs": 10, "batch_size": 16, "lr": 1e-4},
    2: {"mask_class": 1, "epochs": 10, "batch_size": 16, "lr": 1e-4},
    3: {"mask_class": None, "epochs": 15, "batch_size": 8, "lr": 5e-5},
    4: {"mask_class": None, "epochs": 5, "batch_size": 4, "lr": 1e-5},
}


# ---------------------------------------------------------------------------
# Backbone unfreezing (Phase 4)
# ---------------------------------------------------------------------------

def _unfreeze_last_n_layers(branch: nn.Module, n: int = 3) -> None:
    """Unfreeze the last *n* sub-layers of ``branch.features``."""
    features = branch.features
    total = len(features)
    for idx in range(max(0, total - n), total):
        for param in features[idx].parameters():
            param.requires_grad = True


# ---------------------------------------------------------------------------
# Single-phase trainer
# ---------------------------------------------------------------------------

def train_one_phase(
    phase: int,
    model: ForgeLensModel,
    data_root: str | Path,
    epochs: int | None = None,
    batch_size: int | None = None,
    lr: float | None = None,
    checkpoint_dir: str | Path = "./training/checkpoints",
) -> ForgeLensModel:
    """Run one curriculum-learning phase and save the best checkpoint.

    Parameters are resolved from ``_PHASE_DEFAULTS`` when not explicitly
    provided.
    """
    cfg = _PHASE_DEFAULTS[phase]
    epochs = epochs or cfg["epochs"]
    batch_size = batch_size or cfg["batch_size"]
    lr = lr or cfg["lr"]
    mask_class: int | None = cfg["mask_class"]

    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    device = next(model.parameters()).device

    # ---- Phase 4: unfreeze last 3 backbone layers ----------------------
    if phase == 4:
        logger.info("Phase 4 — unfreezing last 3 backbone layers on each branch.")
        _unfreeze_last_n_layers(model.rgb_branch, n=3)
        _unfreeze_last_n_layers(model.ela_branch, n=3)
        _unfreeze_last_n_layers(model.fft_branch, n=3)

    # ---- Dataset & loader -----------------------------------------------
    # Phase 4 reuses the phase-3 dataset config (all three classes)
    dataset_phase = phase if phase <= 3 else 3
    train_set = ForensicDataset(data_root, split="train", phase=dataset_phase)
    val_set = ForensicDataset(data_root, split="val", phase=dataset_phase)

    sampler = get_weighted_sampler(train_set)
    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=0,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
    )

    # ---- Loss / optimiser / scheduler -----------------------------------
    criterion = MaskedCrossEntropy(mask_class=mask_class)
    optimizer = AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)
    scheduler = ReduceLROnPlateau(optimizer, patience=3, factor=0.5)

    best_loss = float("inf")
    best_path = checkpoint_dir / f"phase{phase}_best.pt"

    # ---- Training loop --------------------------------------------------
    for epoch in range(1, epochs + 1):
        # --- train -------------------------------------------------------
        model.train()
        running_loss = 0.0
        num_batches = 0

        for rgb, ela, fft_noise, meta, labels in train_loader:
            rgb = rgb.to(device)
            ela = ela.to(device)
            fft_noise = fft_noise.to(device)
            meta = meta.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            logits, _ = model(rgb, ela, fft_noise, meta)
            loss = criterion(logits, labels)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            running_loss += loss.item()
            num_batches += 1

        avg_train_loss = running_loss / max(num_batches, 1)

        # --- validate ----------------------------------------------------
        model.eval()
        val_loss = 0.0
        val_batches = 0

        with torch.no_grad():
            for rgb, ela, fft_noise, meta, labels in val_loader:
                rgb = rgb.to(device)
                ela = ela.to(device)
                fft_noise = fft_noise.to(device)
                meta = meta.to(device)
                labels = labels.to(device)

                logits, _ = model(rgb, ela, fft_noise, meta)
                loss = criterion(logits, labels)
                val_loss += loss.item()
                val_batches += 1

        avg_val_loss = val_loss / max(val_batches, 1)
        scheduler.step(avg_val_loss)

        current_lr = optimizer.param_groups[0]["lr"]
        print(
            f"Phase {phase} | Epoch {epoch}/{epochs} | "
            f"Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | "
            f"LR: {current_lr:.2e}"
        )

        # --- checkpoint best ---------------------------------------------
        if avg_val_loss < best_loss:
            best_loss = avg_val_loss
            torch.save(
                {
                    "phase": phase,
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "best_loss": best_loss,
                },
                best_path,
            )
            logger.info(
                "Phase %d — saved best checkpoint (epoch %d, val_loss=%.4f)",
                phase,
                epoch,
                best_loss,
            )

    return model


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="ForgeLens curriculum training (Phases 1→4).",
    )
    parser.add_argument(
        "--data_root",
        type=str,
        default="./data",
        help="Root directory containing train/ and val/ splits.",
    )
    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        default="./training/checkpoints",
        help="Directory for saving per-phase checkpoints.",
    )
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info("Using device: %s", device)

    checkpoint_dir = Path(args.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # Instantiate model with frozen backbones (default)
    model = ForgeLensModel(freeze_backbones=True, num_classes=3).to(device)

    for phase in range(1, 5):
        logger.info("=" * 60)
        logger.info("Starting Phase %d", phase)
        logger.info("=" * 60)

        # Load best checkpoint from the previous phase (if it exists)
        if phase > 1:
            prev_path = checkpoint_dir / f"phase{phase - 1}_best.pt"
            if prev_path.is_file():
                ckpt = torch.load(prev_path, map_location=device, weights_only=True)
                model.load_state_dict(ckpt["model_state_dict"])
                logger.info(
                    "Loaded phase %d checkpoint (val_loss=%.4f) as starting "
                    "point for phase %d.",
                    phase - 1,
                    ckpt["best_loss"],
                    phase,
                )
            else:
                logger.warning(
                    "Phase %d checkpoint not found at %s — continuing with "
                    "current weights.",
                    phase - 1,
                    prev_path,
                )

        model = train_one_phase(
            phase=phase,
            model=model,
            data_root=args.data_root,
            checkpoint_dir=args.checkpoint_dir,
        )

    logger.info("All 4 phases complete.")


if __name__ == "__main__":
    main()
