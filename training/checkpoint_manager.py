"""
training/checkpoint_manager.py
================================
Model checkpoint saving and management.

Keeps the N most recent checkpoints and always preserves the
best model (by validation reward). Old checkpoints are pruned
automatically to save disk space.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Dict, List, Optional

from utils.logger import get_logger

log = get_logger(__name__)


class CheckpointManager:
    """
    Manages model checkpoint files on disk.

    Parameters
    ----------
    checkpoint_dir : str | Path
        Directory to save checkpoints.
    save_freq : int
        Save a checkpoint every N training steps.
    keep_n_checkpoints : int
        Maximum number of rolling checkpoints to retain.
        Best model is always kept regardless of this limit.
    """

    METADATA_FILE = "checkpoint_metadata.json"

    def __init__(
        self,
        checkpoint_dir: str | Path,
        save_freq: int = 100_000,
        keep_n_checkpoints: int = 5,
    ) -> None:
        self.checkpoint_dir = Path(checkpoint_dir)
        self.save_freq = save_freq
        self.keep_n_checkpoints = keep_n_checkpoints
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        self._metadata: Dict = self._load_metadata()
        self._best_reward: float = self._metadata.get("best_reward", float("-inf"))

    # ── Public API ────────────────────────────────────────────

    def record_checkpoint(
        self,
        step: int,
        path: str,
        eval_reward: Optional[float] = None,
        extra_info: Optional[dict] = None,
    ) -> bool:
        """
        Record a new checkpoint and prune old ones.

        Parameters
        ----------
        step : int
            Training step at time of save.
        path : str
            Path where the model was saved.
        eval_reward : Optional[float]
            Validation mean reward at this checkpoint.
        extra_info : Optional[dict]
            Any extra metadata to record.

        Returns
        -------
        bool
            True if this is a new best model.
        """
        entry = {
            "step": step,
            "path": str(path),
            "eval_reward": eval_reward,
            "extra": extra_info or {},
        }

        self._metadata.setdefault("checkpoints", []).append(entry)
        is_best = False

        if eval_reward is not None and eval_reward > self._best_reward:
            self._best_reward = eval_reward
            self._metadata["best_reward"] = eval_reward
            self._metadata["best_checkpoint"] = str(path)
            is_best = True
            log.info("New best model", reward=round(eval_reward, 4), step=step)

        self._prune_old_checkpoints()
        self._save_metadata()
        return is_best

    def get_best_checkpoint(self) -> Optional[str]:
        """Return path to the best checkpoint, or None if none recorded."""
        return self._metadata.get("best_checkpoint")

    def get_latest_checkpoint(self) -> Optional[str]:
        """Return path to the most recently saved checkpoint."""
        checkpoints = self._metadata.get("checkpoints", [])
        if not checkpoints:
            return None
        return checkpoints[-1]["path"]

    def list_checkpoints(self) -> List[dict]:
        """Return all recorded checkpoint metadata entries."""
        return list(self._metadata.get("checkpoints", []))

    # ── Private ───────────────────────────────────────────────

    def _prune_old_checkpoints(self) -> None:
        """Remove oldest checkpoints beyond keep_n_checkpoints limit."""
        checkpoints = self._metadata.get("checkpoints", [])
        best_path = self._metadata.get("best_checkpoint")

        # Identify candidates for pruning (exclude best model)
        prunable = [c for c in checkpoints if c["path"] != best_path]

        while len(prunable) > self.keep_n_checkpoints:
            oldest = prunable.pop(0)
            old_path = Path(oldest["path"])

            # Remove .zip file (SB3 adds .zip extension)
            for suffix in ["", ".zip"]:
                candidate = Path(str(old_path) + suffix)
                if candidate.exists():
                    try:
                        candidate.unlink()
                        log.debug("Pruned checkpoint", path=str(candidate))
                    except OSError as e:
                        log.warning("Could not delete checkpoint", path=str(candidate), error=str(e))

            # Remove from metadata
            if oldest in self._metadata["checkpoints"]:
                self._metadata["checkpoints"].remove(oldest)

    def _load_metadata(self) -> dict:
        meta_path = self.checkpoint_dir / self.METADATA_FILE
        if meta_path.exists():
            try:
                with open(meta_path, "r") as f:
                    return json.load(f)
            except (json.JSONDecodeError, OSError):
                log.warning("Could not load checkpoint metadata, starting fresh.")
        return {"checkpoints": [], "best_reward": float("-inf")}

    def _save_metadata(self) -> None:
        meta_path = self.checkpoint_dir / self.METADATA_FILE
        with open(meta_path, "w") as f:
            json.dump(self._metadata, f, indent=2, default=str)