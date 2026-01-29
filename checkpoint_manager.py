#!/usr/bin/env python3
"""
╔═══════════════════════════════════════════════════════════════════════════════╗
║              IndexTTS2 Production-Grade Checkpoint Manager                    ║
║                                                                               ║
║  Provides atomic checkpoint saving, rollback support, and RNG state          ║
║  preservation for reliable training recovery and reproducibility.            ║
╚═══════════════════════════════════════════════════════════════════════════════╝
"""

import functools
import hashlib
import json
import logging
import os
import random
import shutil
import subprocess
import tempfile
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Protocol, Union

import numpy as np
import torch

from indextts.utils.logger import get_logger, logger_manager


# ═══════════════════════════════════════════════════════════════════════════════
# OBSERVABILITY SETUP
# ═══════════════════════════════════════════════════════════════════════════════

logger = get_logger()
# Initialize with defaults if not already done, but usually done by the main script.
if not logger.handlers:
    logger = logger_manager.setup(name="checkpoint_manager", log_file="checkpoint_manager.log")


def trace_execution(func):
    """Decorator to trace function execution with entry/exit logging.

    Args:
        func: Function to wrap.

    Returns:
        Wrapped function with execution tracing.
    """

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        logger.debug(f"ENTERING: {func.__name__} | Args count: {len(args)} | Kwargs: {list(kwargs.keys())}")
        start_time = time.perf_counter()
        try:
            result = func(*args, **kwargs)
            duration = (time.perf_counter() - start_time) * 1000
            logger.debug(f"EXITING: {func.__name__} | Duration: {duration:.2f}ms")
            return result
        except Exception as e:
            logger.exception(f"CRASH in {func.__name__}: {str(e)}")
            raise

    return wrapper


# ═══════════════════════════════════════════════════════════════════════════════
# CHECKPOINT METADATA
# ═══════════════════════════════════════════════════════════════════════════════


class CheckpointMetadata:
    """Manages checkpoint metadata and indexing.

    Attributes:
        metadata_path (Path): Path to the metadata JSON file.
        checkpoints (Dict): Dictionary of checkpoint entries.
    """

    def __init__(self, checkpoint_dir: Union[str, Path]) -> None:
        """Initialize metadata manager.

        Args:
            checkpoint_dir: Directory containing checkpoints.

        Raises:
            ValueError: If checkpoint_dir is empty.
        """
        if not checkpoint_dir:
            raise ValueError("Checkpoint directory cannot be empty.")

        self.checkpoint_dir = Path(checkpoint_dir)
        self.metadata_path = self.checkpoint_dir / "metadata.json"
        self.checkpoints: Dict[int, Dict[str, Any]] = {}
        self._load()

    @trace_execution
    def _load(self) -> None:
        """Load metadata from disk if exists."""
        if self.metadata_path.exists():
            with open(self.metadata_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                self.checkpoints = {int(k): v for k, v in data.get("checkpoints", {}).items()}
            logger.info(f"Loaded metadata with {len(self.checkpoints)} checkpoints")

    @trace_execution
    def _save(self) -> None:
        """Persist metadata to disk atomically."""
        temp_path = self.metadata_path.with_suffix(".tmp")
        with open(temp_path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "checkpoints": self.checkpoints,
                    "last_updated": datetime.now().isoformat(),
                },
                f,
                indent=2,
            )
        temp_path.replace(self.metadata_path)

    def add_checkpoint(
        self,
        step: int,
        path: Path,
        loss: float,
        stage: int,
        timestamp: str,
        git_hash: Optional[str] = None,
    ) -> None:
        """Register a new checkpoint.

        Args:
            step: Training step number.
            path: Path to checkpoint file.
            loss: Training loss at checkpoint.
            stage: Training stage (1, 2, or 3).
            timestamp: ISO format timestamp.
            git_hash: Optional git commit hash.
        """
        self.checkpoints[step] = {
            "path": str(path),
            "loss": loss,
            "stage": stage,
            "timestamp": timestamp,
            "git_hash": git_hash,
        }
        self._save()
        logger.debug(f"Added checkpoint at step {step}")

    def remove_checkpoint(self, step: int) -> None:
        """Remove a checkpoint entry.

        Args:
            step: Step number to remove.
        """
        if step in self.checkpoints:
            del self.checkpoints[step]
            self._save()
            logger.debug(f"Removed checkpoint at step {step}")

    def get_best_checkpoint(self) -> Optional[int]:
        """Get step of checkpoint with lowest loss.

        Returns:
            Step number of best checkpoint, or None if no checkpoints.
        """
        if not self.checkpoints:
            return None
        return min(self.checkpoints.keys(), key=lambda s: self.checkpoints[s]["loss"])

    def get_latest_checkpoint(self) -> Optional[int]:
        """Get step of most recent checkpoint.

        Returns:
            Step number of latest checkpoint, or None if no checkpoints.
        """
        if not self.checkpoints:
            return None
        return max(self.checkpoints.keys())

    def list_checkpoints(self) -> List[Dict[str, Any]]:
        """List all checkpoints sorted by step.

        Returns:
            List of checkpoint info dicts.
        """
        return [{"step": k, **v} for k, v in sorted(self.checkpoints.items())]


# ═══════════════════════════════════════════════════════════════════════════════
# CHECKPOINT MANAGER
# ═══════════════════════════════════════════════════════════════════════════════


def get_git_hash() -> Optional[str]:
    """Get current git commit hash for reproducibility.

    Returns:
        Short git hash string, or None if not in git repo.
    """
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except (subprocess.TimeoutExpired, FileNotFoundError):
        pass
    return None


def compute_state_hash(state_dict: Dict) -> str:
    """Compute a hash of model state for verification.

    Args:
        state_dict: Model state dictionary.

    Returns:
        SHA256 hash prefix for state verification.
    """
    hasher = hashlib.sha256()
    for key in sorted(state_dict.keys()):
        if isinstance(state_dict[key], torch.Tensor):
            hasher.update(key.encode())
            hasher.update(state_dict[key].cpu().numpy().tobytes()[:1024])
    return hasher.hexdigest()[:16]


class CheckpointManager:
    """Production-grade checkpoint manager with rollback support.

    Provides atomic checkpoint saving, automatic cleanup, and reliable
    state restoration for training recovery.

    Attributes:
        checkpoint_dir (Path): Directory for checkpoint storage.
        max_checkpoints (int): Maximum number of checkpoints to keep.
        save_interval (int): Steps between automatic saves.
        metadata (CheckpointMetadata): Checkpoint index manager.

    Example:
        >>> manager = CheckpointManager("checkpoints/training", max_checkpoints=5)
        >>> manager.save(model, optimizer, scheduler, step=1000, loss=0.5, stage=1)
        >>> manager.load_latest(model, optimizer, scheduler)
    """

    def __init__(
        self,
        checkpoint_dir: Union[str, Path],
        max_checkpoints: int = 5,
        save_interval: int = 500,
        debug: bool = False,
    ) -> None:
        """Initialize checkpoint manager.

        Args:
            checkpoint_dir: Directory for storing checkpoints.
            max_checkpoints: Maximum number of checkpoints to retain.
            save_interval: Steps between checkpoint saves.
            debug: Enable debug logging.

        Raises:
            ValueError: If max_checkpoints < 1.
        """
        global logger
        if debug:
            logger = logger_manager.setup(
                name="checkpoint_manager", 
                log_file="checkpoint_manager.log",
                level=logging.DEBUG
            )
        else:
            logger = get_logger()

        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.max_checkpoints = max_checkpoints
        self.save_interval = save_interval
        self.metadata = CheckpointMetadata(self.checkpoint_dir)

        logger.info(
            f"CheckpointManager initialized: dir={checkpoint_dir}, "
            f"max={max_checkpoints}, interval={save_interval}"
        )

    @trace_execution
    def save(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: Optional[Any],
        step: int,
        epoch: int,
        stage: int,
        loss: float,
        config: Optional[Dict] = None,
        loss_history: Optional[List[float]] = None,
    ) -> Path:
        """Save a checkpoint atomically.

        Args:
            model: Model to checkpoint.
            optimizer: Optimizer state.
            scheduler: Learning rate scheduler (optional).
            step: Current training step.
            epoch: Current epoch.
            stage: Training stage (1, 2, or 3).
            loss: Current training loss.
            config: Training configuration dict.
            loss_history: List of historical loss values.

        Returns:
            Path to saved checkpoint file.

        Raises:
            IOError: If checkpoint save fails.
        """
        checkpoint_name = f"step_{step:07d}.pt"
        checkpoint_path = self.checkpoint_dir / checkpoint_name
        temp_path = checkpoint_path.with_suffix(".tmp")

        timestamp = datetime.now().isoformat()
        git_hash = get_git_hash()

        checkpoint = {
            "step": step,
            "epoch": epoch,
            "stage": stage,
            "loss": loss,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict() if scheduler else None,
            "loss_history": loss_history or [],
            "config": config or {},
            "rng_states": {
                "torch": torch.get_rng_state(),
                "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
                "numpy": np.random.get_state(),
                "python": random.getstate(),
            },
            "timestamp": timestamp,
            "git_hash": git_hash,
            "state_hash": compute_state_hash(model.state_dict()),
        }

        # Atomic save: write to temp, then rename
        torch.save(checkpoint, temp_path)
        temp_path.replace(checkpoint_path)

        # Update metadata
        self.metadata.add_checkpoint(step, checkpoint_path, loss, stage, timestamp, git_hash)

        # Update symlinks
        self._update_symlinks(checkpoint_path)

        # Cleanup old checkpoints
        self._cleanup_old_checkpoints()

        logger.info(f"Checkpoint saved: step={step}, loss={loss:.4f}, path={checkpoint_path}")
        return checkpoint_path

    def _update_symlinks(self, checkpoint_path: Path) -> None:
        """Update latest and best checkpoint references.

        Uses file copy instead of symlinks for Windows compatibility.

        Args:
            checkpoint_path: Path to newest checkpoint.
        """
        # Update latest pointer (copy file for Windows compatibility)
        latest_link = self.checkpoint_dir / "latest.pt"
        if latest_link.exists():
            latest_link.unlink()
        shutil.copy2(checkpoint_path, latest_link)
        logger.debug(f"Updated latest.pt -> {checkpoint_path.name}")

        # Update best model pointer
        best_step = self.metadata.get_best_checkpoint()
        if best_step is not None:
            best_link = self.checkpoint_dir / "best_model.pt"
            best_path = Path(self.metadata.checkpoints[best_step]["path"])
            if best_link.exists():
                best_link.unlink()
            shutil.copy2(best_path, best_link)
            logger.debug(f"Updated best_model.pt -> {best_path.name}")

    def _cleanup_old_checkpoints(self) -> None:
        """Remove oldest checkpoints beyond max_checkpoints limit."""
        sorted_steps = sorted(self.metadata.checkpoints.keys())
        best_step = self.metadata.get_best_checkpoint()

        while len(sorted_steps) > self.max_checkpoints:
            oldest_step = sorted_steps[0]
            # Never delete the best checkpoint
            if oldest_step == best_step:
                sorted_steps.pop(0)
                continue

            checkpoint_info = self.metadata.checkpoints.get(oldest_step)
            if checkpoint_info:
                checkpoint_path = Path(checkpoint_info["path"])
                if checkpoint_path.exists():
                    checkpoint_path.unlink()
                    logger.debug(f"Deleted old checkpoint: {checkpoint_path}")
                self.metadata.remove_checkpoint(oldest_step)

            sorted_steps.pop(0)

    @trace_execution
    def load(
        self,
        step: int,
        model: torch.nn.Module,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler: Optional[Any] = None,
        restore_rng: bool = True,
    ) -> Dict[str, Any]:
        """Load a specific checkpoint by step number.

        Args:
            step: Step number to load.
            model: Model to restore state into.
            optimizer: Optimizer to restore (optional).
            scheduler: Scheduler to restore (optional).
            restore_rng: Whether to restore RNG states.

        Returns:
            Checkpoint metadata dict.

        Raises:
            FileNotFoundError: If checkpoint doesn't exist.
            RuntimeError: If state hash verification fails.
        """
        if step not in self.metadata.checkpoints:
            raise FileNotFoundError(f"No checkpoint found for step {step}")

        checkpoint_path = Path(self.metadata.checkpoints[step]["path"])
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint file missing: {checkpoint_path}")

        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

        # Verify state hash
        loaded_hash = compute_state_hash(checkpoint["model_state_dict"])
        if checkpoint.get("state_hash") and loaded_hash != checkpoint["state_hash"]:
            raise RuntimeError(
                f"State hash mismatch! Expected {checkpoint['state_hash']}, got {loaded_hash}"
            )

        # Restore model
        model.load_state_dict(checkpoint["model_state_dict"])

        # Restore optimizer
        if optimizer is not None and checkpoint.get("optimizer_state_dict"):
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        # Restore scheduler
        if scheduler is not None and checkpoint.get("scheduler_state_dict"):
            scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

        # Restore RNG states for reproducibility
        if restore_rng and checkpoint.get("rng_states"):
            rng_states = checkpoint["rng_states"]
            torch.set_rng_state(rng_states["torch"])
            if torch.cuda.is_available() and rng_states.get("cuda"):
                torch.cuda.set_rng_state_all(rng_states["cuda"])
            np.random.set_state(rng_states["numpy"])
            random.setstate(rng_states["python"])

        logger.info(f"Loaded checkpoint: step={step}, loss={checkpoint['loss']:.4f}")

        return {
            "step": checkpoint["step"],
            "epoch": checkpoint["epoch"],
            "stage": checkpoint["stage"],
            "loss": checkpoint["loss"],
            "loss_history": checkpoint.get("loss_history", []),
            "config": checkpoint.get("config", {}),
        }

    def load_latest(
        self,
        model: torch.nn.Module,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler: Optional[Any] = None,
        restore_rng: bool = True,
    ) -> Optional[Dict[str, Any]]:
        """Load the most recent checkpoint.

        Args:
            model: Model to restore.
            optimizer: Optimizer to restore.
            scheduler: Scheduler to restore.
            restore_rng: Whether to restore RNG states.

        Returns:
            Checkpoint metadata, or None if no checkpoints exist.
        """
        latest_step = self.metadata.get_latest_checkpoint()
        if latest_step is None:
            logger.warning("No checkpoints found to load")
            return None
        return self.load(latest_step, model, optimizer, scheduler, restore_rng)

    def load_best(
        self,
        model: torch.nn.Module,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler: Optional[Any] = None,
        restore_rng: bool = True,
    ) -> Optional[Dict[str, Any]]:
        """Load the best checkpoint (lowest loss).

        Args:
            model: Model to restore.
            optimizer: Optimizer to restore.
            scheduler: Scheduler to restore.
            restore_rng: Whether to restore RNG states.

        Returns:
            Checkpoint metadata, or None if no checkpoints exist.
        """
        best_step = self.metadata.get_best_checkpoint()
        if best_step is None:
            logger.warning("No checkpoints found to load")
            return None
        return self.load(best_step, model, optimizer, scheduler, restore_rng)

    def rollback(
        self,
        step: int,
        model: torch.nn.Module,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler: Optional[Any] = None,
    ) -> Dict[str, Any]:
        """Rollback to a specific checkpoint, removing all later checkpoints.

        Args:
            step: Step to rollback to.
            model: Model to restore.
            optimizer: Optimizer to restore.
            scheduler: Scheduler to restore.

        Returns:
            Checkpoint metadata after rollback.

        Raises:
            FileNotFoundError: If requested step doesn't exist.
        """
        logger.warning(f"Rolling back to step {step}")

        # Remove all checkpoints after this step
        steps_to_remove = [s for s in self.metadata.checkpoints.keys() if s > step]
        for remove_step in steps_to_remove:
            checkpoint_path = Path(self.metadata.checkpoints[remove_step]["path"])
            if checkpoint_path.exists():
                checkpoint_path.unlink()
            self.metadata.remove_checkpoint(remove_step)

        # Load the target checkpoint
        return self.load(step, model, optimizer, scheduler, restore_rng=True)

    def list_available_checkpoints(self) -> List[Dict[str, Any]]:
        """List all available checkpoints.

        Returns:
            List of checkpoint info dictionaries.
        """
        return self.metadata.list_checkpoints()

    def should_save(self, step: int) -> bool:
        """Check if checkpoint should be saved at this step.

        Args:
            step: Current training step.

        Returns:
            True if checkpoint should be saved.
        """
        return step > 0 and step % self.save_interval == 0


# ═══════════════════════════════════════════════════════════════════════════════
# CLI INTERFACE
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import argparse

    from rich.console import Console
    from rich.table import Table

    console = Console()

    parser = argparse.ArgumentParser(description="Checkpoint Manager CLI")
    parser.add_argument("--dir", type=str, default="checkpoints/training", help="Checkpoint directory")
    subparsers = parser.add_subparsers(dest="command")

    # List command
    list_parser = subparsers.add_parser("list", help="List all checkpoints")

    # Info command
    info_parser = subparsers.add_parser("info", help="Show checkpoint info")
    info_parser.add_argument("step", type=int, help="Step number")

    args = parser.parse_args()

    manager = CheckpointManager(args.dir)

    if args.command == "list":
        checkpoints = manager.list_available_checkpoints()
        if not checkpoints:
            console.print("[yellow]No checkpoints found[/yellow]")
        else:
            table = Table(title="Available Checkpoints")
            table.add_column("Step", style="cyan")
            table.add_column("Stage")
            table.add_column("Loss", style="green")
            table.add_column("Timestamp")
            table.add_column("Git Hash", style="dim")

            for cp in checkpoints:
                table.add_row(
                    str(cp["step"]),
                    str(cp.get("stage", "?")),
                    f"{cp['loss']:.4f}",
                    cp.get("timestamp", "?")[:19],
                    cp.get("git_hash", "N/A"),
                )

            console.print(table)

    elif args.command == "info":
        if args.step in manager.metadata.checkpoints:
            cp = manager.metadata.checkpoints[args.step]
            console.print(f"[bold]Checkpoint Step {args.step}[/bold]")
            console.print(f"  Path: {cp['path']}")
            console.print(f"  Loss: {cp['loss']:.4f}")
            console.print(f"  Stage: {cp.get('stage', '?')}")
            console.print(f"  Timestamp: {cp.get('timestamp', '?')}")
            console.print(f"  Git Hash: {cp.get('git_hash', 'N/A')}")
        else:
            console.print(f"[red]No checkpoint at step {args.step}[/red]")
