#!/usr/bin/env python3
"""
╔═══════════════════════════════════════════════════════════════════════════════╗
║          IndexTTS2 Duration Control Training Script                           ║
║                                                                               ║
║  Three-stage training paradigm for duration-controlled speech synthesis:     ║
║  Stage 1: Base training with duration embedding (30% dropout)                ║
║  Stage 2: Emotion control with GRL for speaker-emotion disentanglement       ║
║  Stage 3: Fine-tuning with frozen feature conditioners                       ║
╚═══════════════════════════════════════════════════════════════════════════════╝

Usage:
    uv run train_duration.py --stage 1 --data-dir ./data
    uv run train_duration.py --stage 2 --data-dir ./emotional_data --resume latest
    uv run train_duration.py --rollback 10000

Based on: IndexTTS2 Paper (Section: Proposed Method)
"""

# Suppress noisy warnings from dependencies
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning, module="transformers")
warnings.filterwarnings("ignore", message=".*gradient_checkpointing.*")
warnings.filterwarnings("ignore", message=".*weight_norm.*")

import argparse
import functools
import logging
import os
import random
import signal
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, Iterator, List, Optional, Tuple

from indextts.utils.logger import get_logger, logger_manager

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
import librosa
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, Dataset, IterableDataset
from omegaconf import OmegaConf

# Local imports
sys.path.insert(0, str(Path(__file__).parent))

# IndexTTS model imports
from indextts.gpt.model_v2 import UnifiedVoice
from indextts.utils.checkpoint import load_checkpoint
from indextts.utils.maskgct_utils import build_semantic_model, build_semantic_codec
from transformers import SeamlessM4TFeatureExtractor

try:
    import sentencepiece as spm
    HAS_SPM = True
except ImportError:
    HAS_SPM = False
    
try:
    from datasets import load_dataset
    HAS_DATASETS = True
except ImportError:
    HAS_DATASETS = False

from checkpoint_manager import CheckpointManager
from training_ui import TrainingUI, show_loading_animation, ModelLoadingUI, BufferingSpinner, StreamingProgressUI, setup_observability
from rich.console import Console

try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False


# ═══════════════════════════════════════════════════════════════════════════════
# OBSERVABILITY SETUP
# ═══════════════════════════════════════════════════════════════════════════════

logger = get_logger()


def trace_execution(func):
    """Decorator to trace function execution."""

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        logger.debug(f"ENTERING: {func.__name__}")
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
# GRADIENT REVERSAL LAYER
# ═══════════════════════════════════════════════════════════════════════════════


class GradientReversalFunction(torch.autograd.Function):
    """Gradient Reversal Layer for adversarial training.

    Passes input unchanged during forward pass, but reverses gradients
    during backward pass. Used for emotion-speaker disentanglement.
    """

    @staticmethod
    def forward(ctx, x: torch.Tensor, alpha: float) -> torch.Tensor:
        """Forward pass - identity function.

        Args:
            ctx: Autograd context.
            x: Input tensor.
            alpha: Gradient reversal scaling factor.

        Returns:
            Unchanged input tensor.
        """
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> Tuple[torch.Tensor, None]:
        """Backward pass - reverse gradients.

        Args:
            ctx: Autograd context.
            grad_output: Gradient from upstream.

        Returns:
            Tuple of reversed gradient and None for alpha.
        """
        return -ctx.alpha * grad_output, None


class GradientReversalLayer(nn.Module):
    """Module wrapper for GRL function.

    Attributes:
        alpha: Gradient scaling factor.

    Example:
        >>> grl = GradientReversalLayer(alpha=1.0)
        >>> output = grl(emotion_features)
    """

    def __init__(self, alpha: float = 1.0) -> None:
        """Initialize GRL.

        Args:
            alpha: Gradient reversal scaling factor.
        """
        super().__init__()
        self.alpha = alpha

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply gradient reversal.

        Args:
            x: Input tensor.

        Returns:
            Tensor with reversed gradients on backward pass.
        """
        return GradientReversalFunction.apply(x, self.alpha)


# ═══════════════════════════════════════════════════════════════════════════════
# TRAINING CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════════


@dataclass
class TrainingConfig:
    """Training configuration following paper hyperparameters.

    Attributes:
        stage: Training stage (1, 2, or 3).
        data_dir: Path to training data.
        checkpoint_dir: Path for saving checkpoints.
        model_dir: Path to pretrained model.
        batch_size: Training batch size.
        learning_rate: Initial learning rate.
        num_epochs: Total training epochs.
        max_steps: Maximum training steps (overrides epochs if set).
        duration_dropout: Probability of zeroing duration embedding (Stage 1).
        emotion_loss_alpha: Weight for emotion adversarial loss (Stage 2).
        save_interval: Steps between checkpoint saves.
        max_checkpoints: Maximum checkpoints to keep.
        early_stopping_patience: Steps without improvement before stopping (0=disabled).
        early_stopping_min_delta: Minimum improvement to count as progress.
        use_fp16: Use mixed precision training.
        debug: Enable debug logging.
    """

    stage: int = 1
    data_dir: str = "./data"
    checkpoint_dir: str = "./checkpoints/training"
    model_dir: str = "./checkpoints"
    batch_size: int = 4
    learning_rate: float = 2e-4
    num_epochs: int = 100
    max_steps: Optional[int] = None
    duration_dropout: float = 0.3
    emotion_loss_alpha: float = 1.0
    save_interval: int = 500
    max_checkpoints: int = 5
    early_stopping_patience: int = 0  # 0 = disabled
    early_stopping_min_delta: float = 1e-4
    use_fp16: bool = False
    debug: bool = False
    seed: int = 42
    num_workers: int = 4

    def validate(self) -> None:
        """Validate configuration values.

        Raises:
            ValueError: If configuration is invalid.
        """
        if self.stage not in (1, 2, 3):
            raise ValueError(f"Stage must be 1, 2, or 3, got {self.stage}")
        if self.batch_size < 1:
            raise ValueError(f"Batch size must be >= 1, got {self.batch_size}")
        if self.learning_rate <= 0:
            raise ValueError(f"Learning rate must be > 0, got {self.learning_rate}")
        if not (0 <= self.duration_dropout <= 1):
            raise ValueError(f"Duration dropout must be in [0, 1], got {self.duration_dropout}")
        if self.early_stopping_patience < 0:
            raise ValueError(f"Early stopping patience must be >= 0, got {self.early_stopping_patience}")


# ═══════════════════════════════════════════════════════════════════════════════
# FEATURE EXTRACTOR (For processing audio into model inputs)
# ═══════════════════════════════════════════════════════════════════════════════


class FeatureExtractor:
    """Extracts features from audio for training.
    
    Uses the same feature extraction pipeline as inference:
    - W2V-BERT-2.0 for conditioning embeddings
    - MaskGCT semantic codec for semantic tokens
    - BPE tokenizer for text tokens
    
    Attributes:
        device: Torch device for processing.
        semantic_model: W2V-BERT-2.0 model.
        semantic_codec: MaskGCT semantic codec.
        tokenizer: BPE tokenizer for text.
        
    Example:
        >>> extractor = FeatureExtractor(model_dir="./checkpoints", device="cuda")
        >>> features = extractor.extract_audio_features(audio_path)
    """
    
    def __init__(self, model_dir: str, device: str = "cuda") -> None:
        """Initialize feature extractor with models.
        
        Args:
            model_dir: Path to checkpoint directory.
            device: Device to load models on.
            
        Raises:
            FileNotFoundError: If model files not found.
        """
        self.device = device
        self.model_dir = Path(model_dir)
        
        logger.info("Initializing feature extractor...")
        
        # Load config
        cfg_path = self.model_dir / "config.yaml"
        if not cfg_path.exists():
            raise FileNotFoundError(f"Config not found: {cfg_path}")
        self.cfg = OmegaConf.load(cfg_path)
        
        # W2V-BERT-2.0 feature extractor
        self.extract_features = SeamlessM4TFeatureExtractor.from_pretrained(
            "facebook/w2v-bert-2.0"
        )
        
        # Semantic model
        w2v_stat_path = self.model_dir / self.cfg.w2v_stat
        self.semantic_model, self.semantic_mean, self.semantic_std = build_semantic_model(
            str(w2v_stat_path)
        )
        self.semantic_model = self.semantic_model.to(device).eval()
        self.semantic_mean = self.semantic_mean.to(device)
        self.semantic_std = self.semantic_std.to(device)
        
        # Semantic codec (MaskGCT)
        from huggingface_hub import hf_hub_download
        import safetensors.torch
        self.semantic_codec = build_semantic_codec(self.cfg.semantic_codec)
        semantic_code_ckpt = hf_hub_download(
            "amphion/MaskGCT", filename="semantic_codec/model.safetensors"
        )
        safetensors.torch.load_model(self.semantic_codec, semantic_code_ckpt)
        self.semantic_codec = self.semantic_codec.to(device).eval()
        
        # BPE tokenizer
        bpe_path = self.model_dir / self.cfg.dataset.bpe_model
        if HAS_SPM and bpe_path.exists():
            self.tokenizer = spm.SentencePieceProcessor()
            self.tokenizer.load(str(bpe_path))
        else:
            self.tokenizer = None
            logger.warning("BPE tokenizer not loaded - text features unavailable")
        
        logger.info("Feature extractor initialized")
    
    @torch.no_grad()
    def extract_audio_features(
        self, 
        audio_path: str,
        max_length_sec: float = 15.0,
    ) -> Dict[str, torch.Tensor]:
        """Extract features from audio file.
        
        Args:
            audio_path: Path to audio file.
            max_length_sec: Maximum audio length in seconds.
            
        Returns:
            Dictionary with 'conditioning_emb' and 'semantic_tokens'.
            
        Raises:
            FileNotFoundError: If audio file not found.
        """
        # Load audio
        audio, sr = librosa.load(audio_path, sr=None)
        audio = torch.from_numpy(audio).float()
        
        # Truncate if too long
        max_samples = int(max_length_sec * sr)
        if audio.shape[0] > max_samples:
            audio = audio[:max_samples]
        
        # Resample for different components
        audio_16k = torchaudio.transforms.Resample(sr, 16000)(audio.unsqueeze(0))
        
        # Extract W2V-BERT features
        inputs = self.extract_features(
            audio_16k.squeeze(0).numpy(), 
            sampling_rate=16000, 
            return_tensors="pt"
        )
        input_features = inputs["input_features"].to(self.device)
        attention_mask = inputs["attention_mask"].to(self.device)
        
        # Get conditioning embedding
        with torch.no_grad():
            outputs = self.semantic_model(
                input_features=input_features,
                attention_mask=attention_mask,
            )
            conditioning_emb = outputs.last_hidden_state
            # Normalize
            conditioning_emb = (conditioning_emb - self.semantic_mean) / self.semantic_std
            
            # Get semantic tokens
            _, semantic_tokens = self.semantic_codec.quantize(conditioning_emb)
            semantic_tokens = semantic_tokens.squeeze(0)  # Remove batch dim
        
        return {
            "conditioning_emb": conditioning_emb.squeeze(0),  # (T, 1024)
            "semantic_tokens": semantic_tokens.squeeze(0),    # (T,)
        }
    
    def tokenize_text(self, text: str) -> torch.Tensor:
        """Tokenize text using BPE tokenizer.
        
        Args:
            text: Text to tokenize.
            
        Returns:
            Tensor of token IDs.
        """
        if self.tokenizer is None:
            # Fallback: return random tokens for testing
            return torch.randint(0, 8192, (len(text) // 2,))
        
        tokens = self.tokenizer.encode(text)
        return torch.tensor(tokens, dtype=torch.long)


# ═══════════════════════════════════════════════════════════════════════════════
# DATASET IMPLEMENTATIONS (Emilia, ESD, Local)
# ═══════════════════════════════════════════════════════════════════════════════


class LocalDurationDataset(Dataset):
    """Dataset for local audio files organized by speaker.
    
    Expected directory structure:
        data_dir/
        ├── speaker_001/
        │   ├── utterance_001.wav
        │   ├── utterance_001.txt  (transcription)
        │   ├── utterance_002.wav
        │   └── utterance_002.txt
        └── speaker_002/
            └── ...
    
    Attributes:
        data_dir: Path to data directory.
        samples: List of sample metadata.
        feature_extractor: Shared feature extractor.
    """
    
    def __init__(
        self,
        data_dir: str,
        feature_extractor: Optional[FeatureExtractor] = None,
        use_emotional: bool = False,
        max_samples: Optional[int] = None,
    ) -> None:
        """Initialize local dataset.
        
        Args:
            data_dir: Path to training data.
            feature_extractor: Feature extractor instance.
            use_emotional: Include emotion labels (for Stage 2).
            max_samples: Maximum samples to load.
        """
        self.data_dir = Path(data_dir)
        self.feature_extractor = feature_extractor
        self.use_emotional = use_emotional
        self.samples: List[Dict[str, Any]] = []
        self._load_samples(max_samples)
    
    def _load_samples(self, max_samples: Optional[int]) -> None:
        """Scan data directory for audio files."""
        logger.info(f"Scanning local dataset: {self.data_dir}")
        
        if not self.data_dir.exists():
            logger.warning(f"Data directory not found: {self.data_dir}")
            return
        
        # Scan speaker directories
        for speaker_dir in sorted(self.data_dir.iterdir()):
            if not speaker_dir.is_dir():
                continue
            
            speaker_id = speaker_dir.name
            audio_files = list(speaker_dir.glob("*.wav")) + list(speaker_dir.glob("*.mp3"))
            
            for audio_file in audio_files:
                # Look for transcription
                txt_file = audio_file.with_suffix(".txt")
                if txt_file.exists():
                    text = txt_file.read_text(encoding="utf-8").strip()
                else:
                    text = ""  # Will need to handle missing transcriptions
                
                self.samples.append({
                    "id": audio_file.stem,
                    "audio_path": str(audio_file),
                    "text": text,
                    "speaker_id": speaker_id,
                })
                
                if max_samples and len(self.samples) >= max_samples:
                    break
            
            if max_samples and len(self.samples) >= max_samples:
                break
        
        logger.info(f"Found {len(self.samples)} local samples")
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        sample = self.samples[idx]
        
        if self.feature_extractor is not None:
            # Extract real features
            features = self.feature_extractor.extract_audio_features(sample["audio_path"])
            text_tokens = self.feature_extractor.tokenize_text(sample["text"])
        else:
            # Fallback to dummy data for testing
            seq_len = random.randint(50, 200)
            features = {
                "conditioning_emb": torch.randn(seq_len, 1024),
                "semantic_tokens": torch.randint(0, 8192, (seq_len,)),
            }
            text_tokens = torch.randint(0, 8192, (random.randint(10, 50),))
        
        return {
            "text_tokens": text_tokens,
            "semantic_tokens": features["semantic_tokens"],
            "conditioning_emb": features["conditioning_emb"],
            "speaker_id": sample["speaker_id"],
        }


class StreamingDataTimeout(Exception):
    """Raised when streaming data fetch times out."""
    pass


class EmiliaDataset(IterableDataset):
    """Streaming dataset for Emilia corpus from HuggingFace.
    
    Emilia dataset contains multi-lingual speech data for TTS training.
    Paper used 55K hours (30K Chinese + 25K English).
    
    Attributes:
        dataset: HuggingFace dataset in streaming mode.
        feature_extractor: Feature extractor instance.
        languages: List of languages to include.
        fetch_timeout: Timeout in seconds for fetching each sample.
        first_sample_timeout: Longer timeout for initial connection.
    """
    
    # Default timeouts (seconds)
    DEFAULT_FETCH_TIMEOUT = 60  # Per-sample timeout
    FIRST_SAMPLE_TIMEOUT = 300  # 5 minutes for initial connection
    
    def __init__(
        self,
        feature_extractor: Optional[FeatureExtractor] = None,
        languages: Optional[List[str]] = None,
        split: str = "train",
        max_samples: Optional[int] = None,
        fetch_timeout: int = 60,
        num_shards: int = 50,
        language: str = "EN",
        on_sample_callback: Optional[Callable[[int], None]] = None,
    ) -> None:
        """Initialize Emilia streaming dataset.
        
        Args:
            feature_extractor: Feature extractor instance.
            languages: Languages to filter (default: zh, en).
            split: Dataset split.
            max_samples: Maximum samples to yield.
            fetch_timeout: Timeout in seconds for fetching samples.
            num_shards: Number of shards to use (default: 50).
            language: Language subset to use (default: EN).
            on_sample_callback: Callback called with sample count when sample received.
        """
        logger.debug(f"[EmiliaDataset.__init__] START - language={language}, num_shards={num_shards}")
        logger.debug(f"[EmiliaDataset.__init__] fetch_timeout={fetch_timeout}s, max_samples={max_samples}")
        
        if not HAS_DATASETS:
            logger.error("[EmiliaDataset.__init__] datasets library not installed!")
            raise ImportError("datasets library required: pip install datasets")
        
        self.feature_extractor = feature_extractor
        self.languages = languages or [language.lower()]
        self.max_samples = max_samples
        self.fetch_timeout = fetch_timeout
        self.on_sample_callback = on_sample_callback
        self._is_first_sample = True
        self._samples_fetched = 0
        self._init_timestamp = time.time()
        
        # Generate shard patterns for limited download
        shard_patterns = [f"Emilia/{language}/{language}-B0000{i:02d}.tar" for i in range(num_shards)]
        
        logger.info(f"Loading Emilia dataset: {num_shards} shards of {language}")
        logger.info(f"Shard range: {shard_patterns[0]} to {shard_patterns[-1]}")
        logger.info(f"Fetch timeout: {fetch_timeout}s per sample, {self.FIRST_SAMPLE_TIMEOUT}s for first sample")
        logger.debug(f"[EmiliaDataset.__init__] All shard patterns: {shard_patterns}")
        
        # Load Emilia in streaming mode with specific shards
        # CRITICAL: Use .decode(False) to globally disable torchcodec audio decoding
        # This returns raw bytes/paths which we decode manually with soundfile
        logger.debug("[EmiliaDataset.__init__] Calling load_dataset() with streaming=True...")
        load_start = time.time()
        try:
            dataset = load_dataset(
                "amphion/Emilia-Dataset",
                split=split,
                streaming=True,
                data_files=shard_patterns,
            )
            load_duration = time.time() - load_start
            logger.debug(f"[EmiliaDataset.__init__] load_dataset() completed in {load_duration:.2f}s")
        except Exception as e:
            logger.exception(f"[EmiliaDataset.__init__] load_dataset() FAILED: {e}")
            raise
        
        # Disable all automatic feature decoding (bypasses torchcodec completely)
        logger.debug("[EmiliaDataset.__init__] Applying .decode(False) to disable auto-decoding...")
        self.dataset = dataset.decode(False)
        
        init_total = time.time() - self._init_timestamp
        logger.debug(f"[EmiliaDataset.__init__] COMPLETE - total init time: {init_total:.2f}s")
    
    def _iter_with_timeout(self):
        """Iterate over dataset with timeout detection.
        
        Yields:
            Samples from the underlying dataset.
            
        Raises:
            StreamingDataTimeout: If sample fetch exceeds timeout.
        """
        import threading
        import queue
        
        logger.debug("[_iter_with_timeout] START - initializing thread infrastructure")
        iter_start_time = time.time()
        
        sample_queue = queue.Queue(maxsize=1)
        error_queue = queue.Queue(maxsize=1)
        stop_event = threading.Event()
        fetch_started_event = threading.Event()
        
        # Thread-safe counters for debugging
        thread_stats = {
            "samples_put": 0,
            "thread_started": False,
            "thread_ended": False,
            "thread_error": None,
            "last_sample_time": None,
        }
        
        def fetch_samples():
            """Background thread to fetch samples."""
            thread_stats["thread_started"] = True
            logger.debug("[fetch_samples] THREAD STARTED - beginning iteration over self.dataset")
            fetch_started_event.set()
            
            try:
                sample_count = 0
                for sample in self.dataset:
                    if stop_event.is_set():
                        logger.debug(f"[fetch_samples] STOP EVENT received after {sample_count} samples")
                        break
                    
                    sample_count += 1
                    thread_stats["samples_put"] = sample_count
                    thread_stats["last_sample_time"] = time.time()
                    
                    # Debug breadcrumb for first few samples and periodic updates
                    if sample_count <= 5 or sample_count % 100 == 0:
                        sample_keys = list(sample.keys()) if isinstance(sample, dict) else "<non-dict>"
                        logger.debug(f"[fetch_samples] Sample #{sample_count} fetched, keys={sample_keys}")
                    
                    sample_queue.put(sample)
                    logger.debug(f"[fetch_samples] Sample #{sample_count} PUT to queue")
                    
                logger.debug(f"[fetch_samples] THREAD ITERATION COMPLETE - {sample_count} samples total")
                
            except Exception as e:
                thread_stats["thread_error"] = str(e)
                logger.exception(f"[fetch_samples] THREAD EXCEPTION: {e}")
                error_queue.put(e)
            finally:
                thread_stats["thread_ended"] = True
                logger.debug(f"[fetch_samples] THREAD EXITING - stats: {thread_stats}")
        
        fetch_thread = threading.Thread(target=fetch_samples, daemon=True, name="EmiliaFetchThread")
        logger.debug(f"[_iter_with_timeout] Starting fetch thread (daemon=True)")
        fetch_thread.start()
        
        # Wait for thread to actually start
        thread_wait_start = time.time()
        if not fetch_started_event.wait(timeout=10.0):
            logger.warning("[_iter_with_timeout] Fetch thread did not signal start within 10s!")
        else:
            logger.debug(f"[_iter_with_timeout] Fetch thread started in {time.time() - thread_wait_start:.2f}s")
        
        samples_yielded = 0
        try:
            while True:
                # Use longer timeout for first sample (initial connection)
                timeout = self.FIRST_SAMPLE_TIMEOUT if self._is_first_sample else self.fetch_timeout
                
                if self._is_first_sample:
                    logger.debug(f"[_iter_with_timeout] Waiting for FIRST sample (timeout={timeout}s)...")
                    logger.debug(f"[_iter_with_timeout] Thread alive: {fetch_thread.is_alive()}, stats: {thread_stats}")
                elif samples_yielded % 50 == 0:
                    logger.debug(f"[_iter_with_timeout] Yielded {samples_yielded} samples so far, thread stats: {thread_stats}")
                
                wait_start = time.time()
                try:
                    sample = sample_queue.get(timeout=timeout)
                    wait_duration = time.time() - wait_start
                    
                    if self._is_first_sample:
                        total_wait = time.time() - iter_start_time
                        logger.info(f"[_iter_with_timeout] [OK] FIRST SAMPLE RECEIVED! Wait: {wait_duration:.2f}s, total: {total_wait:.2f}s")
                    
                    self._is_first_sample = False
                    self._samples_fetched += 1
                    samples_yielded += 1
                    yield sample
                    
                except queue.Empty:
                    wait_duration = time.time() - wait_start
                    logger.warning(f"[_iter_with_timeout] QUEUE TIMEOUT after {wait_duration:.2f}s (expected timeout: {timeout}s)")
                    logger.warning(f"[_iter_with_timeout] Thread alive: {fetch_thread.is_alive()}, stats: {thread_stats}")
                    
                    # Check if there was an error
                    if not error_queue.empty():
                        err = error_queue.get()
                        logger.error(f"[_iter_with_timeout] Error from fetch thread: {err}")
                        raise err
                    
                    # Timeout occurred
                    if self._is_first_sample:
                        logger.error(f"[_iter_with_timeout] FIRST SAMPLE TIMEOUT - network issue likely")
                        raise StreamingDataTimeout(
                            f"Timed out waiting for first sample after {timeout}s. "
                            f"Check network connection to HuggingFace Hub."
                        )
                    else:
                        logger.error(f"[_iter_with_timeout] SAMPLE TIMEOUT after {self._samples_fetched} samples")
                        raise StreamingDataTimeout(
                            f"Timed out fetching sample after {timeout}s "
                            f"(fetched {self._samples_fetched} samples so far)"
                        )
        finally:
            logger.debug(f"[_iter_with_timeout] CLEANUP - setting stop event, yielded {samples_yielded} samples")
            stop_event.set()
    
    def __iter__(self) -> Iterator[Dict[str, Any]]:
        import io
        import soundfile as sf
        
        count = 0
        skipped_language = 0
        skipped_no_audio = 0
        skipped_errors = 0
        iter_start_time = time.time()
        
        logger.info("Starting to stream data from Emilia dataset...")
        logger.info("(First sample may take a few minutes to arrive due to network latency)")
        logger.debug(f"[__iter__] Language filter: {self.languages}, max_samples: {self.max_samples}")
        
        for sample in self._iter_with_timeout():
            raw_sample_idx = count + skipped_language + skipped_no_audio + skipped_errors
            
            # Call progress callback on raw sample receipt (before processing)
            # This shows samples as they arrive from HuggingFace
            if self.on_sample_callback:
                self.on_sample_callback(raw_sample_idx + 1)
            
            # Log progress periodically with detailed debug
            if count == 0 and raw_sample_idx == 0:
                logger.info("[OK] First sample received from streaming dataset!")
                logger.debug(f"[__iter__] First raw sample keys: {list(sample.keys()) if isinstance(sample, dict) else 'N/A'}")
            elif raw_sample_idx % 100 == 0:
                elapsed = time.time() - iter_start_time
                logger.debug(f"[__iter__] Progress: {count} yielded, {raw_sample_idx} raw samples, {elapsed:.1f}s elapsed")
                logger.debug(f"[__iter__] Skipped: language={skipped_language}, no_audio={skipped_no_audio}, errors={skipped_errors}")
            
            # Filter by language if specified
            # Emilia samples have 'language' field (e.g. 'zh', 'en')
            sample_lang = sample.get("language")
            if self.languages and sample_lang not in self.languages:
                skipped_language += 1
                if skipped_language <= 5 or skipped_language % 100 == 0:
                    logger.debug(f"[__iter__] Skipping sample with language='{sample_lang}' (filter: {self.languages})")
                continue
            
            try:
                # With decode=False, audio is {'path': ..., 'bytes': ...}
                audio_data = sample.get("audio", {})
                audio_bytes = audio_data.get("bytes")
                
                if audio_bytes is None:
                    skipped_no_audio += 1
                    if skipped_no_audio <= 5:
                        logger.debug(f"[__iter__] Skipping sample with no audio bytes (audio_data keys: {list(audio_data.keys()) if isinstance(audio_data, dict) else 'N/A'})")
                    continue
                
                # Debug: log audio byte size for first few samples
                if count < 5:
                    logger.debug(f"[__iter__] Sample #{count}: audio_bytes size = {len(audio_bytes)} bytes")
                
                # Decode audio bytes using soundfile (no FFmpeg needed)
                decode_start = time.time()
                audio, sr = sf.read(io.BytesIO(audio_bytes))
                decode_duration = time.time() - decode_start
                
                if count < 5:
                    logger.debug(f"[__iter__] Sample #{count}: audio shape={audio.shape}, sr={sr}, decode_time={decode_duration:.3f}s")
                
                # Handle stereo -> mono
                if len(audio.shape) > 1:
                    audio = audio.mean(axis=1)

                text = sample.get("text", "")
                speaker_id = sample.get("speaker", "unknown")
                
                if count < 5:
                    logger.debug(f"[__iter__] Sample #{count}: text_len={len(text)}, speaker={speaker_id}")
                
                # Convert to tensor
                audio_tensor = torch.from_numpy(audio).float()
                
                if self.feature_extractor is not None:
                    # Save temp file and extract features
                    import tempfile
                    feature_start = time.time()
                    with tempfile.NamedTemporaryFile(suffix=".wav", delete=True) as f:
                        torchaudio.save(f.name, audio_tensor.unsqueeze(0), sr)
                        features = self.feature_extractor.extract_audio_features(f.name)
                        text_tokens = self.feature_extractor.tokenize_text(text)
                    feature_duration = time.time() - feature_start
                    if count < 5:
                        logger.debug(f"[__iter__] Sample #{count}: feature extraction took {feature_duration:.3f}s")
                else:
                    # Dummy features for testing
                    seq_len = max(1, len(audio) // 320)  # Approximate tokens
                    features = {
                        "conditioning_emb": torch.randn(seq_len, 1024),
                        "semantic_tokens": torch.randint(0, 8192, (seq_len,)),
                    }
                    text_tokens = torch.randint(0, 8192, (max(1, len(text) // 4),))
                    if count < 5:
                        logger.debug(f"[__iter__] Sample #{count}: using dummy features, seq_len={seq_len}")
                
                yield {
                    "text_tokens": text_tokens,
                    "semantic_tokens": features["semantic_tokens"],
                    "conditioning_emb": features["conditioning_emb"],
                    "speaker_id": speaker_id,
                }
                
                count += 1
                
                if self.max_samples and count >= self.max_samples:
                    logger.info(f"Reached max_samples limit: {self.max_samples}")
                    break
                    
            except Exception as e:
                # Skip errors in streaming data
                skipped_errors += 1
                if skipped_errors <= 10:
                    logger.warning(f"[__iter__] Skipping sample due to error #{skipped_errors}: {e}")
                elif skipped_errors % 100 == 0:
                    logger.warning(f"[__iter__] Skipped {skipped_errors} samples due to errors")
                continue
        
        total_elapsed = time.time() - iter_start_time
        logger.info(f"Finished streaming {count} samples from Emilia dataset in {total_elapsed:.1f}s")
        logger.info(f"Skip summary: language={skipped_language}, no_audio={skipped_no_audio}, errors={skipped_errors}")


class ESDDataset(Dataset):
    """Dataset for Emotional Speech Database (ESD).
    
    ESD contains ~29 hours of emotional speech from 10 English and 10 Chinese speakers.
    Used for Stage 2 training (emotion control with GRL).
    
    Paper: "135 hours of emotional data came from 361 speakers, of which 29 hours 
           came from the ESD dataset"
    """
    
    def __init__(
        self,
        data_dir: str,
        feature_extractor: Optional[FeatureExtractor] = None,
        max_samples: Optional[int] = None,
    ) -> None:
        """Initialize ESD dataset.
        
        Args:
            data_dir: Path to ESD dataset.
            feature_extractor: Feature extractor instance.
            max_samples: Maximum samples to load.
        """
        self.data_dir = Path(data_dir)
        self.feature_extractor = feature_extractor
        self.samples: List[Dict[str, Any]] = []
        self.emotion_labels = ["Angry", "Happy", "Neutral", "Sad", "Surprise"]
        self._load_samples(max_samples)
    
    def _load_samples(self, max_samples: Optional[int]) -> None:
        """Load ESD samples organized by speaker/emotion."""
        logger.info(f"Loading ESD dataset from: {self.data_dir}")
        
        if not self.data_dir.exists():
            logger.warning(f"ESD directory not found: {self.data_dir}")
            self._download_esd()
        
        if not any(self.data_dir.iterdir()): 
             self._download_esd()
             
        # ESD structure: {speaker_id}/{emotion}/{utterance}.wav
        for speaker_dir in sorted(self.data_dir.iterdir()):
            if not speaker_dir.is_dir():
                continue
            
            speaker_id = speaker_dir.name
            for emotion_dir in speaker_dir.iterdir():
                if not emotion_dir.is_dir():
                    continue
                
                emotion = emotion_dir.name
                if emotion not in self.emotion_labels:
                    continue
                
                for audio_file in emotion_dir.glob("*.wav"):
                    txt_file = audio_file.with_suffix(".txt")
                    text = txt_file.read_text().strip() if txt_file.exists() else ""
                    
                    self.samples.append({
                        "id": audio_file.stem,
                        "audio_path": str(audio_file),
                        "text": text,
                        "speaker_id": speaker_id,
                        "emotion": emotion,
                    })
                    
                    if max_samples and len(self.samples) >= max_samples:
                        return
        
        logger.info(f"Loaded {len(self.samples)} ESD samples")

    def _download_esd(self) -> None:
        """Download ESD dataset from HuggingFace."""
        logger.info("Downloading ESD dataset from HuggingFace (sonchuate/ESD_dataset)...")
        if not HAS_DATASETS:
            raise ImportError("datasets library required for download: pip install datasets")
            
        try:
            from huggingface_hub import snapshot_download
            snapshot_download(
                repo_id="sonchuate/ESD_dataset",
                local_dir=self.data_dir,
                repo_type="dataset",
                allow_patterns=["*.wav", "*.txt"] 
            )
            logger.info("ESD dataset downloaded successfully.")
        except Exception as e:
            logger.error(f"Failed to download ESD dataset: {e}")
            logger.info("Please manually download to {self.data_dir}")
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        sample = self.samples[idx]
        
        if self.feature_extractor is not None:
            features = self.feature_extractor.extract_audio_features(sample["audio_path"])
            text_tokens = self.feature_extractor.tokenize_text(sample["text"])
        else:
            seq_len = random.randint(50, 200)
            features = {
                "conditioning_emb": torch.randn(seq_len, 1024),
                "semantic_tokens": torch.randint(0, 8192, (seq_len,)),
            }
            text_tokens = torch.randint(0, 8192, (random.randint(10, 50),))
        
        return {
            "text_tokens": text_tokens,
            "semantic_tokens": features["semantic_tokens"],
            "conditioning_emb": features["conditioning_emb"],
            "speaker_id": sample["speaker_id"],
            "emotion": sample.get("emotion", "neutral"),
        }


def collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
    """Collate batch samples with padding.

    Args:
        batch: List of sample dictionaries.

    Returns:
        Batched and padded tensors.
    """
    # Pad sequences
    text_tokens = torch.nn.utils.rnn.pad_sequence(
        [s["text_tokens"] for s in batch], batch_first=True, padding_value=0
    )
    semantic_tokens = torch.nn.utils.rnn.pad_sequence(
        [s["semantic_tokens"] for s in batch], batch_first=True, padding_value=0
    )
    conditioning_emb = torch.nn.utils.rnn.pad_sequence(
        [s["conditioning_emb"] for s in batch], batch_first=True, padding_value=0
    )
    
    text_lengths = torch.tensor([len(s["text_tokens"]) for s in batch])
    semantic_lengths = torch.tensor([len(s["semantic_tokens"]) for s in batch])
    cond_lengths = torch.tensor([s["conditioning_emb"].shape[0] for s in batch])

    return {
        "text_tokens": text_tokens,
        "semantic_tokens": semantic_tokens,
        "conditioning_emb": conditioning_emb,
        "text_lengths": text_lengths,
        "semantic_lengths": semantic_lengths,
        "cond_lengths": cond_lengths,
    }



# ═══════════════════════════════════════════════════════════════════════════════
# EARLY STOPPING
# ═══════════════════════════════════════════════════════════════════════════════


class EarlyStopping:
    """Early stopping to terminate training when loss stops improving.

    Monitors training loss and triggers early termination if no improvement
    is seen for a specified number of steps (patience).

    Attributes:
        patience: Number of steps to wait for improvement.
        min_delta: Minimum change to qualify as improvement.
        best_loss: Best loss value seen so far.
        steps_without_improvement: Counter for steps since last improvement.
        should_stop: Flag indicating if training should stop.

    Example:
        >>> early_stopping = EarlyStopping(patience=1000, min_delta=1e-4)
        >>> if early_stopping(current_loss):
        ...     print("Stopping early!")
    """

    def __init__(self, patience: int = 1000, min_delta: float = 1e-4) -> None:
        """Initialize early stopping.

        Args:
            patience: Steps to wait without improvement before stopping.
            min_delta: Minimum improvement threshold.

        Raises:
            ValueError: If patience < 0.
        """
        if patience < 0:
            raise ValueError(f"Patience must be >= 0, got {patience}")

        self.patience = patience
        self.min_delta = min_delta
        self.best_loss: Optional[float] = None
        self.steps_without_improvement = 0
        self.should_stop = False

        logger.info(f"EarlyStopping initialized: patience={patience}, min_delta={min_delta}")

    def __call__(self, loss: float) -> bool:
        """Check if training should stop.

        Args:
            loss: Current loss value.

        Returns:
            True if training should stop, False otherwise.
        """
        if self.patience == 0:
            return False  # Disabled

        if self.best_loss is None:
            self.best_loss = loss
            return False

        if loss < self.best_loss - self.min_delta:
            # Improvement found
            self.best_loss = loss
            self.steps_without_improvement = 0
            logger.debug(f"EarlyStopping: New best loss {loss:.4f}")
        else:
            # No improvement
            self.steps_without_improvement += 1
            if self.steps_without_improvement >= self.patience:
                self.should_stop = True
                logger.warning(
                    f"EarlyStopping triggered: {self.patience} steps without improvement. "
                    f"Best loss: {self.best_loss:.4f}"
                )
                return True

        return False

    def reset(self) -> None:
        """Reset early stopping state."""
        self.best_loss = None
        self.steps_without_improvement = 0
        self.should_stop = False


# ═══════════════════════════════════════════════════════════════════════════════
# TRAINER CLASS
# ═══════════════════════════════════════════════════════════════════════════════


class DurationTrainer:
    """Main trainer for duration control.

    Implements three-stage training paradigm from IndexTTS2 paper.

    Attributes:
        config: Training configuration.
        model: T2S model.
        optimizer: AdamW optimizer.
        scheduler: Learning rate scheduler.
        checkpoint_manager: Checkpoint saving/loading.
        ui: Training UI.

    Example:
        >>> config = TrainingConfig(stage=1, data_dir="./data")
        >>> trainer = DurationTrainer(config)
        >>> trainer.train()
    """

    def __init__(self, config: TrainingConfig) -> None:
        """Initialize trainer.

        Args:
            config: Training configuration.
        """
        global logger
        if config.debug:
            logger = setup_observability(debug_mode=True)

        self.config = config
        config.validate()

        # Set random seeds
        self._set_seeds(config.seed)

        # Use animated UI for initialization with stdout suppression
        with ModelLoadingUI() as loading_ui:
            # Load configuration
            loading_ui.start_component("config")
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            logger.info(f"Using device: {self.device}")
            loading_ui.complete_component(success=True)
            
            # Load GPT model
            loading_ui.start_component("gpt")
            self.model = self._load_model()
            loading_ui.complete_component(success=True)
            
            # Skip semantic/codec/tokenizer (loaded in train() for feature extractor)
            # Mark them as pending for now
            
            # Setup optimizer
            loading_ui.start_component("optimizer")
            self.optimizer = AdamW(self.model.parameters(), lr=config.learning_rate)
            self.scheduler = CosineAnnealingLR(self.optimizer, T_max=config.num_epochs)
            
            self.checkpoint_manager = CheckpointManager(
                checkpoint_dir=config.checkpoint_dir,
                max_checkpoints=config.max_checkpoints,
                save_interval=config.save_interval,
                debug=config.debug,
            )
            loading_ui.complete_component(success=True)

        # Initialize remaining components (outside loading UI)
        self.ui = TrainingUI(debug=config.debug)

        self.grl = GradientReversalLayer(alpha=config.emotion_loss_alpha)

        # Early stopping
        self.early_stopping = EarlyStopping(
            patience=config.early_stopping_patience,
            min_delta=config.early_stopping_min_delta,
        )

        self.step = 0
        self.epoch = 0
        self.loss_history: List[float] = []
        self.is_paused = False
        self.should_stop = False

        self._setup_signal_handlers()

    def _set_seeds(self, seed: int) -> None:
        """Set random seeds for reproducibility.

        Args:
            seed: Random seed value.
        """
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    @trace_execution
    def _load_model(self) -> nn.Module:
        """Load the UnifiedVoice T2S model.

        Returns:
            Loaded model on appropriate device.
        """
        logger.info(f"Loading model from {self.config.model_dir}")

        # Load configuration
        cfg_path = Path(self.config.model_dir) / "config.yaml"
        if not cfg_path.exists():
            # Fallback to dummy model if config not found (only for testing)
            logger.warning("Config not found, falling back to dummy model")
            return self._create_dummy_model()
            
        cfg = OmegaConf.load(cfg_path)

        # Create model with config parameters
        # Disable acceleration/compilation for training loop compatibility
        model = UnifiedVoice(
            **cfg.gpt,
            use_accel=False,
        )
        
        # Enable gradient checkpointing the modern way to save VRAM
        if hasattr(model, "gradient_checkpointing_enable"):
            model.gradient_checkpointing_enable()
            logger.info("Gradient checkpointing enabled")
        
        # Load pretrained weights
        gpt_path = Path(self.config.model_dir) / cfg.gpt_checkpoint
        if gpt_path.exists():
            load_checkpoint(model, str(gpt_path))
            logger.info(f"Loaded pretrained weights from {gpt_path}")
        else:
            logger.warning(f"Pretrained weights not found at {gpt_path}")

        model = model.to(self.device)

        # Stage-specific freezing logic
        if self.config.stage == 2:
            # Stage 2: Train emotion, freeze speaker
            if hasattr(model, "conditioning_encoder"):
                model.conditioning_encoder.requires_grad_(False)
            if hasattr(model, "perceiver_encoder"):
                model.perceiver_encoder.requires_grad_(False)
            logger.info("Stage 2: Froze speaker conditioner")
            
        elif self.config.stage == 3:
            # Stage 3: Fine-tune, freeze all conditioners
            for module in [
                "conditioning_encoder", "perceiver_encoder", 
                "emo_conditioning_encoder", "emo_perceiver_encoder"
            ]:
                if hasattr(model, module):
                    getattr(model, module).requires_grad_(False)
            logger.info("Stage 3: Froze all feature conditioners")

        return model

    def _create_dummy_model(self) -> nn.Module:
        """Create a placeholder model for debugging."""
        model = nn.Sequential(
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 8192),
        ).to(self.device)
        # Mock attributes for trainer compatibility
        model.number_mel_codes = 8192
        return model

    def _setup_signal_handlers(self) -> None:
        """Setup graceful shutdown handlers."""

        self._signal_count = 0

        def handler(signum, frame):
            self._signal_count += 1
            if self._signal_count >= 2:
                logger.warning(f"Received signal {signum} again. Forcing exit...")
                import sys
                sys.exit(1)
            
            logger.warning(f"Received signal {signum}, initiating graceful shutdown... (Press Ctrl+C again to force exit)")
            self.should_stop = True

        signal.signal(signal.SIGINT, handler)
        signal.signal(signal.SIGTERM, handler)

    @trace_execution
    def _train_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """Execute a single training step.

        Args:
            batch: Batched training data.

        Returns:
            Dictionary of loss values.
        """
        self.optimizer.zero_grad()

        # Move batch to device
        text_tokens = batch["text_tokens"].to(self.device)
        semantic_tokens = batch["semantic_tokens"].to(self.device)
        conditioning_emb = batch["conditioning_emb"].to(self.device) # (B, T_cond, D)
        
        text_lengths = batch["text_lengths"].to(self.device)
        semantic_lengths = batch["semantic_lengths"].to(self.device)
        cond_lengths = batch["cond_lengths"].to(self.device)
        
        # 1. Feature Extraction / Conditioning
        # -----------------------------------
        # For Stage 1 (Duration Control), we use:
        # - Speaker conditioning (from audio via perceiver)
        # - Duration embedding (from target length)
        
        # Get speaker conditioning latent (B, 32, D)
        # Note: UnifiedVoice model computes this internally in forward() via do_spk_cond=True
        # providing conditioning_emb is enough.
        
        # 2. Duration Control Logic
        # -------------------------
        # We need to tell the model how many tokens to generate.
        # use_speed tensor contains the target length (or 0 for free generation)
        use_speed = semantic_lengths.clone()

        # Duration dropout (Stage 1):
        # randomly set p=0 to teach model free-form generation (30% prob)
        if self.config.stage == 1 and random.random() < self.config.duration_dropout:
            use_speed = torch.zeros_like(use_speed)

        # 3. Model Forward
        # ----------------
        # Forward pass returns latent representations
        # We handle next-token prediction
        
        # Ensure conditioning_emb is (B, D, T) format expected by conformer
        speech_cond_input = conditioning_emb.transpose(1, 2) 

        # Forward pass
        # returns: latents (B, T_seq, D_model)
        # Note: UnifiedVoice.forward expects specific arguments
        model_out = self.model(
            speech_conditioning_latent=speech_cond_input,
            text_inputs=text_tokens,
            text_lengths=text_lengths,
            mel_codes=semantic_tokens,
            mel_codes_lengths=semantic_lengths,
            emo_speech_conditioning_latent=speech_cond_input, # Use same audio for emotion in Stage 1
            cond_mel_lengths=cond_lengths,
            emo_cond_mel_lengths=cond_lengths,
            use_speed=use_speed,
            do_spk_cond=True,
        )
        
        # 4. Loss Computation
        # -------------------
        # L_ar = -1/(T+1) * sum(log(q(yt)))
        
        # Project latents to vocabulary size
        logits = self.model.mel_head(model_out) # (B, T_seq, vocab_size)
        
        # Target is shifted semantic tokens (predict next token)
        # semantic_tokens: [B, T] -> target: [B, T]
        # In UnifiedVoice, forward() handles the shift internally for input/target alignment?
        # Actually UnifiedVoice.forward() returns 'mel_logits[:, :-2]'
        # Let's align shapes carefully.
        
        # Standard AR training: 
        # Input: <SOS> t1 t2 t3 ... tN
        # Target: t1 t2 t3 ... tN <EOS>
        
        # UnifiedVoice forward() aligns internally but here returns raw latents.
        # We need to compute loss against shifted targets.
        
        # For simplicity, let's assume standard alignment:
        # logits shape: (B, T_model, V)
        # targets shape: (B, T_model)
        
        # The model output is already trimmed by 2 tokens in forward()
        # "return mel_logits[:, :-2]"
        
        # The internal build_aligned_inputs_and_targets adds 2 tokens (start/stop)
        # So output length matches semantic_tokens length + pad?
        
        vocab_size = self.model.number_mel_codes
        
        # Calculate Cross Entropy Loss
        # We resize to (N, V) and (N) for F.cross_entropy
        # The targets corresponding to model_out need to be aligned
        
        # Targets: semantic_tokens with padding handling
        # Since model handles alignment details, let's try mapping 1:1 first
        # Usually: logits match semantic_tokens sequence length + 1 (stop token)
        
        # To be safe for this integration step without deep diving into 
        # exact alignment indices of UnifiedVoice.forward:
        
        target_len = logits.shape[1]
        
        # Prepare targets (semantic tokens + stop token padded)
        # We need to pad semantic tokens to match logits length if needed
        # Or slice logits to match.
        
        # Let's trust proper padding in collate_fn and standard CE loss ignoring padding
        
        # Pad targets with STOP_MEL_TOKEN (8193)
        targets = F.pad(semantic_tokens, (0, 1), value=self.model.stop_mel_token)
        
        # Truncate/Pad targets to match logits length exactly
        if targets.shape[1] > target_len:
            targets = targets[:, :target_len]
        elif targets.shape[1] < target_len:
            targets = F.pad(targets, (0, target_len - targets.shape[1]), value=0)

        main_loss = F.cross_entropy(
            logits.reshape(-1, vocab_size),
            targets.reshape(-1),
            ignore_index=0, # Ignore padding
        )

        losses = {"main": main_loss.item()}
        total_loss = main_loss

        # Emotion adversarial loss (Stage 2) placeholder
        if self.config.stage == 2:
            # Add GRL loss here
            pass

        # Backward pass
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        self.optimizer.step()

        losses["total"] = total_loss.item()
        return losses

    @trace_execution
    def train(self) -> None:
        """Run the training loop."""
        logger.info(f"Starting Stage {self.config.stage} training")

        # Use animated model loading UI with stdout suppression
        with ModelLoadingUI() as loading_ui:
            # Load feature extractor with visual progress
            loading_ui.start_component("semantic")
            feature_extractor = None
            if not self.config.debug:
                try:
                    feature_extractor = FeatureExtractor(
                        model_dir=self.config.model_dir,
                        device=self.device
                    )
                    loading_ui.complete_component(success=True)
                except Exception as e:
                    logger.warning(f"Failed to initialize feature extractor: {e}")
                    logger.warning("Falling back to dummy features")
                    loading_ui.complete_component(success=False)
            else:
                loading_ui.complete_component(success=True)

            # Load tokenizer (already done in feature extractor, mark complete)
            loading_ui.start_component("tokenizer")
            loading_ui.complete_component(success=True)

            # Prepare dataset with visual progress
            loading_ui.start_component("dataset")
            
            # Select dataset based on config
            use_emotional = self.config.stage == 2
            dataset_type = getattr(self.config, "dataset_type", "local")
            
            # For streaming datasets, create progress UI first
            streaming_progress = None
            
            if dataset_type == "emilia":
                logger.debug(f"[train] Creating EmiliaDataset with lang={getattr(self.config, 'emilia_lang', 'EN')}, shards={getattr(self.config, 'emilia_shards', 50)}")
                if self.config.stage == 2:
                    logger.warning("Emilia is for Stage 1/3 (not emotional). Using Stage 2 ESD might be better.")
                
                # Create streaming progress UI FIRST so we can pass callback to dataset
                streaming_progress = StreamingProgressUI(
                    message="Streaming from HuggingFace...",
                    console=self.ui.console
                )
                streaming_progress.start()
                
                # Define callback that updates progress
                def on_sample_received(count: int):
                    streaming_progress.set_samples(count)
                
                # Emilia is IterableDataset
                dataset_create_start = time.perf_counter()
                dataset = EmiliaDataset(
                    feature_extractor=feature_extractor,
                    language=getattr(self.config, 'emilia_lang', 'EN'),
                    num_shards=getattr(self.config, 'emilia_shards', 50),
                    split="train",
                    max_samples=10000 if self.config.debug else None,
                    on_sample_callback=on_sample_received,
                )
                dataset_create_duration = time.perf_counter() - dataset_create_start
                logger.debug(f"[train] EmiliaDataset created in {dataset_create_duration:.2f}s")
                # Dataloader for IterableDataset
                # NOTE: num_workers=0 required for streaming datasets on Windows
                # due to multiprocessing spawn pickle issues with complex objects
                dataloader = DataLoader(
                    dataset,
                    batch_size=self.config.batch_size,
                    num_workers=0,  # Required for Windows + streaming
                    collate_fn=collate_fn,
                    pin_memory=torch.cuda.is_available(),
                )
                logger.info(f"Using Emilia (Streaming) Dataset: {getattr(self.config, 'emilia_shards', 50)} shards")
                
            elif dataset_type == "esd":
                dataset = ESDDataset(
                    data_dir=self.config.data_dir,
                    feature_extractor=feature_extractor,
                    max_samples=10000 if self.config.debug else None,
                )
                logger.info(f"Using ESD Dataset from {self.config.data_dir}")
                dataloader = DataLoader(
                    dataset,
                    batch_size=self.config.batch_size,
                    shuffle=True,
                    num_workers=self.config.num_workers,
                    collate_fn=collate_fn,
                    pin_memory=torch.cuda.is_available(),
                )
                
            else: # local
                dataset = LocalDurationDataset(
                    data_dir=self.config.data_dir,
                    feature_extractor=feature_extractor,
                    use_emotional=use_emotional,
                    max_samples=10000 if self.config.debug else None,
                )
                logger.info(f"Using Local Dataset from {self.config.data_dir}")
                dataloader = DataLoader(
                    dataset,
                    batch_size=self.config.batch_size,
                    shuffle=True,
                    num_workers=self.config.num_workers,
                    collate_fn=collate_fn,
                    pin_memory=torch.cuda.is_available(),
                )

            # Mark dataset loading complete
            loading_ui.complete_component(success=True)
            
            # Mark optimizer setup (already done in __init__)
            loading_ui.start_component("optimizer")
            loading_ui.complete_component(success=True)

        # Calculate total steps (outside loading UI context)
        if isinstance(dataset, IterableDataset):
            # Estimate or unlimited
            steps_per_epoch = 1000 # Placeholder for progress bar
        else:
            steps_per_epoch = len(dataloader)
        total_steps = self.config.max_steps or (steps_per_epoch * self.config.num_epochs)

        # Resume from checkpoint if available
        checkpoint_info = self.checkpoint_manager.load_latest(
            self.model, self.optimizer, self.scheduler
        )
        if checkpoint_info:
            self.step = checkpoint_info["step"]
            self.epoch = checkpoint_info["epoch"]
            self.loss_history = checkpoint_info.get("loss_history", [])
            logger.info(f"Resumed from step {self.step}")

        # Check if streaming dataset
        is_streaming = isinstance(dataset, IterableDataset)
        
        # For streaming datasets, wait for first batch before starting full UI
        # streaming_progress was already created in the dataset section if emilia
        first_batch_received = False
        first_batch = None
        
        if is_streaming and streaming_progress:
            # Get the first batch to confirm connection is working
            # The progress UI is already showing and will update via callbacks
            logger.info("Waiting for first batch from streaming dataset...")
            logger.info("(This may take several minutes due to network latency)")
            
            try:
                start_time = time.perf_counter()
                dataloader_iter = iter(dataloader)
                first_batch = next(dataloader_iter)
                batch_fetch_time = time.perf_counter() - start_time
                first_batch_received = True
                
                streaming_progress.stop(success=True)
                streaming_progress = None
                
                logger.info(f"✓ First batch received! (took {batch_fetch_time:.1f}s)")
            except StopIteration:
                streaming_progress.stop(success=False)
                raise RuntimeError("Streaming dataset returned no data")
            except Exception as e:
                streaming_progress.stop(success=False)
                raise
        
        # NOW start the main training UI (after streaming connection confirmed)
        self.ui.start()
        self.ui.update(
            stage=self.config.stage,
            total_steps=total_steps,
            step=self.step,
            epoch=self.epoch,
        )

        # Register callbacks
        self.ui.register_callback("pause", lambda p: setattr(self, "is_paused", p))
        self.ui.register_callback("save", self._save_checkpoint)

        try:
            self.model.train()
            start_time = time.perf_counter()
            tokens_processed = 0
            
            logger.debug(f"[train] Starting training loop - is_streaming={is_streaming}")
            logger.debug(f"[train] Dataloader config: batch_size={self.config.batch_size}, workers={self.config.num_workers}")

            for self.epoch in range(self.epoch, self.config.num_epochs):
                # Log start of epoch
                logger.info(f"Starting epoch {self.epoch + 1}/{self.config.num_epochs}")
                epoch_start_time = time.perf_counter()
                
                batch_idx = 0
                logger.debug(f"[train] Creating dataloader iterator for epoch {self.epoch + 1}...")
                
                # For streaming datasets on first epoch, we already have the first batch
                # Create an iterator that yields the pre-fetched batch first
                if is_streaming and first_batch is not None:
                    # Chain: first the pre-fetched batch, then the rest of the dataloader
                    import itertools
                    batch_iterator = itertools.chain([first_batch], dataloader_iter)
                    first_batch = None  # Clear so we don't reuse it
                else:
                    batch_iterator = iter(dataloader)
                
                for batch in batch_iterator:
                    batch_idx += 1
                    
                    # Periodic debug logging
                    if batch_idx % 50 == 0:
                        elapsed = time.perf_counter() - epoch_start_time
                        logger.debug(f"[train] Epoch {self.epoch+1}, batch {batch_idx}: {elapsed:.1f}s elapsed")
                    
                    if self.should_stop:
                        logger.debug(f"[train] should_stop=True at batch {batch_idx}")
                        break

                    while self.is_paused:
                        time.sleep(0.1)
                        self.ui.update(is_paused=True)

                    # Training step
                    losses = self._train_step(batch)
                    self.step += 1
                    self.loss_history.append(losses["total"])
                    tokens_processed += batch["semantic_tokens"].numel()

                    # Calculate speed metrics
                    elapsed = time.perf_counter() - start_time
                    tokens_per_sec = tokens_processed / elapsed if elapsed > 0 else 0
                    samples_per_sec = self.step / elapsed if elapsed > 0 else 0

                    # Get compute stats (GPU or CPU)
                    if torch.cuda.is_available():
                        gpu_used = torch.cuda.memory_allocated() / 1e9
                        gpu_total = torch.cuda.get_device_properties(0).total_memory / 1e9
                    else:
                        gpu_used, gpu_total = 0, 0
                        # Update CPU percent (only every 10 steps to reduce overhead)
                        if HAS_PSUTIL and self.step % 10 == 0:
                            self.ui.state.cpu_percent = psutil.cpu_percent()

                    # Update UI
                    self.ui.update(
                        step=self.step,
                        epoch=self.epoch + 1,
                        loss=losses["total"],
                        learning_rate=self.scheduler.get_last_lr()[0],
                        tokens_per_second=tokens_per_sec,
                        samples_per_second=samples_per_sec,
                        gpu_memory=(gpu_used, gpu_total),
                        best_loss=self.early_stopping.best_loss,
                        steps_no_improve=self.early_stopping.steps_without_improvement,
                        early_stopping_patience=self.config.early_stopping_patience,
                        is_paused=False,
                    )

                    # Save checkpoint
                    if self.checkpoint_manager.should_save(self.step):
                        self._save_checkpoint()

                    # Check step limit
                    if self.config.max_steps and self.step >= self.config.max_steps:
                        logger.info(f"Reached max steps: {self.config.max_steps}")
                        self.should_stop = True
                        break

                    # Early stopping check
                    if self.early_stopping(losses["total"]):
                        logger.info(
                            f"Early stopping: no improvement for {self.config.early_stopping_patience} steps"
                        )
                        self.should_stop = True
                        break

                self.scheduler.step()

                if self.should_stop:
                    break

        finally:
            self.ui.stop()
            self._save_checkpoint()
            logger.info(f"Training stopped at step {self.step}")

    def _save_checkpoint(self) -> None:
        """Save a training checkpoint."""
        loss = self.loss_history[-1] if self.loss_history else float("inf")
        self.checkpoint_manager.save(
            model=self.model,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            step=self.step,
            epoch=self.epoch,
            stage=self.config.stage,
            loss=loss,
            config=vars(self.config),
            loss_history=self.loss_history[-1000:],  # Keep last 1000
        )

    def rollback(self, step: int) -> None:
        """Rollback to a specific checkpoint.

        Args:
            step: Target step to rollback to.
        """
        info = self.checkpoint_manager.rollback(
            step, self.model, self.optimizer, self.scheduler
        )
        self.step = info["step"]
        self.epoch = info["epoch"]
        self.loss_history = info.get("loss_history", [])
        logger.info(f"Rolled back to step {step}")


# ═══════════════════════════════════════════════════════════════════════════════
# DATASET DOWNLOAD
# ═══════════════════════════════════════════════════════════════════════════════


def download_emilia_dataset(
    num_shards: int = 50,
    language: str = "EN",
    cache_dir: Optional[str] = None,
) -> None:
    """Download Emilia dataset shards for offline/HPC use.
    
    Pre-downloads specified number of shards from the Emilia dataset
    to the HuggingFace cache for offline training.
    
    Args:
        num_shards: Number of shards to download (default: 50).
        language: Language subset (EN, ZH, JA, KO, FR, DE).
        cache_dir: Optional custom cache directory.
        
    Raises:
        ImportError: If huggingface_hub is not installed.
        
    Example:
        >>> download_emilia_dataset(num_shards=50, language="EN")
        Downloading 50 shards of Emilia-EN...
    """
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        raise ImportError("huggingface_hub required: pip install huggingface_hub")
    
    logger.info("Emilia Dataset Downloader")
    logger.info(f"  Language: {language}")
    logger.info(f"  Shards:   {num_shards}")
    logger.info(f"  Pattern:  {patterns[0]} ... {patterns[-1]}")
    
    logger.warning("Make sure you have access to the gated dataset: https://huggingface.co/datasets/amphion/Emilia-Dataset")
    logger.info("Run 'huggingface-cli login' if you haven't authenticated.")
    
    try:
        logger.info("Starting download...")
        
        result = snapshot_download(
            repo_id="amphion/Emilia-Dataset",
            repo_type="dataset",
            allow_patterns=patterns,
            cache_dir=cache_dir,
        )
        
        logger.info("Download complete!")
        logger.info(f"  Cache location: {result}")
        logger.info("You can now run training with --dataset emilia")
        
    except Exception as e:
        logger.error(f"Download failed: {e}")
        logger.info("Troubleshooting:")
        logger.info("  1. Run: huggingface-cli login")
        logger.info("  2. Request access at: https://huggingface.co/datasets/amphion/Emilia-Dataset")
        logger.info("  3. Check your internet connection")
        raise


# ═══════════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════════


def main() -> None:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="IndexTTS2 Duration Control Training",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # Training stage
    parser.add_argument(
        "--stage", type=int, default=1, choices=[1, 2, 3],
        help="Training stage: 1=Base, 2=Emotion, 3=Fine-tune"
    )

    # Data/model paths
    parser.add_argument("--data-dir", type=str, default="./data")
    parser.add_argument("--model-dir", type=str, default="./checkpoints")
    parser.add_argument("--checkpoint-dir", type=str, default="./checkpoints/training")
    parser.add_argument(
        "--dataset", type=str, default="local", choices=["local", "emilia", "esd"],
        help="Dataset type: local (folder), emilia (streaming HF), esd (emotional)"
    )
    parser.add_argument(
        "--emilia-shards", type=int, default=50,
        help="Number of Emilia shards to use (default: 50)"
    )

    # Hyperparameters
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--duration-dropout", type=float, default=0.3)
    parser.add_argument("--emotion-alpha", type=float, default=1.0)

    # Checkpointing
    parser.add_argument("--save-interval", type=int, default=500)
    parser.add_argument("--max-checkpoints", type=int, default=5)
    parser.add_argument("--resume", type=str, default=None, help="Resume from checkpoint")
    parser.add_argument("--rollback", type=int, default=None, help="Rollback to step")

    # Early stopping
    parser.add_argument(
        "--early-stopping", type=int, default=0, metavar="PATIENCE",
        help="Stop if no improvement for N steps (0=disabled)"
    )
    parser.add_argument(
        "--early-stopping-delta", type=float, default=1e-4,
        help="Minimum loss improvement to count as progress"
    )

    # Performance
    parser.add_argument("--fp16", action="store_true", help="Use mixed precision")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)

    # Debug
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    parser.add_argument("--dry-run", action="store_true", help="Run one step only")
    parser.add_argument("--demo-ui", action="store_true", help="Demo UI only")
    
    # Dataset download
    parser.add_argument(
        "--download-emilia", type=int, default=None, metavar="N",
        help="Download first N shards of Emilia-EN dataset and exit (e.g., --download-emilia 50)"
    )
    parser.add_argument(
        "--emilia-lang", type=str, default="EN", choices=["EN", "ZH", "JA", "KO", "FR", "DE"],
        help="Language subset for Emilia download (default: EN)"
    )

    args = parser.parse_args()

    # Initialize Central Logging
    log_level = logging.DEBUG if args.debug else logging.INFO
    logger_manager.setup(
        name="train_duration",
        log_file="train_duration.log",
        level=log_level
    )

    # Demo mode
    if args.demo_ui:
        from training_ui import demo_ui
        demo_ui()
        return
    
    # Download Emilia dataset mode
    if args.download_emilia is not None:
        download_emilia_dataset(
            num_shards=args.download_emilia,
            language=args.emilia_lang,
        )
        return

    # Create config
    config = TrainingConfig(
        stage=args.stage,
        data_dir=args.data_dir,
        checkpoint_dir=args.checkpoint_dir,
        model_dir=args.model_dir,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        num_epochs=args.epochs,
        max_steps=1 if args.dry_run else args.max_steps,
        duration_dropout=args.duration_dropout,
        emotion_loss_alpha=args.emotion_alpha,
        save_interval=args.save_interval,
        max_checkpoints=args.max_checkpoints,
        early_stopping_patience=args.early_stopping,
        early_stopping_min_delta=args.early_stopping_delta,
        use_fp16=args.fp16,
        debug=args.debug,
        seed=args.seed,
        num_workers=args.workers,
    )
    # Inject dataset type and emilia settings into config
    setattr(config, "dataset_type", args.dataset)
    setattr(config, "emilia_shards", args.emilia_shards)
    setattr(config, "emilia_lang", args.emilia_lang)

    # Create trainer
    trainer = DurationTrainer(config)

    # Handle rollback
    if args.rollback is not None:
        trainer.rollback(args.rollback)
        return

    # Train
    trainer.train()


if __name__ == "__main__":
    main()
