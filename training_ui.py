#!/usr/bin/env python3
"""
╔═══════════════════════════════════════════════════════════════════════════════╗
║               IndexTTS2 Animated Training UI                                  ║
║                                                                               ║
║  Premium animated interface for monitoring training progress with             ║
║  real-time metrics, loss visualization, and interactive controls.            ║
╚═══════════════════════════════════════════════════════════════════════════════╝
"""

import functools
import logging
import time
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from logging.handlers import RotatingFileHandler
from typing import Any, Callable, Deque, Dict, List, Optional, Tuple

from rich import box
from rich.align import Align
from rich.columns import Columns
from rich.console import Console, Group, RenderableType
from rich.layout import Layout
from rich.live import Live
from rich.padding import Padding
from rich.panel import Panel
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TaskProgressColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)
from rich.rule import Rule
from rich.style import Style
from rich.table import Table
from rich.text import Text


# ═══════════════════════════════════════════════════════════════════════════════
# OBSERVABILITY SETUP
# ═══════════════════════════════════════════════════════════════════════════════


def setup_observability(debug_mode: bool = False) -> logging.Logger:
    """Initialize file-based logging for UI operations.

    Args:
        debug_mode (bool): If True, set log level to DEBUG.

    Returns:
        logging.Logger: Configured logger instance.
    """
    log_level = logging.DEBUG if debug_mode else logging.INFO
    logger = logging.getLogger("training_ui")
    logger.setLevel(log_level)

    if not logger.handlers:
        handler = RotatingFileHandler(
            "training_ui.log", maxBytes=5 * 1024 * 1024, backupCount=2
        )
        formatter = logging.Formatter(
            "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    return logger


logger = setup_observability()


def trace_execution(func):
    """Decorator for execution tracing."""

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
# BRANDING & STYLING
# ═══════════════════════════════════════════════════════════════════════════════

BRAND_COLOR = "cyan"
ACCENT_COLOR = "magenta"
SUCCESS_COLOR = "green"
WARNING_COLOR = "yellow"
ERROR_COLOR = "red"
MUTED_COLOR = "dim"

# Animated frames for loading indicator
SPINNER_FRAMES = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]
PULSE_FRAMES = ["◐", "◓", "◑", "◒"]

STAGE_COLORS = {
    1: "bright_blue",
    2: "bright_magenta",
    3: "bright_green",
}

STAGE_NAMES = {
    1: "Base Training",
    2: "Emotion Control",
    3: "Fine-Tuning",
}

LOGO = """
[bold cyan]╦[/][cyan]┌┐┌┌┬┐┌─┐─┐ ┬[/][bold cyan]╔╦╗╔╦╗╔═╗[/][cyan]2[/]
[bold cyan]║[/][cyan]│││ ││├┤ ┌┴┬┘[/][bold cyan] ║  ║ ╚═╗[/]
[bold cyan]╩[/][cyan]┘└┘─┴┘└─┘┴ └─[/][bold cyan] ╩  ╩ ╚═╝[/]
"""


# ═══════════════════════════════════════════════════════════════════════════════
# SPARKLINE VISUALIZATION
# ═══════════════════════════════════════════════════════════════════════════════


def create_sparkline(values: List[float], width: int = 20, style: str = "green") -> str:
    """Create a mini sparkline chart from values.

    Args:
        values: List of numeric values.
        width: Maximum width of sparkline.
        style: Rich color style.

    Returns:
        Sparkline string with Rich markup.

    Example:
        >>> create_sparkline([1, 2, 3, 2, 1], width=10)
        '[green]▁▄█▄▁[/green]'
    """
    if not values:
        return f"[{MUTED_COLOR}]─" * width + f"[/{MUTED_COLOR}]"

    # Subsample if too many values
    if len(values) > width:
        step = len(values) / width
        values = [values[int(i * step)] for i in range(width)]

    min_val = min(values)
    max_val = max(values)
    val_range = max_val - min_val if max_val != min_val else 1

    # Unicode blocks for different heights
    blocks = " ▁▂▃▄▅▆▇█"

    sparkline = ""
    for val in values:
        normalized = (val - min_val) / val_range
        idx = int(normalized * (len(blocks) - 1))
        sparkline += blocks[idx]

    return f"[{style}]{sparkline}[/{style}]"


def create_gpu_bar(used: float, total: float, width: int = 20) -> str:
    """Create an animated GPU memory bar.

    Args:
        used: Used memory in GB.
        total: Total memory in GB.
        width: Bar width.

    Returns:
        GPU memory bar with Rich markup.
    """
    pct = used / total if total > 0 else 0
    filled = int(pct * width)
    empty = width - filled

    if pct < 0.7:
        color = SUCCESS_COLOR
    elif pct < 0.9:
        color = WARNING_COLOR
    else:
        color = ERROR_COLOR

    bar = f"[{color}]{'█' * filled}[/{color}][dim]{'░' * empty}[/dim]"
    return f"{bar} {used:.1f}/{total:.1f}GB"


# ═══════════════════════════════════════════════════════════════════════════════
# TRAINING STATE
# ═══════════════════════════════════════════════════════════════════════════════


@dataclass
class TrainingState:
    """Container for training state metrics.

    Attributes:
        step: Current training step.
        epoch: Current epoch.
        stage: Training stage (1, 2, 3).
        total_steps: Total steps for current stage.
        loss: Current loss value.
        loss_history: Recent loss values for visualization.
        learning_rate: Current learning rate.
        tokens_per_second: Training speed.
        samples_per_second: Samples processed per second.
        gpu_memory_used: GPU memory used in GB (0 if CPU-only).
        gpu_memory_total: Total GPU memory in GB (0 if CPU-only).
        cpu_percent: CPU utilization percentage (for CPU-only mode).
        is_gpu_mode: Whether running on GPU.
        best_loss: Best loss achieved (for early stopping).
        steps_no_improve: Steps without improvement.
        early_stopping_patience: Total patience (0=disabled).
        emotion_loss: Emotion classification loss (Stage 2).
        is_paused: Whether training is paused.
        start_time: Training start datetime.
    """

    step: int = 0
    epoch: int = 0
    stage: int = 1
    total_steps: int = 100000
    loss: float = 0.0
    loss_history: Deque[float] = field(default_factory=lambda: deque(maxlen=50))
    learning_rate: float = 2e-4
    tokens_per_second: float = 0.0
    samples_per_second: float = 0.0
    gpu_memory_used: float = 0.0
    gpu_memory_total: float = 0.0
    cpu_percent: float = 0.0
    is_gpu_mode: bool = True
    best_loss: Optional[float] = None
    steps_no_improve: int = 0
    early_stopping_patience: int = 0
    emotion_loss: Optional[float] = None
    is_paused: bool = False
    start_time: datetime = field(default_factory=datetime.now)

    def update_loss(self, loss: float) -> None:
        """Update loss and add to history.

        Args:
            loss: New loss value.
        """
        self.loss = loss
        self.loss_history.append(loss)

    def elapsed_time(self) -> timedelta:
        """Get elapsed training time.

        Returns:
            Time delta since training started.
        """
        return datetime.now() - self.start_time

    def estimated_remaining(self) -> Optional[timedelta]:
        """Get estimated time remaining.

        Returns:
            Estimated time delta, or None if cannot compute.
        """
        if self.step == 0 or self.samples_per_second == 0:
            return None
        remaining_steps = self.total_steps - self.step
        seconds_remaining = remaining_steps / max(self.samples_per_second, 0.001)
        return timedelta(seconds=int(seconds_remaining))


# ═══════════════════════════════════════════════════════════════════════════════
# UI COMPONENTS
# ═══════════════════════════════════════════════════════════════════════════════


def create_header(state: TrainingState, frame: int = 0) -> Panel:
    """Create animated header with stage indicator.

    Args:
        state: Current training state.
        frame: Animation frame counter.

    Returns:
        Panel with header content.
    """
    stage_color = STAGE_COLORS.get(state.stage, BRAND_COLOR)
    stage_name = STAGE_NAMES.get(state.stage, f"Stage {state.stage}")

    # Animated pulse indicator
    pulse = PULSE_FRAMES[frame % len(PULSE_FRAMES)]

    status = "PAUSED" if state.is_paused else "TRAINING"
    status_color = WARNING_COLOR if state.is_paused else SUCCESS_COLOR

    header_text = Text.from_markup(
        LOGO
        + f"\n[bold white]Duration Control Training[/bold white]\n"
        f"[{stage_color}]{pulse} Stage {state.stage}: {stage_name}[/{stage_color}] "
        f"[{status_color}]● {status}[/{status_color}]"
    )

    return Panel(
        Align.center(header_text),
        box=box.DOUBLE_EDGE,
        border_style=stage_color,
        padding=(0, 2),
    )


def create_metric_card(
    label: str, value: str, style: str = "white"
) -> Panel:
    """Create a single metric display card.

    Args:
        label: Metric label.
        value: Metric value.
        style: Value color style.

    Returns:
        Panel with metric content.
    """
    return Panel(
        Align.center(Text.from_markup(f"[bold {style}]{value}[/bold {style}]")),
        title=f"[dim]{label}[/dim]",
        box=box.ROUNDED,
        border_style=style if style != "dim" else MUTED_COLOR,
    )


def create_metrics_row(state: TrainingState) -> Columns:
    """Create row of metric cards.

    Args:
        state: Current training state.

    Returns:
        Columns of metric cards.
    """
    elapsed = state.elapsed_time()
    elapsed_str = str(elapsed).split(".")[0]

    remaining = state.estimated_remaining()
    remaining_str = str(remaining).split(".")[0] if remaining else "..."

    loss_style = SUCCESS_COLOR if state.loss < 1.0 else WARNING_COLOR

    cards = [
        create_metric_card("Step", f"{state.step:,}", BRAND_COLOR),
        create_metric_card("Epoch", str(state.epoch), ACCENT_COLOR),
        create_metric_card("Loss", f"{state.loss:.4f}", loss_style),
        create_metric_card("LR", f"{state.learning_rate:.2e}", MUTED_COLOR),
        create_metric_card("Elapsed", elapsed_str, MUTED_COLOR),
        create_metric_card("ETA", remaining_str, MUTED_COLOR),
    ]

    return Columns(cards, equal=True, expand=True)


def create_loss_panel(state: TrainingState) -> Panel:
    """Create loss visualization panel with sparkline.

    Args:
        state: Current training state.

    Returns:
        Panel with loss curve.
    """
    sparkline = create_sparkline(list(state.loss_history), width=40, style=SUCCESS_COLOR)
    
    min_loss = min(state.loss_history) if state.loss_history else 0
    max_loss = max(state.loss_history) if state.loss_history else 0

    content = Text.from_markup(
        f"\n{sparkline}\n\n"
        f"[dim]Min: {min_loss:.4f}  Max: {max_loss:.4f}  Current: {state.loss:.4f}[/dim]"
    )

    return Panel(
        Align.center(content),
        title="[bold]Loss Curve[/bold]",
        box=box.ROUNDED,
        border_style=BRAND_COLOR,
        padding=(0, 2),
    )


def create_compute_panel(state: TrainingState) -> Panel:
    """Create GPU/CPU monitoring panel.

    Shows GPU memory if available, otherwise shows CPU usage.

    Args:
        state: Current training state.

    Returns:
        Panel with compute stats.
    """
    if state.is_gpu_mode and state.gpu_memory_total > 0:
        # GPU mode
        gpu_bar = create_gpu_bar(state.gpu_memory_used, state.gpu_memory_total, width=30)
        title = "GPU Memory"
        resource_line = gpu_bar
    else:
        # CPU mode
        cpu_pct = state.cpu_percent
        width = 30
        filled = int(cpu_pct / 100 * width)
        empty = width - filled

        if cpu_pct < 70:
            color = SUCCESS_COLOR
        elif cpu_pct < 90:
            color = WARNING_COLOR
        else:
            color = ERROR_COLOR

        cpu_bar = f"[{color}]{'█' * filled}[/{color}][dim]{'░' * empty}[/dim] {cpu_pct:.0f}%"
        title = "CPU Usage"
        resource_line = cpu_bar

    content = Text.from_markup(
        f"\n{resource_line}\n\n"
        f"[dim]Speed: {state.tokens_per_second:.0f} tok/s  |  "
        f"{state.samples_per_second:.1f} samples/s[/dim]"
    )

    return Panel(
        Align.center(content),
        title=f"[bold]{title}[/bold]",
        box=box.ROUNDED,
        border_style=ACCENT_COLOR,
        padding=(0, 2),
    )


def create_stage_progress(state: TrainingState) -> Panel:
    """Create stage progress indicator.

    Args:
        state: Current training state.

    Returns:
        Panel with stage progress.
    """
    progress_pct = (state.step / state.total_steps) * 100 if state.total_steps > 0 else 0
    filled = int(progress_pct / 2)
    empty = 50 - filled

    stage_color = STAGE_COLORS.get(state.stage, BRAND_COLOR)
    bar = f"[{stage_color}]{'▓' * filled}[/{stage_color}][dim]{'░' * empty}[/dim]"

    content = Text.from_markup(
        f"\n{bar}\n\n"
        f"[bold]{progress_pct:.1f}%[/bold] complete  "
        f"[dim]({state.step:,} / {state.total_steps:,} steps)[/dim]"
    )

    return Panel(
        Align.center(content),
        title=f"[bold]Stage {state.stage} Progress[/bold]",
        box=box.ROUNDED,
        border_style=stage_color,
        padding=(0, 2),
    )


def create_early_stopping_panel(state: TrainingState) -> Panel:
    """Create early stopping status panel.

    Args:
        state: Current training state.

    Returns:
        Panel with early stopping info.
    """
    if state.early_stopping_patience == 0:
        content = Text.from_markup(
            f"\n[dim]Disabled[/dim]\n\n"
            f"[dim]Use --early-stopping N to enable[/dim]"
        )
        style = MUTED_COLOR
    else:
        # Calculate progress toward early stopping
        pct = (state.steps_no_improve / state.early_stopping_patience) * 100
        best_str = f"{state.best_loss:.4f}" if state.best_loss is not None else "--"
        
        if pct < 50:
            style = SUCCESS_COLOR
            status = "Improving"
        elif pct < 80:
            style = WARNING_COLOR
            status = "Stalling"
        else:
            style = ERROR_COLOR
            status = "Near stop"

        content = Text.from_markup(
            f"\n[bold]Best: {best_str}[/bold]\n"
            f"[{style}]{state.steps_no_improve}/{state.early_stopping_patience} steps[/{style}]\n"
            f"[{style}]{status}[/{style}]"
        )

    return Panel(
        Align.center(content),
        title="[bold]Early Stopping[/bold]",
        box=box.ROUNDED,
        border_style=style,
        padding=(0, 2),
    )


def create_controls_panel() -> Panel:
    """Create keyboard controls help panel.

    Returns:
        Panel with controls info.
    """
    content = Text.from_markup(
        "[dim]"
        "[bold]Ctrl+C[/bold] Graceful shutdown (saves checkpoint)"
        "[/dim]"
    )

    return Panel(
        Align.center(content),
        box=box.SIMPLE,
        border_style=MUTED_COLOR,
    )


def show_loading_animation(console: Console, message: str = "Initializing", duration: float = 2.0) -> None:
    """Display animated loading screen during initialization.

    Args:
        console: Rich console instance.
        message: Loading message to display.
        duration: How long to show animation (seconds).

    Example:
        >>> console = Console()
        >>> show_loading_animation(console, "Loading model...")
    """
    frames = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]
    steps = [
        "Loading configuration...",
        "Initializing model...",
        "Setting up optimizer...",
        "Preparing data loader...",
        "Starting training UI...",
    ]

    start_time = time.perf_counter()
    frame_idx = 0
    step_idx = 0

    with console.screen():
        while time.perf_counter() - start_time < duration:
            frame = frames[frame_idx % len(frames)]
            step = steps[min(step_idx, len(steps) - 1)]

            content = Text.from_markup(
                LOGO +
                f"\n\n[bold cyan]{frame}[/bold cyan] [white]{message}[/white]\n\n"
                f"[dim]{step}[/dim]"
            )

            panel = Panel(
                Align.center(content),
                box=box.DOUBLE_EDGE,
                border_style=BRAND_COLOR,
                padding=(2, 4),
            )

            console.clear()
            console.print(Align.center(panel, vertical="middle"))

            frame_idx += 1
            if frame_idx % 5 == 0:
                step_idx += 1

            time.sleep(0.1)


# ═══════════════════════════════════════════════════════════════════════════════
# MODEL LOADING UI WITH STDOUT SUPPRESSION
# ═══════════════════════════════════════════════════════════════════════════════


class SuppressStdout:
    """Context manager to suppress stdout during noisy operations.
    
    Captures all print() output and stores it for optional access later.
    
    Example:
        >>> with SuppressStdout() as suppressor:
        ...     print("This won't appear")
        >>> print(suppressor.captured)  # Access captured output
    """
    
    def __init__(self):
        import io
        import sys
        self._stdout = None
        self._buffer = io.StringIO()
        
    def __enter__(self):
        import sys
        self._stdout = sys.stdout
        sys.stdout = self._buffer
        return self
    
    def __exit__(self, *args):
        import sys
        sys.stdout = self._stdout
        
    @property
    def captured(self) -> str:
        return self._buffer.getvalue()


class ModelLoadingUI:
    """Animated model loading progress display.
    
    Shows each component loading with spinners and visual feedback,
    suppressing noisy print statements from underlying libraries.
    
    Attributes:
        console: Rich console instance.
        components: List of components to load.
        current_idx: Current component index.
        
    Example:
        >>> ui = ModelLoadingUI()
        >>> with ui:
        ...     ui.start_component("GPT Model")
        ...     model = load_gpt()  # Noisy prints suppressed
        ...     ui.complete_component()
    """
    
    COMPONENTS = [
        ("config", "Loading configuration", "⚙️"),
        ("gpt", "Loading GPT model", "🧠"),
        ("semantic", "Loading semantic model", "🔊"),
        ("codec", "Loading MaskGCT codec", "🎵"),
        ("tokenizer", "Loading BPE tokenizer", "📝"),
        ("optimizer", "Setting up optimizer", "📈"),
        ("dataset", "Preparing dataset", "📦"),
    ]
    
    def __init__(self):
        self.console = Console()
        self.live = None
        self.current_idx = 0
        self.completed = []
        self.start_time = None
        self._suppressor = None
        
    def __enter__(self):
        self.start_time = time.perf_counter()
        self._suppressor = SuppressStdout()
        self._suppressor.__enter__()
        self.live = Live(
            self._render(),
            console=self.console,
            refresh_per_second=10,
            transient=True,
        )
        self.live.start()
        return self
        
    def __exit__(self, *args):
        if self.live:
            self.live.stop()
        if self._suppressor:
            self._suppressor.__exit__(*args)
        # Show final completion status
        elapsed = time.perf_counter() - self.start_time
        self.console.print(
            f"\n[{SUCCESS_COLOR}]✓[/{SUCCESS_COLOR}] "
            f"[bold]All components loaded[/bold] "
            f"[dim]({elapsed:.1f}s)[/dim]\n"
        )
        
    def start_component(self, name: str) -> None:
        """Mark a component as started.
        
        Args:
            name: Component key from COMPONENTS list.
        """
        for i, (key, _, _) in enumerate(self.COMPONENTS):
            if key == name:
                self.current_idx = i
                break
        if self.live:
            self.live.update(self._render())
            
    def complete_component(self, success: bool = True) -> None:
        """Mark current component as complete.
        
        Args:
            success: Whether component loaded successfully.
        """
        if self.current_idx < len(self.COMPONENTS):
            key = self.COMPONENTS[self.current_idx][0]
            self.completed.append((key, success))
            self.current_idx += 1
        if self.live:
            self.live.update(self._render())
            
    def _render(self) -> Panel:
        """Render the loading UI."""
        elapsed = time.perf_counter() - self.start_time if self.start_time else 0
        
        lines = []
        lines.append(LOGO)
        lines.append("")
        lines.append(f"[bold white]Initializing Training Environment[/bold white]")
        lines.append("")
        
        for i, (key, label, icon) in enumerate(self.COMPONENTS):
            if i < len(self.completed):
                # Completed
                success = self.completed[i][1]
                if success:
                    lines.append(f"  [{SUCCESS_COLOR}]✓[/{SUCCESS_COLOR}] {icon} {label}")
                else:
                    lines.append(f"  [{ERROR_COLOR}]✗[/{ERROR_COLOR}] {icon} {label}")
            elif i == self.current_idx:
                # Currently loading
                frame = SPINNER_FRAMES[int(elapsed * 10) % len(SPINNER_FRAMES)]
                lines.append(f"  [{BRAND_COLOR}]{frame}[/{BRAND_COLOR}] {icon} [bold]{label}...[/bold]")
            else:
                # Pending
                lines.append(f"  [{MUTED_COLOR}]○[/{MUTED_COLOR}] [dim]{icon} {label}[/dim]")
        
        lines.append("")
        lines.append(f"[dim]Elapsed: {elapsed:.1f}s[/dim]")
        
        content = Text.from_markup("\n".join(lines))
        
        return Panel(
            Align.center(content),
            box=box.DOUBLE_EDGE,
            border_style=BRAND_COLOR,
            padding=(1, 4),
        )


# ═══════════════════════════════════════════════════════════════════════════════
# BUFFERING SPINNER
# ═══════════════════════════════════════════════════════════════════════════════


class BufferingSpinner:
    """Animated spinner for showing data buffering progress.
    
    Shows a spinner with elapsed time while waiting for streaming data.
    Use as context manager or call start()/stop() manually.
    
    Attributes:
        console: Rich console instance.
        message: Message to display.
        start_time: When buffering started.
        
    Example:
        >>> with BufferingSpinner("Waiting for data..."):
        ...     data = fetch_data()  # Long operation
    """
    
    def __init__(self, message: str = "Buffering streaming data...", console: Optional[Console] = None):
        """Initialize buffering spinner.
        
        Args:
            message: Message to display while waiting.
            console: Rich console instance.
        """
        self.message = message
        self.console = console or Console()
        self.live = None
        self.start_time = None
        self._stop_flag = False
        self._thread = None
        
    def __enter__(self):
        self.start()
        return self
        
    def __exit__(self, *args):
        self.stop()
        
    def start(self) -> None:
        """Start the buffering spinner animation."""
        import threading
        
        self.start_time = time.perf_counter()
        self._stop_flag = False
        
        # Use Rich's Progress for the spinner
        self.progress = Progress(
            SpinnerColumn(style=BRAND_COLOR),
            TextColumn("[progress.description]{task.description}"),
            TimeElapsedColumn(),
            console=self.console,
            transient=True,
        )
        self.progress.start()
        self.task_id = self.progress.add_task(self.message, total=None)
        
    def stop(self, success: bool = True) -> None:
        """Stop the buffering spinner.
        
        Args:
            success: Whether the operation completed successfully.
        """
        if self.progress:
            self.progress.stop()
            elapsed = time.perf_counter() - self.start_time if self.start_time else 0
            
            if success:
                self.console.print(
                    f"[{SUCCESS_COLOR}]✓[/{SUCCESS_COLOR}] {self.message} "
                    f"[dim]({elapsed:.1f}s)[/dim]"
                )
            else:
                self.console.print(
                    f"[{ERROR_COLOR}]✗[/{ERROR_COLOR}] {self.message} "
                    f"[dim](failed after {elapsed:.1f}s)[/dim]"
                )
                
    def update_message(self, message: str) -> None:
        """Update the spinner message.
        
        Args:
            message: New message to display.
        """
        self.message = message
        if self.progress and self.task_id is not None:
            self.progress.update(self.task_id, description=message)


# ═══════════════════════════════════════════════════════════════════════════════
# STREAMING PROGRESS UI
# ═══════════════════════════════════════════════════════════════════════════════


class StreamingProgressUI:
    """Progress indicator for streaming dataset downloads.
    
    Shows samples fetched, elapsed time, and throughput in real-time.
    Use as context manager or call start()/stop() manually.
    
    Attributes:
        console: Rich console instance.
        samples_fetched: Number of samples received.
        start_time: When streaming started.
        
    Example:
        >>> with StreamingProgressUI() as progress:
        ...     for batch in dataloader:
        ...         progress.update(samples=len(batch))
    """
    
    def __init__(
        self, 
        message: str = "Streaming from HuggingFace...", 
        console: Optional[Console] = None
    ):
        """Initialize streaming progress UI.
        
        Args:
            message: Message to display while streaming.
            console: Rich console instance.
        """
        self.message = message
        self.console = console or Console()
        self.progress = None
        self.task_id = None
        self.start_time = None
        self.samples_fetched = 0
        
    def __enter__(self):
        self.start()
        return self
        
    def __exit__(self, *args):
        self.stop()
        
    def start(self) -> None:
        """Start the streaming progress display."""
        self.start_time = time.perf_counter()
        self.samples_fetched = 0
        
        self.progress = Progress(
            SpinnerColumn(style=BRAND_COLOR),
            TextColumn("[progress.description]{task.description}"),
            TextColumn("•"),
            TimeElapsedColumn(),
            TextColumn("•"),
            TextColumn("[cyan]{task.fields[samples]}[/cyan] samples"),
            TextColumn("•"),
            TextColumn("[dim]{task.fields[throughput]}[/dim]"),
            console=self.console,
            transient=False,  # Keep visible after completion
        )
        self.progress.start()
        self.task_id = self.progress.add_task(
            self.message, 
            total=None,
            samples="0",
            throughput="-- samples/s"
        )
        
    def stop(self, success: bool = True) -> None:
        """Stop the streaming progress display.
        
        Args:
            success: Whether the operation completed successfully.
        """
        if self.progress:
            self.progress.stop()
            elapsed = time.perf_counter() - self.start_time if self.start_time else 0
            throughput = self.samples_fetched / elapsed if elapsed > 0 else 0
            
            if success:
                self.console.print(
                    f"[{SUCCESS_COLOR}]✓[/{SUCCESS_COLOR}] Streaming ready! "
                    f"[dim]{self.samples_fetched} samples in {elapsed:.1f}s "
                    f"({throughput:.1f} samples/s)[/dim]"
                )
            else:
                self.console.print(
                    f"[{ERROR_COLOR}]✗[/{ERROR_COLOR}] Streaming failed "
                    f"[dim](after {elapsed:.1f}s)[/dim]"
                )
                
    def update(self, samples: int = 1) -> None:
        """Update the progress with newly fetched samples.
        
        Args:
            samples: Number of new samples fetched (added to total).
        """
        self.samples_fetched += samples
        self._refresh_display()
    
    def set_samples(self, count: int) -> None:
        """Set the sample count directly (for callbacks).
        
        Args:
            count: Total number of samples fetched.
        """
        self.samples_fetched = count
        self._refresh_display()
    
    def _refresh_display(self) -> None:
        """Refresh the progress display with current values."""
        elapsed = time.perf_counter() - self.start_time if self.start_time else 0
        throughput = self.samples_fetched / elapsed if elapsed > 0 else 0
        
        if self.progress and self.task_id is not None:
            self.progress.update(
                self.task_id,
                samples=str(self.samples_fetched),
                throughput=f"{throughput:.1f} samples/s"
            )


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN UI CLASS
# ═══════════════════════════════════════════════════════════════════════════════


class TrainingUI:
    """Animated training UI with real-time updates.

    Attributes:
        console: Rich console instance.
        state: Current training state.
        live: Rich Live display.
        frame: Animation frame counter.
        is_running: Whether UI is active.

    Example:
        >>> ui = TrainingUI()
        >>> ui.start()
        >>> ui.update(step=100, loss=0.5, gpu_memory=(40, 80))
        >>> ui.stop()
    """

    def __init__(self, debug: bool = False) -> None:
        """Initialize training UI.

        Args:
            debug: Enable debug logging.
        """
        global logger
        if debug:
            logger = setup_observability(debug_mode=True)

        self.console = Console()
        self.state = TrainingState()
        self.live: Optional[Live] = None
        self.frame = 0
        self.is_running = False
        self._callbacks: Dict[str, Callable] = {}

    @trace_execution
    def start(self) -> None:
        """Start the live UI display."""
        self.is_running = True
        self.state.start_time = datetime.now()
        self.live = Live(
            self._render(),
            console=self.console,
            refresh_per_second=10,
            screen=True,
        )
        self.live.start()
        logger.info("Training UI started")

    def stop(self) -> None:
        """Stop the live UI display."""
        self.is_running = False
        if self.live:
            self.live.stop()
            self.live = None
        logger.info("Training UI stopped")

    def update(
        self,
        step: Optional[int] = None,
        epoch: Optional[int] = None,
        stage: Optional[int] = None,
        total_steps: Optional[int] = None,
        loss: Optional[float] = None,
        learning_rate: Optional[float] = None,
        tokens_per_second: Optional[float] = None,
        samples_per_second: Optional[float] = None,
        gpu_memory: Optional[Tuple[float, float]] = None,
        best_loss: Optional[float] = None,
        steps_no_improve: Optional[int] = None,
        early_stopping_patience: Optional[int] = None,
        emotion_loss: Optional[float] = None,
        is_paused: Optional[bool] = None,
    ) -> None:
        """Update training state and refresh display.

        Args:
            step: Current step.
            epoch: Current epoch.
            stage: Training stage.
            total_steps: Total steps in stage.
            loss: Current loss.
            learning_rate: Current LR.
            tokens_per_second: Speed metric.
            samples_per_second: Speed metric.
            gpu_memory: Tuple of (used, total) in GB.
            best_loss: Best loss for early stopping.
            steps_no_improve: Steps without improvement.
            early_stopping_patience: Total patience steps.
            emotion_loss: Emotion loss (Stage 2).
            is_paused: Pause state.
        """
        if step is not None:
            self.state.step = step
        if epoch is not None:
            self.state.epoch = epoch
        if stage is not None:
            self.state.stage = stage
        if total_steps is not None:
            self.state.total_steps = total_steps
        if loss is not None:
            self.state.update_loss(loss)
        if learning_rate is not None:
            self.state.learning_rate = learning_rate
        if tokens_per_second is not None:
            self.state.tokens_per_second = tokens_per_second
        if samples_per_second is not None:
            self.state.samples_per_second = samples_per_second
        if gpu_memory is not None:
            self.state.gpu_memory_used, self.state.gpu_memory_total = gpu_memory
            self.state.is_gpu_mode = gpu_memory[1] > 0
        if best_loss is not None:
            self.state.best_loss = best_loss
        if steps_no_improve is not None:
            self.state.steps_no_improve = steps_no_improve
        if early_stopping_patience is not None:
            self.state.early_stopping_patience = early_stopping_patience
        if emotion_loss is not None:
            self.state.emotion_loss = emotion_loss
        if is_paused is not None:
            self.state.is_paused = is_paused

        self.frame += 1
        if self.live:
            self.live.update(self._render())

    def _render(self) -> RenderableType:
        """Render the full UI layout.

        Returns:
            Renderable UI layout.
        """
        layout = Layout()

        layout.split_column(
            Layout(create_header(self.state, self.frame), name="header", size=8),
            Layout(create_metrics_row(self.state), name="metrics", size=4),
            Layout(name="charts", size=8),
            Layout(create_stage_progress(self.state), name="progress", size=6),
            Layout(name="extras", size=6),
            Layout(create_controls_panel(), name="controls", size=3),
        )

        layout["charts"].split_row(
            Layout(create_loss_panel(self.state)),
            Layout(create_compute_panel(self.state)),
        )

        layout["extras"].split_row(
            Layout(create_early_stopping_panel(self.state)),
        )

        return layout

    def register_callback(self, event: str, callback: Callable) -> None:
        """Register callback for UI events.

        Args:
            event: Event name (pause, save, quit).
            callback: Callback function.
        """
        self._callbacks[event] = callback

    def toggle_pause(self) -> None:
        """Toggle pause state."""
        self.state.is_paused = not self.state.is_paused
        if "pause" in self._callbacks:
            self._callbacks["pause"](self.state.is_paused)

    def request_save(self) -> None:
        """Request checkpoint save."""
        if "save" in self._callbacks:
            self._callbacks["save"]()


# ═══════════════════════════════════════════════════════════════════════════════
# DEMO MODE
# ═══════════════════════════════════════════════════════════════════════════════


def demo_ui() -> None:
    """Run a demo of the training UI with simulated data."""
    import random
    import signal
    import sys

    ui = TrainingUI()
    ui.start()

    # Handle Ctrl+C gracefully
    def signal_handler(sig, frame):
        ui.stop()
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)

    try:
        loss = 2.0
        step = 0
        epoch = 1

        while True:
            step += 1
            loss = max(0.1, loss - random.uniform(0.001, 0.01) + random.uniform(0, 0.005))

            if step % 1000 == 0:
                epoch += 1

            ui.update(
                step=step,
                epoch=epoch,
                stage=1 if step < 10000 else (2 if step < 20000 else 3),
                total_steps=30000,
                loss=loss,
                learning_rate=2e-4 * (0.99 ** (step // 1000)),
                tokens_per_second=random.uniform(8000, 12000),
                samples_per_second=random.uniform(30, 50),
                gpu_memory=(random.uniform(40, 60), 80.0),
                duration_accuracy=min(99, 80 + step / 500),
            )

            time.sleep(0.1)

    except KeyboardInterrupt:
        ui.stop()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Training UI Demo")
    parser.add_argument("--demo", action="store_true", help="Run demo mode")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    args = parser.parse_args()

    if args.demo or True:  # Default to demo
        demo_ui()
