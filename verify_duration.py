#!/usr/bin/env python3
"""
╔═══════════════════════════════════════════════════════════════════════════════╗
║                    IndexTTS2 Duration Control Verification                    ║
║                                                                               ║
║  Tests the duration control feature by generating speech with specified       ║
║  target durations and measuring actual output duration accuracy.              ║
╚═══════════════════════════════════════════════════════════════════════════════╝

Usage:
    uv run verify_duration.py
    uv run verify_duration.py --duration 5.0
    uv run verify_duration.py --prompt voice.wav --text "Hello" --duration 3 5 8
"""

import argparse
import os
import sys
import time
import logging
from pathlib import Path
from typing import Optional

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from rich.console import Console, Group
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn, TaskProgressColumn
from rich.table import Table
from rich.text import Text
from rich.live import Live
from rich.layout import Layout
from rich.align import Align
from rich import box
from rich.rule import Rule
from rich.padding import Padding
from rich.columns import Columns
from rich.markdown import Markdown
from rich.prompt import Prompt, Confirm
from rich.syntax import Syntax
from rich.style import Style

console = Console()


# ═══════════════════════════════════════════════════════════════════════════════
# BRANDING & STYLING
# ═══════════════════════════════════════════════════════════════════════════════

BRAND_COLOR = "cyan"
SUCCESS_COLOR = "green"
WARNING_COLOR = "yellow"
ERROR_COLOR = "red"
MUTED_COLOR = "dim"

LOGO = """
[bold cyan]╦[/][cyan]┌┐┌┌┬┐┌─┐─┐ ┬[/][bold cyan]╔╦╗╔╦╗╔═╗[/][cyan]2[/]
[bold cyan]║[/][cyan]│││ ││├┤ ┌┴┬┘[/][bold cyan] ║  ║ ╚═╗[/]
[bold cyan]╩[/][cyan]┘└┘─┴┘└─┘┴ └─[/][bold cyan] ╩  ╩ ╚═╝[/]
"""

HEADER = """[bold white]Duration Control Verification Tool[/bold white]
[dim]Test precise speech synthesis timing control[/dim]"""


def create_header() -> Panel:
    """Create the application header."""
    content = Align.center(Text.from_markup(LOGO + "\n" + HEADER))
    return Panel(
        content,
        box=box.DOUBLE_EDGE,
        border_style=BRAND_COLOR,
        padding=(1, 2),
    )


def create_stat_card(label: str, value: str, style: str = "white") -> Panel:
    """Create a statistic display card."""
    content = Align.center(
        Text.from_markup(f"[dim]{label}[/dim]\n[bold {style}]{value}[/bold {style}]")
    )
    return Panel(content, box=box.ROUNDED, border_style=MUTED_COLOR, padding=(0, 2))


def format_error(error: float, error_pct: float) -> tuple[str, str]:
    """Format error values with appropriate colors."""
    if abs(error_pct) < 5:
        style = SUCCESS_COLOR
    elif abs(error_pct) < 15:
        style = WARNING_COLOR
    else:
        style = ERROR_COLOR
    
    error_str = f"[{style}]{error:+.2f}s[/{style}]"
    pct_str = f"[{style}]{error_pct:+.1f}%[/{style}]"
    return error_str, pct_str


def get_status_badge(error_pct: float) -> str:
    """Get a status badge based on error percentage."""
    if abs(error_pct) < 5:
        return f"[bold {SUCCESS_COLOR}]● EXCELLENT[/bold {SUCCESS_COLOR}]"
    elif abs(error_pct) < 10:
        return f"[bold {SUCCESS_COLOR}]✓ PASS[/bold {SUCCESS_COLOR}]"
    elif abs(error_pct) < 15:
        return f"[bold {WARNING_COLOR}]◐ ACCEPTABLE[/bold {WARNING_COLOR}]"
    else:
        return f"[bold {ERROR_COLOR}]✗ FAIL[/bold {ERROR_COLOR}]"


# ═══════════════════════════════════════════════════════════════════════════════
# AUDIO UTILITIES
# ═══════════════════════════════════════════════════════════════════════════════

def get_audio_duration(filepath: str) -> float:
    """Get duration of audio file in seconds."""
    import torchaudio
    waveform, sample_rate = torchaudio.load(filepath)
    return waveform.shape[1] / sample_rate


def format_duration(seconds: float) -> str:
    """Format duration as MM:SS.ms"""
    mins = int(seconds // 60)
    secs = seconds % 60
    if mins > 0:
        return f"{mins}:{secs:05.2f}"
    return f"{secs:.2f}s"


# ═══════════════════════════════════════════════════════════════════════════════
# TEST RUNNER
# ═══════════════════════════════════════════════════════════════════════════════

def run_duration_test(
    tts,
    prompt_audio: str,
    text: str,
    target_duration: float,
    output_dir: str = "outputs/duration_tests",
    progress_callback=None,
) -> dict:
    """Run a single duration control test."""
    os.makedirs(output_dir, exist_ok=True)
    timestamp = int(time.time())
    output_path = os.path.join(output_dir, f"duration_{target_duration:.1f}s_{timestamp}.wav")
    
    start_time = time.perf_counter()
    tts.infer(
        spk_audio_prompt=prompt_audio,
        text=text,
        output_path=output_path,
        target_duration=target_duration,
        verbose=False,
    )
    inference_time = time.perf_counter() - start_time
    
    actual_duration = get_audio_duration(output_path)
    error = actual_duration - target_duration
    error_pct = (error / target_duration) * 100 if target_duration > 0 else 0
    
    return {
        "target": target_duration,
        "actual": actual_duration,
        "error": error,
        "error_pct": error_pct,
        "inference_time": inference_time,
        "output_path": output_path,
        "rtf": inference_time / actual_duration if actual_duration > 0 else 0,
    }


def display_results(results: list[dict], text: str) -> None:
    """Display test results in a beautiful format."""
    console.print()
    console.print(Rule("[bold]Test Results[/bold]", style=BRAND_COLOR))
    console.print()
    
    # Summary cards
    avg_error = sum(abs(r["error_pct"]) for r in results) / len(results)
    avg_rtf = sum(r["rtf"] for r in results) / len(results)
    total_time = sum(r["inference_time"] for r in results)
    
    cards = Columns([
        create_stat_card("Tests Run", str(len(results)), BRAND_COLOR),
        create_stat_card("Avg Error", f"{avg_error:.1f}%", 
                        SUCCESS_COLOR if avg_error < 10 else WARNING_COLOR),
        create_stat_card("Avg RTF", f"{avg_rtf:.2f}x", MUTED_COLOR),
        create_stat_card("Total Time", format_duration(total_time), MUTED_COLOR),
    ], equal=True, expand=True)
    console.print(cards)
    console.print()
    
    # Results table
    table = Table(
        box=box.ROUNDED,
        header_style=f"bold {BRAND_COLOR}",
        border_style=MUTED_COLOR,
        show_lines=True,
        expand=True,
    )
    
    table.add_column("Target", justify="center", style="bold")
    table.add_column("Actual", justify="center")
    table.add_column("Error", justify="center")
    table.add_column("Error %", justify="center")
    table.add_column("RTF", justify="center", style=MUTED_COLOR)
    table.add_column("Status", justify="center")
    
    for r in results:
        error_str, pct_str = format_error(r["error"], r["error_pct"])
        table.add_row(
            f"[bold]{r['target']:.1f}s[/bold]",
            f"{r['actual']:.2f}s",
            error_str,
            pct_str,
            f"{r['rtf']:.2f}x",
            get_status_badge(r["error_pct"]),
        )
    
    console.print(table)
    console.print()
    
    # Text preview
    text_preview = text[:80] + "..." if len(text) > 80 else text
    console.print(Panel(
        f"[dim]Test text:[/dim] [italic]{text_preview}[/italic]",
        box=box.ROUNDED,
        border_style=MUTED_COLOR,
    ))
    console.print()
    
    # Overall assessment
    if avg_error < 5:
        assessment = Panel(
            Align.center(Text.from_markup(
                f"[bold {SUCCESS_COLOR}]🎯 Excellent Precision![/bold {SUCCESS_COLOR}]\n"
                f"[dim]Duration control is working with high accuracy.\n"
                f"Average error: {avg_error:.1f}%[/dim]"
            )),
            box=box.DOUBLE,
            border_style=SUCCESS_COLOR,
            padding=(1, 4),
        )
    elif avg_error < 10:
        assessment = Panel(
            Align.center(Text.from_markup(
                f"[bold {SUCCESS_COLOR}]✓ Good Performance[/bold {SUCCESS_COLOR}]\n"
                f"[dim]Duration control is within acceptable tolerance.\n"
                f"Average error: {avg_error:.1f}%[/dim]"
            )),
            box=box.DOUBLE,
            border_style=SUCCESS_COLOR,
            padding=(1, 4),
        )
    elif avg_error < 15:
        assessment = Panel(
            Align.center(Text.from_markup(
                f"[bold {WARNING_COLOR}]◐ Needs Calibration[/bold {WARNING_COLOR}]\n"
                f"[dim]Consider adjusting TOKENS_PER_SECOND rate.\n"
                f"Average error: {avg_error:.1f}%[/dim]"
            )),
            box=box.DOUBLE,
            border_style=WARNING_COLOR,
            padding=(1, 4),
        )
    else:
        assessment = Panel(
            Align.center(Text.from_markup(
                f"[bold {ERROR_COLOR}]✗ High Variance[/bold {ERROR_COLOR}]\n"
                f"[dim]Duration control may need investigation.\n"
                f"Average error: {avg_error:.1f}%[/dim]"
            )),
            box=box.DOUBLE,
            border_style=ERROR_COLOR,
            padding=(1, 4),
        )
    
    console.print(assessment)
    console.print()
    
    # Output location
    output_dir = os.path.dirname(results[0]["output_path"])
    console.print(f"[dim]📁 Output files saved to: [underline]{output_dir}[/underline][/dim]")
    console.print()


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="IndexTTS2 Duration Control Verification Tool",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--prompt", "-p",
        type=str,
        default="examples/voice_01.wav",
        help="Speaker prompt audio file",
    )
    parser.add_argument(
        "--text", "-t",
        type=str,
        default="Welcome to the duration control test. This sentence will be spoken at the pace you specify, demonstrating precise timing control.",
        help="Text to synthesize",
    )
    parser.add_argument(
        "--duration", "-d",
        type=float,
        nargs="+",
        default=[3.0, 5.0, 8.0],
        help="Target duration(s) in seconds",
    )
    parser.add_argument(
        "--model-dir",
        type=str,
        default="checkpoints",
        help="Model checkpoints directory",
    )
    parser.add_argument(
        "--fp16",
        action="store_true",
        help="Use FP16 inference (faster, less VRAM)",
    )
    parser.add_argument(
        "--quiet", "-q",
        action="store_true",
        help="Suppress model loading output",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug logging to debug.log file",
    )
    
    args = parser.parse_args()
    
    # Configure debug logging if requested
    if args.debug:
        logging.basicConfig(
            level=logging.DEBUG,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('debug.log', mode='w')
            ]
        )
        console.print(f"[{WARNING_COLOR}]Debug logging enabled - writing to debug.log[/{WARNING_COLOR}]")
    
    # Clear screen and show header
    console.clear()
    console.print(create_header())
    console.print()
    
    # Validate inputs
    if not os.path.exists(args.prompt):
        console.print(Panel(
            f"[bold {ERROR_COLOR}]Error:[/bold {ERROR_COLOR}] Prompt audio not found\n"
            f"[dim]Path: {args.prompt}[/dim]",
            border_style=ERROR_COLOR,
        ))
        sys.exit(1)
    
    if not os.path.exists(args.model_dir):
        console.print(Panel(
            f"[bold {ERROR_COLOR}]Error:[/bold {ERROR_COLOR}] Model directory not found\n"
            f"[dim]Path: {args.model_dir}[/dim]\n\n"
            f"[{WARNING_COLOR}]Download models with:[/{WARNING_COLOR}]\n"
            f"[dim]hf download IndexTeam/IndexTTS-2 --local-dir=checkpoints[/dim]",
            border_style=ERROR_COLOR,
        ))
        sys.exit(1)
    
    # Configuration display
    config_table = Table(box=box.SIMPLE, show_header=False, border_style=MUTED_COLOR)
    config_table.add_column("Key", style="dim")
    config_table.add_column("Value")
    config_table.add_row("Prompt", os.path.basename(args.prompt))
    config_table.add_row("Durations", ", ".join(f"{d}s" for d in args.duration))
    config_table.add_row("Precision", "FP16" if args.fp16 else "FP32")
    config_table.add_row("Text", args.text[:50] + "..." if len(args.text) > 50 else args.text)
    
    console.print(Panel(config_table, title="[bold]Configuration[/bold]", border_style=MUTED_COLOR))
    console.print()
    
    # Load model
    console.print(Rule("[bold]Loading Model[/bold]", style=BRAND_COLOR))
    console.print()
    
    with Progress(
        SpinnerColumn(style=BRAND_COLOR),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(bar_width=40, style=BRAND_COLOR, complete_style=SUCCESS_COLOR),
        TaskProgressColumn(),
        TimeElapsedColumn(),
        console=console,
        transient=not args.quiet,
    ) as progress:
        task = progress.add_task("[cyan]Initializing IndexTTS2...", total=100)
        
        # Redirect stdout to suppress model loading messages if quiet
        if args.quiet:
            import io
            old_stdout = sys.stdout
            sys.stdout = io.StringIO()
        
        try:
            progress.update(task, advance=10, description="[cyan]Loading configuration...")
            from indextts.infer_v2 import IndexTTS2
            
            progress.update(task, advance=20, description="[cyan]Loading GPT model...")
            tts = IndexTTS2(
                model_dir=args.model_dir,
                cfg_path=os.path.join(args.model_dir, "config.yaml"),
                use_fp16=args.fp16,
            )
            progress.update(task, advance=70, description="[cyan]Ready!")
        finally:
            if args.quiet:
                sys.stdout = old_stdout
    
    console.print(f"[{SUCCESS_COLOR}]✓[/{SUCCESS_COLOR}] Model loaded successfully")
    console.print()
    
    # Run tests
    console.print(Rule("[bold]Running Tests[/bold]", style=BRAND_COLOR))
    console.print()
    
    results = []
    with Progress(
        SpinnerColumn(style=BRAND_COLOR),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(bar_width=40, style=BRAND_COLOR, complete_style=SUCCESS_COLOR),
        TaskProgressColumn(),
        TimeElapsedColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("Running tests...", total=len(args.duration))
        
        for i, duration in enumerate(args.duration):
            progress.update(
                task,
                description=f"[cyan]Generating {duration}s audio ({i+1}/{len(args.duration)})..."
            )
            result = run_duration_test(tts, args.prompt, args.text, duration)
            results.append(result)
            progress.advance(task)
        
        progress.update(task, description=f"[{SUCCESS_COLOR}]Complete![/{SUCCESS_COLOR}]")
    
    console.print()
    
    # Display results
    display_results(results, args.text)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        console.print(f"\n[{WARNING_COLOR}]Interrupted by user[/{WARNING_COLOR}]")
        sys.exit(130)
    except Exception as e:
        console.print(Panel(
            f"[bold {ERROR_COLOR}]Error:[/bold {ERROR_COLOR}] {e}",
            border_style=ERROR_COLOR,
        ))
        raise
