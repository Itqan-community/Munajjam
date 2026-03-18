"""
Command-line interface for Munajjam.

Usage:
    munajjam align <audio_file> [--surah <number>] [--strategy <name>] [--output <file>] [--format <fmt>] [--whisper-backend <backend>]
    munajjam batch <directory> [--pattern <glob>] [--output-dir <dir>] [--format <fmt>] [--whisper-backend <backend>]
    munajjam --version
    munajjam --help
"""

import argparse
import io
import json
import re
import sys
from pathlib import Path

from munajjam import __version__
from munajjam.transcription.whisperFactory import WhisperBackend, WhisperFactory
from munajjam.core.arabic import infer_surah_number 
# Valid surah range
MIN_SURAH = 1
MAX_SURAH = 114


def create_parser() -> argparse.ArgumentParser:
    """Create the argument parser for the munajjam CLI."""
    parser = argparse.ArgumentParser(
        prog="munajjam",
        description="Munajjam — Synchronize Quran ayat with audio recitations.",
        epilog="For more information, visit: https://github.com/Itqan-community/Munajjam",
    ) 
 

    parser.add_argument(
        "--version",
        action="version",
        version=f"munajjam {__version__}",
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # --- align subcommand ---
    align_parser = subparsers.add_parser(
        "align",
        help="Align a single audio file to Quran ayahs",
        description="Transcribe and align a single audio file to Quran ayahs.",
    )
    align_parser.add_argument(
        "audio_file",
        type=str,
        help="Path to the audio file (e.g., 001.mp3)",
    )
    align_parser.add_argument(
        "--surah",
        type=int,
        default=None,
        help=f"Surah number ({MIN_SURAH}-{MAX_SURAH}). If not provided, inferred from filename.",
    )
    align_parser.add_argument(
        "--strategy",
        type=str,
        choices=["auto", "greedy", "dp", "hybrid"],
        default="auto",
        help="Alignment strategy to use (default: auto)",
    )
    align_parser.add_argument(
        "--output",
        "-o",
        type=str,
        default=None,
        help="Output file path. If not provided, prints to stdout.",
    )
    align_parser.add_argument(
        "--format",
        "-f",
        type=str,
        choices=["json", "text", "csv"],
        default="json",
        help="Output format (default: json)",
    )
    align_parser.add_argument(
            "--whisper-backend",
            type=str,
            choices=["fasterwhisper", "whisperx"],
            default="whisperx",
            help="Whisper backend to use (default: whisperx)",
        )
    # --- batch subcommand ---
    batch_parser = subparsers.add_parser(
        "batch",
        help="Batch process multiple audio files",
        description="Transcribe and align multiple audio files in a directory.",
    )
    batch_parser.add_argument(
        "directory",
        type=str,
        help="Directory containing audio files",
    )
    batch_parser.add_argument(
        "--pattern",
        type=str,
        default="*.mp3",
        help="Glob pattern for audio files (default: *.mp3)",
    )
    batch_parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory for results. If not provided, uses input directory.",
    )
    batch_parser.add_argument(
        "--format",
        "-f",
        type=str,
        choices=["json", "text", "csv"],
        default="json",
        help="Output format (default: json)",
    )
    batch_parser.add_argument(
        "--strategy",
        type=str,
        choices=["auto", "greedy", "dp", "hybrid"],
        default="auto",
        help="Alignment strategy to use (default: auto)",
    )
    batch_parser.add_argument(
            "--whisper-backend",
            type=str,   
            choices=["fasterwhisper", "whisperx"],
            default="whisperx",
            help="Whisper backend to use (default: whisperx)",
        )
    return parser


def _validate_surah_number(surah_num: int) -> None:
    """Validate that a surah number is within the valid range (1-114).

    Raises:
        ValueError: If surah number is out of range.
    """
    if not (MIN_SURAH <= surah_num <= MAX_SURAH):
        raise ValueError(
            f"Invalid surah number: {surah_num}. Must be between {MIN_SURAH} and {MAX_SURAH}."
        )







def _format_results(results: list, fmt: str) -> str:
    """Format alignment results to the specified format."""
    if fmt == "json":
        output = []
        for r in results:
            output.append(
                {
                    "ayah_number": r.ayah.ayah_number,
                    "start_time": round(r.start_time, 2),
                    "end_time": round(r.end_time, 2),
                    "text": r.ayah.text,
                }
            )
        return json.dumps(output, ensure_ascii=False, indent=2)
    elif fmt == "csv":
        lines = ["ayah_number,start_time,end_time,text"]
        for r in results:
            # Escape text for CSV (wrap in quotes if it contains commas)
            text = r.ayah.text.replace('"', '""')
            lines.append(f'{r.ayah.ayah_number},{r.start_time:.2f},{r.end_time:.2f},"{text}"')
        return "\n".join(lines)
    else:  # text
        lines = []
        for r in results:
            lines.append(f"Ayah {r.ayah.ayah_number}: {r.start_time:.2f}s - {r.end_time:.2f}s")
        return "\n".join(lines)


def _write_output(content: str, output_path: str | None) -> None:
    """Write content to file or stdout."""
    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        Path(output_path).write_text(content, encoding="utf-8")
        print(f"Results written to {output_path}", file=sys.stderr)
    else:
        try:
            print(content)
        except UnicodeEncodeError:
            # Fallback for Windows console with restricted encoding
            if hasattr(sys.stdout, 'buffer'):
                sys.stdout.buffer.write(content.encode('utf-8'))
                sys.stdout.buffer.write(b'\n')
            else:
                # Last resort: replace unencodable characters
                print(content.encode(sys.stdout.encoding, errors='replace').decode(sys.stdout.encoding))


def cmd_align(args: argparse.Namespace) -> int:
    """Execute the align command."""
    from munajjam.core import align
    from munajjam.data import load_surah_ayahs
    from munajjam.transcription import WhisperTranscriber

    audio_path = args.audio_file
    if not Path(audio_path).exists():
        print(f"Error: Audio file not found: {audio_path}", file=sys.stderr)
        return 1

    # Determine surah number
    surah_num = args.surah
    if surah_num is None:
        try:
            surah_num = infer_surah_number(audio_path)
        except ValueError as e:
            print(f"Error: {e}", file=sys.stderr)
            return 1

    # Validate surah number
    try:
        _validate_surah_number(surah_num)
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1

    print(f"Processing surah {surah_num} from {audio_path}...", file=sys.stderr)
    print(f"Strategy: {args.strategy}", file=sys.stderr)

    from munajjam.config import get_settings
    settings = get_settings()
    
    transcriber = WhisperFactory().create_whisper(
        backend=WhisperBackend(args.whisper_backend),
        model_name=settings.model_id,
        device=settings.device
    )
  
    segments = transcriber.transcribe(audio_path)

    # Align
    ayahs = load_surah_ayahs(surah_num)
    results = align(audio_path, segments, ayahs, strategy=args.strategy)

    # Format and output
    content = _format_results(results, args.format)
    _write_output(content, args.output)

    return 0


def cmd_batch(args: argparse.Namespace) -> int:
    """Execute the batch command."""
    from munajjam.core import align
    from munajjam.data import load_surah_ayahs
    from munajjam.transcription import WhisperTranscriber
    from munajjam.config import get_settings
    input_dir = Path(args.directory)
    if not input_dir.is_dir():
        print(f"Error: Directory not found: {args.directory}", file=sys.stderr)
        return 1

    audio_files = sorted(input_dir.glob(args.pattern))
    if not audio_files:
        print(
            f"Error: No files matching '{args.pattern}' in {args.directory}",
            file=sys.stderr,
        )
        return 1

    output_dir = Path(args.output_dir) if args.output_dir else input_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    settings = get_settings()
    print(f"Found {len(audio_files)} audio files to process.", file=sys.stderr)
    transcriber = WhisperFactory().create_whisper(backend=WhisperBackend(args.whisper_backend),model_name=settings.model_id,device=settings.devic )
  


    errors = 0

    for audio_file in audio_files:
        try:
            surah_num = infer_surah_number(str(audio_file))
            _validate_surah_number(surah_num)
            print(
                f"Processing surah {surah_num}: {audio_file.name}...",
                file=sys.stderr,
            )

            segments = transcriber.transcribe(str(audio_file))
            ayahs = load_surah_ayahs(surah_num)
            results = align(str(audio_file), segments, ayahs, strategy=args.strategy)

            # Determine output extension
            ext = {"json": ".json", "csv": ".csv", "text": ".txt"}[args.format]
            output_path = output_dir / f"{audio_file.stem}{ext}"

            content = _format_results(results, args.format)
            output_path.write_text(content, encoding="utf-8")
            print(f"  -> {output_path}", file=sys.stderr)

        except Exception as e:
            print(f"  Error processing {audio_file.name}: {e}", file=sys.stderr)
            errors += 1

    total = len(audio_files)
    print(f"\nBatch complete: {total - errors}/{total} succeeded.", file=sys.stderr)
    return 1 if errors > 0 else 0


def main(argv: list[str] | None = None) -> int:
    """Main entry point for the munajjam CLI."""
    parser = create_parser()
    args = parser.parse_args(argv)

    if args.command is None:
        parser.print_help()
        return 0

    if args.command == "align":
        return cmd_align(args)
    elif args.command == "batch":
        return cmd_batch(args)
    else:
        parser.print_help()
        return 0


def cli() -> None:
    """Entry point for the console_scripts."""
    # Ensure UTF-8 output based on platform and environment
    if sys.platform == "win32":
        try:
            # Python 3.7+ approach for reconfiguring standard streams
            if hasattr(sys.stdout, "reconfigure"):
                sys.stdout.reconfigure(encoding="utf-8")
                sys.stderr.reconfigure(encoding="utf-8")
            else:
                # Legacy or restricted environments
                import io
                sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
                sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8")
        except Exception:
            pass
            
    sys.exit(main())


if __name__ == "__main__":
    cli()
