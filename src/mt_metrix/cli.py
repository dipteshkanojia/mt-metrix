"""Command-line entry point for mt-metrix.

Subcommands::

    mt-metrix score --config <run.yaml> [--override key=value ...]
    mt-metrix submit --config <run.yaml> [sbatch overrides...]
    mt-metrix list-models [--family comet|tower|sacrebleu]
    mt-metrix list-datasets
    mt-metrix correlate --run <run_id_or_path>
    mt-metrix download --family comet --to <path>

``submit`` is a thin Python wrapper around ``scripts/submit.sh`` — the
canonical pre-flight-checked submit path on AISURREY. Any flags after the
config are forwarded verbatim to ``sbatch`` inside the wrapper.
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from mt_metrix import __version__
from mt_metrix.config import load_run_config
from mt_metrix.logging_utils import setup_logging
from mt_metrix.runner import run as run_pipeline

log = logging.getLogger(__name__)


def _cmd_score(args: argparse.Namespace) -> int:
    setup_logging(level=args.log_level)
    cfg = load_run_config(args.config, overrides=args.override or [])
    if args.output_root:
        cfg.output.root = args.output_root
    out_dir = run_pipeline(cfg)
    print(f"✓ run complete → {out_dir}")
    return 0


def _cmd_submit(args: argparse.Namespace) -> int:
    """Submit via scripts/submit.sh (pre-flight + sbatch --test-only + exclude aisurrey26)."""
    from mt_metrix.submit.slurm import submit_via_wrapper

    setup_logging(level=args.log_level)
    # Config is validated inside submit.sh; we don't need to load it here,
    # but validate existence up-front for a better error message.
    cfg_path = Path(args.config).resolve()
    if not cfg_path.is_file():
        print(f"config not found: {cfg_path}", file=sys.stderr)
        return 1

    sbatch_args: list[str] = list(args.sbatch_args or [])
    if args.partition:
        sbatch_args.extend(["-p", args.partition])
    if args.gpus:
        sbatch_args.extend([f"--gres=gpu:{args.gpus}"])
    if args.time:
        sbatch_args.extend([f"--time={args.time}"])

    returncode, job_id = submit_via_wrapper(
        cfg_path, sbatch_args=sbatch_args, dry_run=args.dry_run
    )
    if args.dry_run:
        print("[dry-run] pre-flight passed, not submitted" if returncode == 0
              else f"[dry-run] pre-flight FAILED (rc={returncode})")
    else:
        print(f"sbatch return code: {returncode}; job id: {job_id}")
    return returncode


def _cmd_list_models(args: argparse.Namespace) -> int:
    setup_logging(level=args.log_level)
    from mt_metrix.config import _find_catalogues

    roots = [Path.cwd(), Path(__file__).resolve().parents[2]]
    cats = _find_catalogues(roots)
    rows = sorted(cats.items())
    if args.family:
        rows = [(k, v) for k, v in rows if v["family"] == args.family]
    if not rows:
        print("(no models registered — check configs/models/*.yaml)")
        return 1
    col_w = max(len(k) for k, _ in rows) + 2
    print(f"{'ref'.ljust(col_w)}  model_id")
    print(f"{'-' * col_w}  {'-' * 40}")
    for key, entry in rows:
        print(f"{key.ljust(col_w)}  {entry.get('model', '-')}")
    return 0


def _cmd_list_datasets(args: argparse.Namespace) -> int:
    setup_logging(level=args.log_level)
    configs_dir = Path.cwd() / "configs" / "datasets"
    if not configs_dir.is_dir():
        print(f"(no dataset configs at {configs_dir})")
        return 1
    for path in sorted(configs_dir.glob("*.yaml")):
        print(f"{path.stem}  ({path})")
    return 0


def _cmd_correlate(args: argparse.Namespace) -> int:
    setup_logging(level=args.log_level)
    import json

    import pandas as pd

    from mt_metrix.io.writers import _correlations

    run_path = Path(args.run).resolve()
    tsv_path = run_path if run_path.suffix == ".tsv" else run_path / "segments.tsv"
    if not tsv_path.exists():
        print(f"segments.tsv not found at {tsv_path}", file=sys.stderr)
        return 1
    df = pd.read_csv(tsv_path, sep="\t")
    gold_col = "gold"
    if gold_col not in df.columns or df[gold_col].isna().all():
        print("no gold column in segments.tsv — nothing to correlate against", file=sys.stderr)
        return 1
    gold = pd.to_numeric(df[gold_col], errors="coerce")
    metric_cols = [
        c for c in df.columns
        if c not in {"segment_id", "lang_pair", "domain", "source", "target", "reference", "gold"}
    ]
    out: dict[str, object] = {}
    for col in metric_cols:
        pred = pd.to_numeric(df[col], errors="coerce")
        mask = pred.notna() & gold.notna()
        if mask.sum() < 2:
            out[col] = None
            continue
        out[col] = _correlations(pred[mask].values, gold[mask].values)
    print(json.dumps(out, indent=2))
    return 0


def _cmd_download(args: argparse.Namespace) -> int:
    setup_logging(level=args.log_level)
    from huggingface_hub import snapshot_download

    from mt_metrix.config import _find_catalogues

    roots = [Path.cwd(), Path(__file__).resolve().parents[2]]
    cats = _find_catalogues(roots)
    to_download = [
        (k, v) for k, v in cats.items()
        if (not args.family or v["family"] == args.family) and v.get("model")
    ]
    if args.ref:
        to_download = [(k, v) for k, v in to_download if k in set(args.ref)]
    if not to_download:
        print("nothing to download", file=sys.stderr)
        return 1
    out_root = Path(args.to).expanduser().resolve()
    out_root.mkdir(parents=True, exist_ok=True)
    print(f"downloading {len(to_download)} model(s) into {out_root}")
    for key, entry in to_download:
        repo = entry["model"]
        target = out_root / repo.replace("/", "__")
        print(f"- {key}  ({repo})  →  {target}")
        snapshot_download(repo_id=repo, local_dir=str(target))
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="mt-metrix",
        description="Comprehensive MT evaluation suite (COMET, Tower, reference metrics).",
    )
    p.add_argument("--version", action="version", version=f"mt-metrix {__version__}")
    p.add_argument("--log-level", default="INFO", help="DEBUG|INFO|WARNING|ERROR")
    sub = p.add_subparsers(dest="cmd", required=True)

    # score
    score = sub.add_parser("score", help="run a scoring job locally")
    score.add_argument("--config", required=True, help="path to a run YAML")
    score.add_argument("--output-root", default=None, help="override output.root")
    score.add_argument("--override", action="append", help="key=value override (repeatable)")
    score.set_defaults(func=_cmd_score)

    # submit — delegates to scripts/submit.sh (pre-flight + sbatch)
    submit = sub.add_parser(
        "submit",
        help="submit via scripts/submit.sh (pre-flight + sbatch on AISURREY)",
    )
    submit.add_argument("--config", required=True, help="path to a run YAML")
    submit.add_argument("--partition", default=None,
                        help="partition (a100|rtx_a6000_risk|l40s_risk|3090|...). "
                             "NOT 'gpu' — that doesn't exist on AISURREY.")
    submit.add_argument("--gpus", type=int, default=None,
                        help="number of GPUs (soft cap: 4). Forwards to --gres=gpu:N.")
    submit.add_argument("--time", default=None, help="SLURM --time (HH:MM:SS)")
    submit.add_argument("--dry-run", action="store_true",
                        help="run pre-flight checks only, don't submit")
    submit.add_argument("sbatch_args", nargs="*",
                        help="extra args forwarded to sbatch (e.g. --mem=128G)")
    submit.set_defaults(func=_cmd_submit)

    # list-models
    lm = sub.add_parser("list-models", help="list known scorer refs")
    lm.add_argument("--family", default=None, choices=["comet", "tower", "sacrebleu"])
    lm.set_defaults(func=_cmd_list_models)

    # list-datasets
    ld = sub.add_parser("list-datasets", help="list dataset configs under configs/datasets/")
    ld.set_defaults(func=_cmd_list_datasets)

    # correlate
    corr = sub.add_parser("correlate", help="recompute correlations from an existing segments.tsv")
    corr.add_argument("--run", required=True, help="path to outputs/<run_id>/ or segments.tsv")
    corr.set_defaults(func=_cmd_correlate)

    # download
    dl = sub.add_parser("download", help="prefetch scorer model weights")
    dl.add_argument("--family", default=None, choices=["comet", "tower"])
    dl.add_argument("--ref", action="append", help="specific scorer ref (repeatable)")
    dl.add_argument("--to", required=True, help="target directory for snapshots")
    dl.set_defaults(func=_cmd_download)

    return p


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
