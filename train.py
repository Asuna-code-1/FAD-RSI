"""Delegate to the original project's audited entry point. Default is configuration-only."""
import argparse
import os
from pathlib import Path
import sys

project = Path(__file__).resolve().parent / "ATI-OPD"
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--pair", required=True, choices=("deepseek", "nemotron"))
parser.add_argument("--mode", required=True, choices=("opd", "ati-opd", "fad-opd", "fixed", "fad-rsi"))
parser.add_argument("--train", action="store_true")
args = parser.parse_args()
config = project/"local/fad_configs/comparison-v1"/f"{args.pair}-{args.mode}.json"
command = [sys.executable, str(project/"local/run_four_gpu_entry.py"), "--config", str(config)]
if not args.train:
    command.extend(("--check-only", "--no-write-resolved"))
os.chdir(project)
os.execv(sys.executable, command)
