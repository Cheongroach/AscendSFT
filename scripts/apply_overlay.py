#!/usr/bin/env python3
"""将公开补丁应用到已检出的固定 LLaMA-Factory 版本。"""

import argparse
import hashlib
import json
import shutil
import subprocess
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("upstream", type=Path)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    target = args.upstream.resolve()
    manifest = json.loads((root / "upstream/source.json").read_text())
    commit = subprocess.check_output(["git", "-C", str(target), "rev-parse", "HEAD"], text=True).strip()
    if commit != manifest["commit"]:
        parser.error(f"Expected upstream commit {manifest['commit']}, found {commit}")
    trainer = target / manifest["trainer_path"]
    if hashlib.sha256(trainer.read_bytes()).hexdigest() != manifest["trainer_sha256"]:
        parser.error("Upstream trainer differs from the public source; refusing to overwrite it")
    relative = Path("src/llamafactory/model/model_utils/ascend_checkpointing.py")
    destination = target / relative
    if destination.exists():
        parser.error(f"Destination already exists: {relative}")
    patch = root / "patches/length-aware-checkpointing.patch"
    subprocess.run(["git", "-C", str(target), "apply", "--check", str(patch)], check=True)
    subprocess.run(["git", "-C", str(target), "apply", str(patch)], check=True)
    shutil.copyfile(root / "overlay" / relative, destination)
    print("Applied AscendSFT's experimental LLaMA-Factory overlay.")


if __name__ == "__main__":
    main()
