#!/usr/bin/env python3
"""
HARP — Genie Bundle Downloader/Preparer

Downloads precompiled Genie bundles from Qualcomm AI Hub Models (if authenticated)
or prepares skeleton structure for manual placement.

Usage:
    python scripts/prepare_genie_bundles.py --list          # Show available bundles
    python scripts/prepare_genie_bundles.py --download-all  # Download all (needs auth)
    python scripts/prepare_genie_bundles.py --skeleton      # Create empty dirs for manual placement
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

BUILD_DIR = Path(__file__).resolve().parent.parent / "build"

# Known bundles from qualcomm/ai-hub-models (v0.56.0+)
BUNDLES = {
    "whisper-base-w4a16": {
        "model_id": "whisper-base",
        "modality": "audio",
        "ai_hub_slug": "whisper_base_quantized",
        "size_gb": 0.15,
    },
    "moondream-vl-w4a16": {
        "model_id": "vision-specialist",
        "modality": "vision",
        "ai_hub_slug": "moondream_quantized",
        "size_gb": 1.8,
    },
    "minicpm-v-w4a16": {
        "model_id": "vision-specialist",
        "modality": "vision",
        "ai_hub_slug": "minicpm_v_quantized",
        "size_gb": 2.1,
    },
}

def create_skeleton():
    """Create empty bundle directories with README for manual placement."""
    for name, info in BUNDLES.items():
        bundle_dir = BUILD_DIR / name
        bundle_dir.mkdir(parents=True, exist_ok=True)
        
        # Create metadata.json
        metadata = {
            "model_id": info["model_id"],
            "model_name": name.replace("-w4a16", "").replace("-", " ").title(),
            "precision": "w4a16",
            "runtime": "genie",
            "qairt_version": "2.45.0",
            "htp_arch": "v73",
            "modalities": [info["modality"]],
        }
        (bundle_dir / "metadata.json").write_text(json.dumps(metadata, indent=2))
        
        # Create placeholder files
        (bundle_dir / "genie_config.json").write_text('{"dialog": {"context": {"size": 4096}}}\n')
        (bundle_dir / "README.txt").write_text(
            f"Place Genie bundle files here:\n"
            f"  - genie_config.json (full config)\n"
            f"  - *.bin (context binaries)\n"
            f"  - tokenizer/ (tokenizer files)\n"
            f"\nDownload from: qualcomm/ai-hub-models/{info['ai_hub_slug']}\n"
            f"Expected model_id: {info['model_id']}\n"
            f"Modality: {info['modality']}\n"
            f"Size: ~{info['size_gb']} GB\n"
        )
        print(f"Created skeleton: {bundle_dir}")

def download_bundle(name: str, info: dict) -> bool:
    """Attempt to download bundle via qai_hub_models export (requires auth)."""
    bundle_dir = BUILD_DIR / name
    bundle_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        # This requires qai_hub_models installed and QAI_HUB_API_KEY set
        result = subprocess.run([
            sys.executable, "-m", f"qai_hub_models.models.{info['ai_hub_slug']}.export",
            "--output-dir", str(bundle_dir)
        ], capture_output=True, text=True, timeout=300)
        
        if result.returncode == 0:
            print(f"✓ Downloaded {name}")
            return True
        else:
            print(f"✗ Failed to download {name}: {result.stderr[:200]}")
            return False
    except FileNotFoundError:
        print(f"✗ qai_hub_models not installed. Install with: pip install qai-hub-models")
        return False
    except subprocess.TimeoutExpired:
        print(f"✗ Timeout downloading {name}")
        return False

def list_bundles():
    """List available bundles and their status."""
    print("Available Genie Bundles:")
    print("=" * 60)
    for name, info in BUNDLES.items():
        bundle_dir = BUILD_DIR / name
        exists = bundle_dir.exists() and (bundle_dir / "genie_config.json").exists()
        has_bins = any(bundle_dir.glob("*.bin"))
        
        status = "✓ READY" if (exists and has_bins) else ("⚠ partial" if exists else "✗ missing")
        
        print(f"  {name}")
        print(f"    model_id: {info['model_id']}")
        print(f"    modality: {info['modality']}")
        print(f"    size: ~{info['size_gb']} GB")
        print(f"    status: {status}")
        print()

def main():
    parser = argparse.ArgumentParser(description="Prepare HARP Genie bundles")
    parser.add_argument("--list", action="store_true", help="List available bundles")
    parser.add_argument("--download-all", action="store_true", help="Download all bundles (requires qai_hub_models + auth)")
    parser.add_argument("--download", help="Download specific bundle by name")
    parser.add_argument("--skeleton", action="store_true", help="Create empty dirs with metadata for manual placement")
    args = parser.parse_args()
    
    BUILD_DIR.mkdir(parents=True, exist_ok=True)
    
    if args.list:
        list_bundles()
    elif args.skeleton:
        create_skeleton()
    elif args.download_all:
        print("Downloading all bundles (requires qai_hub_models + QAI_HUB_API_KEY)...")
        for name, info in BUNDLES.items():
            download_bundle(name, info)
    elif args.download:
        if args.download in BUNDLES:
            download_bundle(args.download, BUNDLES[args.download])
        else:
            print(f"Unknown bundle: {args.download}")
            sys.exit(1)
    else:
        parser.print_help()

if __name__ == "__main__":
    main()