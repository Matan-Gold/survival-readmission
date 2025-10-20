"""
Download MIMIC-IV Demo dataset from PhysioNet.

This script downloads the required MIMIC-IV Demo files from PhysioNet
and verifies their integrity using SHA-256 checksums.

Usage:
    python -m app.download_demo
    python -m app.download_demo --dest data/raw/mimic-iv-demo
"""

import argparse
import hashlib
import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple

try:
    import requests
except ImportError:
    print("Error: requests library is required. Install with: pip install requests")
    sys.exit(1)


# Base URL for MIMIC-IV Demo v2.2 (use /files/ for direct file access)
BASE_URL = "https://physionet.org/files/mimic-iv-demo/2.2"

# Files to download
FILES_TO_DOWNLOAD = [
    "hosp/admissions.csv.gz",
    "hosp/patients.csv.gz",
    "hosp/diagnoses_icd.csv.gz",
    "hosp/d_labitems.csv.gz",
    "hosp/labevents.csv.gz",
]

# Checksum file
CHECKSUM_FILE = "SHA256SUMS.txt"

# Console-safe symbols (avoid Unicode that breaks on some Windows consoles)
OK = "[OK]"
FAIL = "[FAIL]"
WARN = "[WARN]"


def compute_sha256(filepath: Path) -> str:
    """
    Compute SHA-256 hash of a file.
    
    Args:
        filepath: Path to the file.
        
    Returns:
        Hexadecimal SHA-256 hash string.
    """
    sha256_hash = hashlib.sha256()
    
    with open(filepath, "rb") as f:
        # Read in chunks to handle large files
        for byte_block in iter(lambda: f.read(4096), b""):
            sha256_hash.update(byte_block)
    
    return sha256_hash.hexdigest()


def parse_checksum_file(checksum_content: str) -> Dict[str, str]:
    """
    Parse SHA256SUMS.txt file content.
    
    Args:
        checksum_content: Content of the checksum file.
        
    Returns:
        Dictionary mapping file paths to their SHA-256 hashes.
    """
    checksums = {}
    
    for line in checksum_content.strip().split('\n'):
        if not line.strip():
            continue
        
        parts = line.split()
        if len(parts) >= 2:
            # Format: <hash>  <filepath>
            hash_value = parts[0]
            filepath = parts[1]
            checksums[filepath] = hash_value
    
    return checksums


def download_file(url: str, dest_path: Path, show_progress: bool = True) -> bool:
    """
    Download a file from URL to destination path.
    
    Args:
        url: URL to download from.
        dest_path: Destination file path.
        show_progress: Whether to show download progress.
        
    Returns:
        True if download successful, False otherwise.
    """
    try:
        response = requests.get(url, stream=True, timeout=30)
        response.raise_for_status()
        
        # Get total file size if available
        total_size = int(response.headers.get('content-length', 0))
        
        # Create parent directories if needed
        dest_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Download with progress
        downloaded = 0
        with open(dest_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
                    downloaded += len(chunk)
                    
                    if show_progress and total_size > 0:
                        progress = (downloaded / total_size) * 100
                        print(f"\r  Downloading: {progress:.1f}%", end='', flush=True)
        
        if show_progress and total_size > 0:
            print()  # New line after progress
        
        return True
        
    except requests.exceptions.RequestException as e:
        print(f"\n  Error downloading {url}: {e}")
        return False


def verify_checksum(filepath: Path, expected_hash: str) -> bool:
    """
    Verify file's SHA-256 checksum.
    
    Args:
        filepath: Path to file to verify.
        expected_hash: Expected SHA-256 hash.
        
    Returns:
        True if checksum matches, False otherwise.
    """
    actual_hash = compute_sha256(filepath)
    return actual_hash == expected_hash


def main():
    """Main download and verification workflow."""
    parser = argparse.ArgumentParser(
        description="Download MIMIC-IV Demo dataset from PhysioNet"
    )
    parser.add_argument(
        "--dest",
        type=str,
        default="data/raw/mimic-iv-demo",
        help="Destination directory (default: data/raw/mimic-iv-demo)"
    )
    args = parser.parse_args()
    
    dest_dir = Path(args.dest)
    
    print("=" * 70)
    print("MIMIC-IV Demo Dataset Downloader")
    print("=" * 70)
    print(f"Destination: {dest_dir.absolute()}")
    print()
    
    # Create destination directory
    dest_dir.mkdir(parents=True, exist_ok=True)
    
    # Step 1: Download checksum file
    print("[1/3] Downloading checksum file...")
    checksum_url = f"{BASE_URL}/{CHECKSUM_FILE}"
    checksum_path = dest_dir / CHECKSUM_FILE
    
    try:
        response = requests.get(checksum_url, timeout=30)
        response.raise_for_status()
        checksum_content = response.text
        
        # Save checksum file
        with open(checksum_path, 'w') as f:
            f.write(checksum_content)
        
        print(f"  {OK} Downloaded {CHECKSUM_FILE}")
    except requests.exceptions.RequestException as e:
        print(f"  {FAIL} Error downloading checksum file: {e}")
        sys.exit(1)
    
    # Parse checksums
    checksums = parse_checksum_file(checksum_content)
    print(f"  {OK} Parsed {len(checksums)} checksums")
    print()
    
    # Step 2: Download data files
    print("[2/3] Downloading data files...")
    download_results: List[Tuple[str, bool, str]] = []
    
    for file_path in FILES_TO_DOWNLOAD:
        file_url = f"{BASE_URL}/{file_path}"
        dest_file = dest_dir / file_path
        
        print(f"\n  {file_path}")
        
        # Check if file exists and has correct checksum
        if dest_file.exists():
            if file_path in checksums:
                print("    Checking existing file...", end='', flush=True)
                if verify_checksum(dest_file, checksums[file_path]):
                    print(f" {OK} Checksum valid, skipping download")
                    download_results.append((file_path, True, "cached"))
                    continue
                else:
                    print(f" {FAIL} Checksum mismatch, re-downloading")
            else:
                print("    No checksum available, re-downloading")
        
        # Download file
        if download_file(file_url, dest_file):
            download_results.append((file_path, True, "downloaded"))
            print(f"    {OK} Downloaded successfully")
        else:
            download_results.append((file_path, False, "failed"))
            print(f"    {FAIL} Download failed")
    
    print()
    
    # Step 3: Verify checksums
    print("[3/3] Verifying checksums...")
    verification_results: List[Tuple[str, bool]] = []
    
    for file_path in FILES_TO_DOWNLOAD:
        dest_file = dest_dir / file_path
        
        if not dest_file.exists():
            print(f"  {FAIL} {file_path} - File missing")
            verification_results.append((file_path, False))
            continue
        
        if file_path not in checksums:
            print(f"  {WARN} {file_path} - No checksum available")
            verification_results.append((file_path, True))  # Assume OK if no checksum
            continue
        
        if verify_checksum(dest_file, checksums[file_path]):
            print(f"  {OK} {file_path} - Checksum valid")
            verification_results.append((file_path, True))
        else:
            print(f"  {FAIL} {file_path} - Checksum mismatch!")
            verification_results.append((file_path, False))
    
    print()
    
    # Summary
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    
    total_files = len(FILES_TO_DOWNLOAD)
    downloaded = sum(1 for _, success, status in download_results 
                     if success and status == "downloaded")
    cached = sum(1 for _, success, status in download_results 
                 if success and status == "cached")
    failed_download = sum(1 for _, success, _ in download_results if not success)
    verified = sum(1 for _, success in verification_results if success)
    failed_verify = sum(1 for _, success in verification_results if not success)
    
    print(f"Total files:        {total_files}")
    print(f"Downloaded:         {downloaded}")
    print(f"Cached (reused):    {cached}")
    print(f"Failed downloads:   {failed_download}")
    print(f"Verified:           {verified}")
    print(f"Failed verification: {failed_verify}")
    print()
    
    if failed_download > 0 or failed_verify > 0:
        print(f"{WARN} Some files failed to download or verify. Please check errors above.")
        sys.exit(1)
    else:
        print(f"{OK} All files downloaded and verified successfully!")
        print()
        print(f"Data location: {dest_dir.absolute()}")
        print()
        print("You can now run preprocessing:")
        print("  python preprocess.py")
    
    print("=" * 70)


if __name__ == "__main__":
    main()

