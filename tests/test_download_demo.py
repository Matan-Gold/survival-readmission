"""
Unit tests for download_demo module.
"""

import hashlib
import pytest
from pathlib import Path
from app.download_demo import compute_sha256, parse_checksum_file, verify_checksum


def test_compute_sha256(tmp_path):
    """Test SHA-256 hash computation."""
    # Create a test file
    test_file = tmp_path / "test.txt"
    test_content = b"Hello, MIMIC-IV!"
    test_file.write_bytes(test_content)
    
    # Compute expected hash
    expected_hash = hashlib.sha256(test_content).hexdigest()
    
    # Test function
    actual_hash = compute_sha256(test_file)
    assert actual_hash == expected_hash


def test_parse_checksum_file():
    """Test parsing of SHA256SUMS.txt format."""
    checksum_content = """
abc123def456  hosp/admissions.csv.gz
789ghi012jkl  hosp/patients.csv.gz
mno345pqr678  hosp/diagnoses_icd.csv.gz
    """
    
    checksums = parse_checksum_file(checksum_content)
    
    assert len(checksums) == 3
    assert checksums["hosp/admissions.csv.gz"] == "abc123def456"
    assert checksums["hosp/patients.csv.gz"] == "789ghi012jkl"
    assert checksums["hosp/diagnoses_icd.csv.gz"] == "mno345pqr678"


def test_parse_checksum_file_empty():
    """Test parsing empty checksum file."""
    checksums = parse_checksum_file("")
    assert len(checksums) == 0


def test_verify_checksum(tmp_path):
    """Test checksum verification."""
    # Create a test file
    test_file = tmp_path / "test.txt"
    test_content = b"Test data"
    test_file.write_bytes(test_content)
    
    # Compute correct hash
    correct_hash = hashlib.sha256(test_content).hexdigest()
    incorrect_hash = "0" * 64
    
    # Test verification
    assert verify_checksum(test_file, correct_hash) is True
    assert verify_checksum(test_file, incorrect_hash) is False


def test_parse_checksum_file_with_spaces():
    """Test parsing checksum file with various spacing."""
    checksum_content = """
abc123  hosp/admissions.csv.gz
def456    hosp/patients.csv.gz
    """
    
    checksums = parse_checksum_file(checksum_content)
    
    assert len(checksums) == 2
    assert "hosp/admissions.csv.gz" in checksums
    assert "hosp/patients.csv.gz" in checksums

