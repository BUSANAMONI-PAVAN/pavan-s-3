import pytest

def test_import_rdkit():
    try:
        from rdkit import Chem
    except ImportError:
        pytest.fail("Failed to import RDKit")

def test_import_torch():
    try:
        import torch
    except ImportError:
        pytest.fail("Failed to import PyTorch")