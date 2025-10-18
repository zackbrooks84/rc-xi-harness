# harness/protocols/identity.py
from __future__ import annotations
from typing import List

def identity_texts(texts: List[str]) -> List[str]:
    """
    Pass-through: return the transcript unchanged.
    """
    return list(texts)
