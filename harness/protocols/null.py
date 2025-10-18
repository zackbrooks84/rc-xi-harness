# harness/protocols/null.py
from __future__ import annotations
from typing import List

# Deterministic drift catalog spanning unrelated topics
_DRIFT = [
    "satellite weather map: cumulonimbus over the gulf",
    "recipe step: whisk 3 eggs with 90g sugar until ribbons form",
    "sports update: left wing scores in the 88th minute",
    "history note: the treaty was ratified in 1783 after months of debate",
    "code snippet: for i in range(10): print(i*i)",
    "finance ticker: XYZ up 1.2% on volume",
    "geometry fact: area of a circle is pi r squared",
    "music cue: key change from A minor to C major",
    "astronomy: M31 visible as a faint smudge under dark skies",
    "poetry fragment: the quiet river remembers our names",
    "medical: systolic pressure recorded at 118 mmHg",
    "chemistry: NaCl dissociates fully in aqueous solution",
    "cartography: isopleths indicate gradient strength",
    "linguistics: ablative case marks movement away",
    "security note: rotate keys and revoke stale tokens",
]

def topic_drift_texts(texts: List[str], stride: int = 3) -> List[str]:
    """
    Replace every `stride`-th line with a drift line to break temporal recursion.
    Length is preserved; content becomes deliberately heterogeneous.
    """
    out = list(texts)
    if not out:
        return out
    j = 0
    for i in range(len(out)):
        if (i + 1) % max(1, stride) == 0:
            out[i] = _DRIFT[j % len(_DRIFT)]
            j += 1
    return out
