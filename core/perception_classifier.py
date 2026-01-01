from __future__ import annotations

import json
import math
from dataclasses import dataclass
from typing import List, Optional, Tuple


@dataclass(frozen=True)
class Classification:
    label: str
    confidence: float


def _euclidean(a: List[float], b: List[float]) -> float:
    if len(a) != len(b):
        return float("inf")
    s = 0.0
    for x, y in zip(a, b):
        d = x - y
        s += d * d
    return math.sqrt(s)


def knn_classify(
    query: List[float],
    examples: List[Tuple[str, List[float]]],
    *,
    k: int = 5,
    max_distance: float = 2.5,
) -> Optional[Classification]:
    """
    Simple kNN using Euclidean distance. Returns label + pseudo-confidence.
    """
    if not query or not examples:
        return None

    scored: List[Tuple[float, str]] = []
    for label, vec in examples:
        dist = _euclidean(query, vec)
        if math.isfinite(dist):
            scored.append((dist, label))

    scored.sort(key=lambda t: t[0])
    top = scored[: max(1, int(k))]
    if not top:
        return None

    best_dist, best_label = top[0]
    if best_dist > max_distance:
        return None

    # Confidence decreases with distance.
    conf = max(0.0, min(1.0, 1.0 - (best_dist / max_distance)))
    return Classification(label=best_label, confidence=conf)


def parse_vectors(rows: List[Tuple[str, str]]) -> List[Tuple[str, List[float]]]:
    out: List[Tuple[str, List[float]]] = []
    for label, vector_json in rows:
        try:
            v = json.loads(vector_json)
        except Exception:
            continue
        if not isinstance(v, list):
            continue
        try:
            vec = [float(x) for x in v]
        except Exception:
            continue
        out.append((label, vec))
    return out


def infer_emotion_from_blendshapes(
    *,
    vector: List[float],
    labels: List[str],
) -> Optional[Classification]:
    """
    Heurística simple con blendshapes para:
    - enojo
    - felicidad
    - tristeza
    - angustia
    """
    if not vector or not labels or len(vector) != len(labels):
        return None

    m = {labels[i]: float(vector[i]) for i in range(len(labels))}

    def s(*names: str) -> float:
        return sum(m.get(n, 0.0) for n in names)

    # Happiness: smiles + cheek/eye squint
    felicidad = s("mouthSmileLeft", "mouthSmileRight", "cheekSquintLeft", "cheekSquintRight", "eyeSquintLeft", "eyeSquintRight")

    # Anger: brows down + jaw clench + nose sneer + mouth frown
    enojo = s("browDownLeft", "browDownRight", "jawClench", "noseSneerLeft", "noseSneerRight", "mouthFrownLeft", "mouthFrownRight")

    # Sadness: brow inner up + mouth frown + eyes down
    tristeza = s("browInnerUp", "mouthFrownLeft", "mouthFrownRight", "eyeLookDownLeft", "eyeLookDownRight")

    # Anguish/anxiety: eyes wide + brow inner up + mouth open/stretch + jaw drop
    angustia = s(
        "eyeWideLeft",
        "eyeWideRight",
        "browInnerUp",
        "jawDrop",
        "mouthOpen",
        "mouthStretchLeft",
        "mouthStretchRight",
    )

    scores = {
        "felicidad": felicidad,
        "enojo": enojo,
        "tristeza": tristeza,
        "angustia": angustia,
    }

    best_label = max(scores, key=lambda k: scores[k])
    best = float(scores[best_label])
    total = float(sum(scores.values())) + 1e-6
    # Normalize into 0..1
    conf = max(0.0, min(1.0, best / total * 2.0))
    if best < 0.15:
        return None
    return Classification(label=best_label, confidence=conf)
