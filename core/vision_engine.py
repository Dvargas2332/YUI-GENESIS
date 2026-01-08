from __future__ import annotations

import math
import os
import time
from collections import Counter, deque
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import requests

try:
    import cv2
except Exception:  # pragma: no cover
    cv2 = None  # type: ignore

try:
    import numpy as np
except Exception:  # pragma: no cover
    np = None  # type: ignore

try:
    import mediapipe as mp  # type: ignore
except Exception:  # pragma: no cover
    mp = None  # type: ignore


HAND_MODEL_URL = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
FACE_MODEL_URL = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task"
GESTURE_MODEL_URL = "https://storage.googleapis.com/mediapipe-models/gesture_recognizer/gesture_recognizer/float16/1/gesture_recognizer.task"


@dataclass(frozen=True)
class VisionResult:
    gestures: List[str]
    face_present: bool
    smile: bool
    hands_present: bool
    face_vector: Optional[List[float]] = None
    face_vector_labels: Optional[List[str]] = None
    hand_vector: Optional[List[float]] = None


class VisionEngine:
    """
    Visión “con IA” usando MediaPipe Tasks (compatible con mediapipe>=0.10 en Python 3.13):
    - FaceLandmarker (para sonrisa).
    - HandLandmarker (para gestos de manos).
    """

    def __init__(self, models_dir: Path | None = None):
        self.enabled = False
        self._last_wave_ts = 0.0
        self._last_wrist_x: Optional[float] = None
        self._last_wrist_dir: int = 0
        self._wave_switches = 0
        self._gesture_window: deque[set[str]] = deque(maxlen=max(1, int(os.getenv("YUI_GESTURE_STABLE_FRAMES", "4"))))

        if mp is None or cv2 is None or np is None:
            return

        # MediaPipe in this environment exposes Tasks via `mp.tasks.*` (not `mp.solutions`).
        if not hasattr(mp, "tasks") or not hasattr(mp, "Image") or not hasattr(mp, "ImageFormat"):
            return

        self._vision = getattr(mp.tasks, "vision", None)
        if self._vision is None:
            return

        self._models_dir = models_dir or (Path(__file__).resolve().parents[1] / "models" / "mediapipe")
        self._models_dir.mkdir(parents=True, exist_ok=True)

        hand_path = self._models_dir / "hand_landmarker.task"
        face_path = self._models_dir / "face_landmarker.task"
        gesture_path = self._models_dir / "gesture_recognizer.task"

        try:
            self._ensure_model(hand_path, HAND_MODEL_URL)
            self._ensure_model(face_path, FACE_MODEL_URL)
            if os.getenv("YUI_GESTURE_RECOGNIZER", "1") not in {"0", "false", "False"}:
                self._ensure_model(gesture_path, GESTURE_MODEL_URL)
        except Exception as e:
            print(f"[YUI] MediaPipe model download failed: {type(e).__name__}: {e}")
            return

        try:
            RunningMode = self._vision.RunningMode
            det = float(os.getenv("YUI_HAND_DET_CONF", "0.5"))
            pres = float(os.getenv("YUI_HAND_PRES_CONF", "0.5"))
            track = float(os.getenv("YUI_HAND_TRACK_CONF", "0.5"))

            self._gesture = None
            if os.getenv("YUI_GESTURE_RECOGNIZER", "1") not in {"0", "false", "False"}:
                try:
                    gesture_options = self._vision.GestureRecognizerOptions(
                        base_options=mp.tasks.BaseOptions(model_asset_path=str(gesture_path)),
                        running_mode=RunningMode.VIDEO,
                        num_hands=2,
                        min_hand_detection_confidence=det,
                        min_hand_presence_confidence=pres,
                        min_tracking_confidence=track,
                    )
                    self._gesture = self._vision.GestureRecognizer.create_from_options(gesture_options)
                except Exception as e:
                    print(f"[YUI] GestureRecognizer init failed, falling back to heuristics: {type(e).__name__}: {e}")
                    self._gesture = None

            hand_options = self._vision.HandLandmarkerOptions(
                base_options=mp.tasks.BaseOptions(model_asset_path=str(hand_path)),
                running_mode=RunningMode.VIDEO,
                num_hands=2,
                min_hand_detection_confidence=det,
                min_hand_presence_confidence=pres,
                min_tracking_confidence=track,
            )
            self._hand = self._vision.HandLandmarker.create_from_options(hand_options)

            face_options = self._vision.FaceLandmarkerOptions(
                base_options=mp.tasks.BaseOptions(model_asset_path=str(face_path)),
                running_mode=RunningMode.VIDEO,
                num_faces=1,
                output_face_blendshapes=True,
                min_face_detection_confidence=0.5,
                min_face_presence_confidence=0.5,
                min_tracking_confidence=0.5,
            )
            self._face = self._vision.FaceLandmarker.create_from_options(face_options)

            self.enabled = True
        except Exception as e:
            print(f"[YUI] MediaPipe init failed: {type(e).__name__}: {e}")
            self.enabled = False

    def process(self, frame_bgr) -> VisionResult:
        if not self.enabled or cv2 is None or np is None or mp is None:
            return VisionResult(gestures=[], face_present=False, smile=False, hands_present=False)

        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        ts_ms = int(time.time() * 1000)

        face_present, smile, face_vec, face_vec_labels = self._process_face(image, ts_ms)
        hand_gestures, hands_present, hand_vec = self._process_hands(image, ts_ms)

        gestures: List[str] = []
        if smile:
            gestures.append("smile")
        gestures.extend(hand_gestures)

        # Deduplicar
        seen = set()
        out: List[str] = []
        for g in gestures:
            if g not in seen:
                seen.add(g)
                out.append(g)

        return VisionResult(
            gestures=out,
            face_present=face_present,
            smile=smile,
            hands_present=hands_present,
            face_vector=face_vec,
            face_vector_labels=face_vec_labels,
            hand_vector=hand_vec,
        )

    def _process_face(self, image, ts_ms: int) -> Tuple[bool, bool, Optional[List[float]], Optional[List[str]]]:
        try:
            res = self._face.detect_for_video(image, ts_ms)
        except Exception:
            return False, False, None, None

        landmarks = getattr(res, "face_landmarks", None)
        if not landmarks:
            return False, False, None, None

        lm = landmarks[0]
        smile = self._detect_smile_landmarks(lm)
        vec, labels = self._extract_blendshape_vector(res)
        return True, smile, vec, labels

    def _extract_blendshape_vector(self, face_result) -> Tuple[Optional[List[float]], Optional[List[str]]]:
        """
        Returns a stable vector of blendshape scores sorted by category name.
        """
        blend = getattr(face_result, "face_blendshapes", None)
        if not blend:
            return None, None
        try:
            categories = blend[0]
        except Exception:
            return None, None

        items: List[tuple[str, float]] = []
        for c in categories:
            name = getattr(c, "category_name", None)
            score = getattr(c, "score", None)
            if name is None or score is None:
                continue
            items.append((str(name), float(score)))

        if not items:
            return None, None

        items.sort(key=lambda t: t[0])
        labels = [t[0] for t in items]
        vec = [t[1] for t in items]
        return vec, labels

    def _detect_smile_landmarks(self, lm) -> bool:
        """
        Heurística simple: ratio apertura/ancho de boca.
        Indices siguen el FaceMesh topology.
        """
        try:
            left = lm[61]
            right = lm[291]
            top = lm[13]
            bottom = lm[14]
        except Exception:
            return False

        mouth_w = self._dist2d(left, right)
        mouth_h = self._dist2d(top, bottom)
        if mouth_w <= 0:
            return False
        ratio = mouth_h / mouth_w
        return ratio > 0.20

    def _process_hands(self, image, ts_ms: int) -> Tuple[List[str], bool, Optional[List[float]]]:
        landmarks = None
        handedness = None
        gesture_hits: List[str] = []
        hand_vec: Optional[List[float]] = None
        hands: List[tuple[Optional[str], object]] = []
        recognized_by_hand: List[bool] = []

        # By default, only use heuristics when GestureRecognizer is unavailable (it tends to be noisy).
        heur_env = (os.getenv("YUI_GESTURE_USE_HEURISTICS") or "").strip()
        if heur_env:
            use_heuristics = heur_env not in {"0", "false", "False"}
        else:
            use_heuristics = getattr(self, "_gesture", None) is None

        # Prefer MediaPipe GestureRecognizer (more accurate), fall back to HandLandmarker + heuristics.
        if getattr(self, "_gesture", None) is not None:
            try:
                res = self._gesture.recognize_for_video(image, ts_ms)
            except Exception:
                res = None

            if res is not None:
                landmarks = getattr(res, "hand_landmarks", None)
                handedness = getattr(res, "handedness", None)
                top_gestures = getattr(res, "gestures", None)
                min_score = float(os.getenv("YUI_GESTURE_MIN_SCORE", "0.55"))
                min_score_thumbs = float(os.getenv("YUI_GESTURE_MIN_SCORE_THUMBS_UP", str(max(min_score, 0.85))))
                if landmarks:
                    for idx, lm in enumerate(landmarks):
                        if not self._hand_quality_ok(lm):
                            continue
                        if hand_vec is None:
                            hand_vec = self._hand_vector(lm)

                        # GestureRecognizer returns a list of candidate gestures per hand.
                        handed = self._handedness(handedness, idx)
                        recognized_here = False
                        try:
                            state = self._finger_state(lm, handedness=handed)
                            cand = (top_gestures[idx] if top_gestures else None) or []
                            best = cand[0] if cand else None
                            name = getattr(best, "category_name", None)
                            score = getattr(best, "score", None)
                            if name and (score is None or float(score) >= min_score):
                                mapped = self._map_mp_gesture(str(name))
                                if mapped == "thumbs_up":
                                    if score is not None and float(score) < min_score_thumbs:
                                        mapped = None
                                    elif not self._validate_thumbs_up_landmarks(lm, handedness=handed):
                                        mapped = None
                                    elif not self._validate_thumbs_up(state):
                                        mapped = None
                                if mapped:
                                    gesture_hits.append(mapped)
                                    recognized_here = True
                                    if mapped == "open_palm" and self._detect_wave(lm[0]):
                                        gesture_hits.append("wave")
                        except Exception:
                            pass
                        hands.append((handed, lm))
                        recognized_by_hand.append(recognized_here)

        if not landmarks:
            try:
                res2 = self._hand.detect_for_video(image, ts_ms)
                landmarks = getattr(res2, "hand_landmarks", None)
                handedness = getattr(res2, "handedness", None)
            except Exception:
                self._reset_wave()
                return [], False, None

        if not landmarks:
            self._reset_wave()
            self._clear_gesture_window()
            return [], False, None

        # If we didn't build hands from GestureRecognizer, build from HandLandmarker now.
        if not hands:
            for idx, lm in enumerate(landmarks):
                if not self._hand_quality_ok(lm):
                    continue
                handed = self._handedness(handedness, idx)
                hands.append((handed, lm))
                recognized_by_hand.append(False)
                if hand_vec is None:
                    hand_vec = self._hand_vector(lm)

        if not hands:
            self._reset_wave()
            self._clear_gesture_window()
            return [], False, None

        # Heuristic fallback: disabled by default when GestureRecognizer is available
        # (it can be noisy and cause false positives like "thumbs_up").
        if use_heuristics:
            if len(recognized_by_hand) != len(hands):
                recognized_by_hand = [False] * len(hands)
            for (handed, lm), had_recognized in zip(hands, recognized_by_hand):
                if had_recognized:
                    continue
                state = self._finger_state(lm, handedness=handed)
                heuristic = self._classify_gestures(lm, state)
                for g in heuristic:
                    if g == "thumbs_up" and not self._validate_thumbs_up_landmarks(lm, handedness=handed):
                        continue
                    gesture_hits.append(g)

        # Temporal smoothing (reduces flicker): keep gestures that persist across frames.
        wave_now = "wave" in gesture_hits
        stable = self._smooth_gestures([g for g in gesture_hits if g != "wave"])
        if wave_now:
            stable.append("wave")

        return stable, True, hand_vec

    def _hand_quality_ok(self, lm) -> bool:
        """
        Extra guard to reduce false positives when MediaPipe hallucinates hands.
        Uses the landmark bounding box area in normalized coords.
        """
        try:
            xs = [float(p.x) for p in lm]
            ys = [float(p.y) for p in lm]
        except Exception:
            return False

        if not xs or not ys:
            return False

        # Reject clearly invalid normalized coords.
        if any((x < -0.25 or x > 1.25) for x in xs) or any((y < -0.25 or y > 1.25) for y in ys):
            return False

        dx = (max(xs) - min(xs))
        dy = (max(ys) - min(ys))
        area = dx * dy

        # Typical hand bbox area is ~0.03-0.20 depending on distance; noise tends to be tiny.
        min_area = float(os.getenv("YUI_HAND_MIN_BBOX_AREA", "0.02"))
        return area >= min_area

    def _clear_gesture_window(self) -> None:
        try:
            self._gesture_window.clear()
        except Exception:
            # Fallback: recreate deque with same maxlen.
            self._gesture_window = deque(maxlen=int(getattr(self._gesture_window, "maxlen", 4) or 4))

    def _smooth_gestures(self, gestures: List[str]) -> List[str]:
        stable_frames = int(self._gesture_window.maxlen or 1)
        min_hits = int(os.getenv("YUI_GESTURE_STABLE_MIN", "3"))
        min_hits = max(1, min(min_hits, stable_frames))

        self._gesture_window.append(set(gestures))
        if len(self._gesture_window) < stable_frames:
            return []

        counts: Counter[str] = Counter()
        for s in self._gesture_window:
            counts.update(s)

        keep = [g for g, c in counts.items() if c >= min_hits]
        order = {"wave": 0, "thumbs_up": 1, "open_palm": 2, "peace": 3, "fist": 4}
        keep.sort(key=lambda x: order.get(x, 99))
        return keep

    def _map_mp_gesture(self, name: str) -> Optional[str]:
        n = (name or "").strip().lower()
        if not n:
            return None
        # Common MediaPipe gesture labels.
        mapping = {
            "open_palm": "open_palm",
            "closed_fist": "fist",
            "thumb_up": "thumbs_up",
            "victory": "peace",
            "pointing_up": "point",
            "iloveyou": "iloveyou",
        }
        n2 = n.replace(" ", "_")
        return mapping.get(n2)

    def _handedness(self, handedness, idx: int) -> Optional[str]:
        try:
            if handedness is None:
                return None
            h = handedness[idx]
            if isinstance(h, list) and h:
                c = h[0]
            else:
                c = h
            name = getattr(c, "category_name", None) or getattr(c, "display_name", None)
            if not name:
                return None
            return str(name).strip().lower()
        except Exception:
            return None

    def _hand_vector(self, lm) -> Optional[List[float]]:
        """
        Flattened normalized landmarks (x,y,z) relative to wrist, scaled by hand size.
        """
        try:
            wrist = lm[0]
            mcp = lm[9]
        except Exception:
            return None

        scale = self._dist2d(wrist, mcp)
        if scale <= 0:
            scale = 1.0

        out: List[float] = []
        for p in lm:
            out.append((float(p.x) - float(wrist.x)) / scale)
            out.append((float(p.y) - float(wrist.y)) / scale)
            out.append((float(p.z) - float(wrist.z)) / scale)
        return out

    def _finger_state(self, lm, *, handedness: Optional[str]) -> Dict[str, bool]:
        # indices: https://developers.google.com/mediapipe/solutions/vision/hand_landmarker
        wrist = lm[0]

        def ext(tip: int, dip: int, pip: int, mcp: int) -> bool:
            try:
                return (lm[tip].y < lm[dip].y) and (lm[dip].y < lm[pip].y) and (lm[pip].y < lm[mcp].y)
            except Exception:
                return False

        index_ext = ext(8, 7, 6, 5)
        middle_ext = ext(12, 11, 10, 9)
        ring_ext = ext(16, 15, 14, 13)
        pinky_ext = ext(20, 19, 18, 17)

        thumb_ext = False
        try:
            thumb_tip = lm[4]
            thumb_ip = lm[3]
            if handedness in {"right"}:
                thumb_ext = (float(thumb_tip.x) - float(thumb_ip.x)) > 0.04
            elif handedness in {"left"}:
                thumb_ext = (float(thumb_ip.x) - float(thumb_tip.x)) > 0.04
            else:
                thumb_ext = self._dist2d(thumb_tip, wrist) > self._dist2d(thumb_ip, wrist)
        except Exception:
            thumb_ext = False

        return {
            "thumb": bool(thumb_ext),
            "index": bool(index_ext),
            "middle": bool(middle_ext),
            "ring": bool(ring_ext),
            "pinky": bool(pinky_ext),
        }

    def _validate_thumbs_up(self, state: Dict[str, bool]) -> bool:
        # Reduce false positives: require thumb up and all other fingers down.
        if not state.get("thumb"):
            return False
        others = ["index", "middle", "ring", "pinky"]
        return all(not state.get(k, False) for k in others)

    def _validate_thumbs_up_landmarks(self, lm, *, handedness: Optional[str]) -> bool:
        """
        Extra guard to reduce thumbs-up false positives.
        Requires:
        - Thumb tip clearly above wrist (relative to hand size).
        - Other fingertips not extended up (tip not significantly above MCP).
        """
        try:
            wrist = lm[0]
            thumb_tip = lm[4]
            thumb_ip = lm[3]
            middle_mcp = lm[9]
        except Exception:
            return False

        scale = self._dist2d(wrist, middle_mcp)
        if scale <= 0:
            scale = 0.2  # reasonable fallback in normalized coords

        # Thumb must point "up" (smaller y). These ratios are conservative.
        if float(thumb_tip.y) >= float(wrist.y) - (0.15 * scale):
            return False
        if float(thumb_tip.y) >= float(thumb_ip.y) - (0.08 * scale):
            return False

        # Other fingers should be folded: tip should not be clearly above MCP.
        for tip_i, mcp_i in [(8, 5), (12, 9), (16, 13), (20, 17)]:
            try:
                if float(lm[tip_i].y) < float(lm[mcp_i].y) - (0.05 * scale):
                    return False
            except Exception:
                return False

        return True

    def _classify_gestures(self, lm, state: Dict[str, bool]) -> List[str]:
        gestures: List[str] = []
        fingers_up = sum(1 for v in state.values() if v)

        if fingers_up == 5:
            gestures.append("open_palm")
            if self._detect_wave(lm[0]):
                gestures.append("wave")
        elif fingers_up == 0:
            gestures.append("fist")
        elif state["thumb"] and not state["index"] and not state["middle"] and not state["ring"] and not state["pinky"]:
            gestures.append("thumbs_up")
        elif state["index"] and state["middle"] and not state["ring"] and not state["pinky"]:
            gestures.append("peace")

        return gestures

    def _detect_wave(self, wrist) -> bool:
        now = time.time()
        x = float(wrist.x)

        if self._last_wrist_x is None:
            self._last_wrist_x = x
            self._last_wrist_dir = 0
            self._wave_switches = 0
            self._last_wave_ts = now
            return False

        dx = x - self._last_wrist_x
        self._last_wrist_x = x

        dir_now = 1 if dx > 0.015 else (-1 if dx < -0.015 else 0)
        if dir_now != 0 and self._last_wrist_dir != 0 and dir_now != self._last_wrist_dir:
            self._wave_switches += 1
        if dir_now != 0:
            self._last_wrist_dir = dir_now

        if now - self._last_wave_ts > 1.5:
            self._reset_wave()
            self._last_wave_ts = now
            return False

        if self._wave_switches >= 2:
            self._reset_wave()
            return True

        return False

    def _reset_wave(self) -> None:
        self._last_wrist_x = None
        self._last_wrist_dir = 0
        self._wave_switches = 0
        self._last_wave_ts = 0.0

    def _dist2d(self, a, b) -> float:
        dx = float(a.x) - float(b.x)
        dy = float(a.y) - float(b.y)
        return math.hypot(dx, dy)

    def _ensure_model(self, path: Path, url: str) -> None:
        if path.exists() and path.stat().st_size > 0:
            return
        resp = requests.get(url, timeout=60)
        resp.raise_for_status()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(resp.content)
