from __future__ import annotations

import time

import cv2


def probe(max_index: int = 5):
    backends = [
        ("dshow", getattr(cv2, "CAP_DSHOW", None)),
        ("msmf", getattr(cv2, "CAP_MSMF", None)),
        ("any", None),
    ]

    for idx in range(max_index + 1):
        for name, api in backends:
            cap = None
            try:
                cap = cv2.VideoCapture(idx) if api is None else cv2.VideoCapture(idx, api)
                cap.set(cv2.CAP_PROP_CONVERT_RGB, 1)
            except Exception:
                cap = None

            opened = False
            if cap is not None:
                try:
                    opened = cap.isOpened()
                except Exception:
                    opened = False

            if not opened:
                if cap is not None:
                    try:
                        cap.release()
                    except Exception:
                        pass
                continue

            ok, frame = cap.read()
            shape = None if not ok else getattr(frame, "shape", None)
            print(f"index={idx} backend={name} opened={opened} read_ok={ok} shape={shape}")
            try:
                cap.release()
            except Exception:
                pass
            time.sleep(0.1)


if __name__ == "__main__":
    probe()
