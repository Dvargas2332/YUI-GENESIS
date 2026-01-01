from __future__ import annotations

from yui_io.stt import list_devices


def main() -> None:
    devs = list_devices()
    if devs.speech_recognition:
        print("SpeechRecognition devices:")
        for idx, name in devs.speech_recognition:
            print(f"  {idx}: {name}")
    else:
        print("SpeechRecognition devices: (none)")

    if devs.sounddevice:
        print("SoundDevice input devices:")
        for idx, name in devs.sounddevice:
            print(f"  {idx}: {name}")
    else:
        print("SoundDevice input devices: (none)")


if __name__ == "__main__":
    main()
