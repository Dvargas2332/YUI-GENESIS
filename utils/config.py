# Configuraciones globales
class Config:
    CAMERA_INDEX = 0
    KNOWN_FACES_DIR = "known_faces/"
    GESTURE_THRESHOLDS = {
        'hand_open': 0.8,
        'thumbs_up': 0.7
    }
    VOICE_SETTINGS = {
        'energy_threshold': 4000,
        'pause_threshold': 0.8
    }