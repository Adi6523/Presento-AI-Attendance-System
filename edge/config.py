from dataclasses import dataclass
import os
from dotenv import load_dotenv

load_dotenv()

@dataclass(frozen=True)
class Settings:
    # Device identification
    device_id: str = os.getenv("DEVICE_ID", "pi-classroom-a101")

    # API configuration
    api_url: str = os.getenv("API_URL", "http://127.0.0.1:8000").rstrip("/")
    api_token: str = os.getenv("PRESENTO_AI_API_KEY", os.getenv("API_TOKEN", "change-me-before-production-ai-key"))

    # Class context (optional, for class-scoped recognition)
    class_id: str | None = os.getenv("CLASS_ID")

    # Hardware pins
    pir_pin: int = int(os.getenv("PIR_PIN", "17"))
    buzzer_pin: int = int(os.getenv("BUZZER_PIN", "18"))
    camera_index: int = int(os.getenv("CAMERA_INDEX", "0"))
    camera_source: str = os.getenv("CAMERA_SOURCE", "stream")
    pi_ip: str = os.getenv("PI_IP", "presento-pi.local")
    stream_port: int = int(os.getenv("PRESENTO_STREAM_PORT", "5001"))
    control_port: int = int(os.getenv("PRESENTO_CONTROL_PORT", "5002"))
    control_token: str = os.getenv("PRESENTO_CONTROL_TOKEN", os.getenv("API_TOKEN", "presento-camera-secret"))
    camera_session_timeout: float = max(60.0, float(os.getenv("PRESENTO_CAMERA_SESSION_TIMEOUT", "60.0")))
    camera_heartbeat_interval: float = float(os.getenv("PRESENTO_CAMERA_HEARTBEAT_INTERVAL", "15.0"))

    # Recognition settings
    tolerance: float = float(os.getenv("MATCH_TOLERANCE", "0.48"))
    cooldown_seconds: float = float(os.getenv("COOLDOWN_SECONDS", "10.0"))
    frame_skip: int = int(os.getenv("FRAME_SKIP", "3"))  # Process every Nth frame

    # Liveness settings
    liveness_seconds: int = int(os.getenv("LIVENESS_SECONDS", "5"))
    min_blinks_required: int = int(os.getenv("MIN_BLINKS_REQUIRED", "2"))
    blink_threshold: float = float(os.getenv("BLINK_THRESHOLD", "0.20"))

    # Quality settings
    min_face_size_ratio: float = float(os.getenv("MIN_FACE_SIZE_RATIO", "0.2"))
    max_face_size_ratio: float = float(os.getenv("MAX_FACE_SIZE_RATIO", "0.8"))
    position_tolerance: float = float(os.getenv("POSITION_TOLERANCE", "0.15"))
    blur_threshold: float = float(os.getenv("BLUR_THRESHOLD", "100.0"))
    lighting_min: int = int(os.getenv("LIGHTING_MIN", "50"))
    lighting_max: int = int(os.getenv("LIGHTING_MAX", "200"))

    # Performance settings
    cache_size: int = int(os.getenv("EMBEDDING_CACHE_SIZE", "100"))
    cache_ttl: int = int(os.getenv("EMBEDDING_CACHE_TTL", "300"))  # 5 minutes
    use_tracking: bool = os.getenv("USE_FACE_TRACKING", "true").lower() == "true"

    # Phase 2 Decoupled Detection Settings
    detection_width: int = int(os.getenv("DETECTION_WIDTH", "640"))
    detection_height: int = int(os.getenv("DETECTION_HEIGHT", "360"))
    detection_fps: float = float(os.getenv("DETECTION_FPS", "9.0"))  # Target 8-10 FPS for HOG
    box_persistence_seconds: float = float(os.getenv("BOX_PERSISTENCE_SECONDS", "0.6"))
    box_smoothing_factor: float = float(os.getenv("BOX_SMOOTHING_FACTOR", "0.35"))

    # Phase 1 Neural SCRFD Detection Settings
    scrfd_model_path: str = os.getenv("SCRFD_MODEL_PATH", "data/models/scrfd_500m_bnkps.onnx")
    scrfd_input_size: int = int(os.getenv("SCRFD_INPUT_SIZE", "640"))
    scrfd_conf_thresh: float = float(os.getenv("SCRFD_CONF_THRESH", "0.5"))
    scrfd_nms_thresh: float = float(os.getenv("SCRFD_NMS_THRESH", "0.4"))
    openvino_device: str = os.getenv("OPENVINO_DEVICE", "GPU")
    scrfd_target_fps: float = float(os.getenv("SCRFD_TARGET_FPS", "30.0"))

    # Phase 2A Neural ArcFace Embedding Settings
    arcface_model_path: str = os.getenv("ARCFACE_MODEL_PATH", "data/models/w600k_mbf.onnx")
    anti_spoof_model_path: str = os.getenv(
        "ANTI_SPOOF_MODEL_PATH",
        "data/models/anti-spoof-mn3.onnx",
    )
    minifas_model_path: str = os.getenv(
        "MINIFAS_MODEL_PATH",
        "data/models/minifasnet_v2se.onnx",
    )

    arcface_enabled: bool = os.getenv("ARCFACE_ENABLED", "true").lower() == "true"
    arcface_target_fps: float = float(os.getenv("ARCFACE_TARGET_FPS", "12.0"))  # ~10-15 FPS per face

    # Phase 2B Face Enrollment & Identity Matching Settings
    face_enrollment_path: str = os.getenv("FACE_ENROLLMENT_PATH", "data/face_embeddings.npz")
    enrollment_samples_target: int = int(os.getenv("ENROLLMENT_SAMPLES_TARGET", "12"))
    enrollment_samples_min: int = int(os.getenv("ENROLLMENT_SAMPLES_MIN", "8"))
    enrollment_min_face_size: int = int(os.getenv("ENROLLMENT_MIN_FACE_SIZE", "80"))
    enrollment_min_conf: float = float(os.getenv("ENROLLMENT_MIN_CONF", "0.70"))
    enrollment_duplicate_threshold: float = float(os.getenv("ENROLLMENT_DUPLICATE_THRESHOLD", "0.985"))  # Configurable enrollment-only sample similarity check

    # Phase V2 Smart Enrollment & Identity Settings
    enrollment_raw_target: int = int(os.getenv("ENROLLMENT_RAW_TARGET", "60"))
    enrollment_min_samples: int = int(os.getenv("ENROLLMENT_MIN_SAMPLES", "15"))
    enrollment_max_references: int = int(os.getenv("ENROLLMENT_MAX_REFERENCES", "25"))
    enrollment_diversity_threshold: float = float(os.getenv("ENROLLMENT_DIVERSITY_THRESHOLD", "0.92"))
    identity_engine_mode: str = os.getenv("IDENTITY_ENGINE_MODE", "baseline")  # "baseline", "classifier", "fused"
    identity_weight_multiref: float = float(os.getenv("IDENTITY_WEIGHT_MULTIREF", "0.50"))
    identity_weight_prototype: float = float(os.getenv("IDENTITY_WEIGHT_PROTOTYPE", "0.25"))
    identity_weight_classifier: float = float(os.getenv("IDENTITY_WEIGHT_CLASSIFIER", "0.25"))
    identity_min_margin: float = float(os.getenv("IDENTITY_MIN_MARGIN", "0.10"))
    identity_classifier_path: str = os.getenv("IDENTITY_CLASSIFIER_PATH", "data/identity_classifier.npz")

    # Phase V2 Liveness Session & Temporal Settings
    liveness_window_size: int = int(os.getenv("LIVENESS_WINDOW_SIZE", "30"))
    liveness_real_entry_ratio: float = float(os.getenv("LIVENESS_REAL_ENTRY_RATIO", "0.80"))
    liveness_spoof_entry_ratio: float = float(os.getenv("LIVENESS_SPOOF_ENTRY_RATIO", "0.85"))
    liveness_real_exit_ratio: float = float(os.getenv("LIVENESS_REAL_EXIT_RATIO", "0.50"))
    liveness_required_consecutive_windows: int = int(os.getenv("LIVENESS_REQUIRED_CONSECUTIVE_WINDOWS", "2"))
    liveness_ema_alpha: float = float(os.getenv("LIVENESS_EMA_ALPHA", "0.15"))
    liveness_lease_duration: float = float(os.getenv("LIVENESS_LEASE_DURATION", "6.0"))
    liveness_hold_timeout: float = float(os.getenv("LIVENESS_HOLD_TIMEOUT", "1.5"))

    # ── Pi deployment settings ────────────────────────────────────────────────
    # Headless mode: disable cv2.imshow()/waitKey() for Pi without a display
    headless: bool = os.getenv("HEADLESS", "false").lower() == "true"

    # Periodic embedding refresh: fetch new enrollments from the API every N seconds
    # Set to 0 to disable (only load at startup)
    embedding_refresh_seconds: int = int(os.getenv("EMBEDDING_REFRESH_SECONDS", "300"))  # 5 min

    # Room identifier for this edge device (sent with attendance events)
    room: str = os.getenv("ROOM", "A101")

    # Log level: DEBUG, INFO, WARNING, ERROR
    log_level: str = os.getenv("LOG_LEVEL", "INFO")

settings = Settings()
