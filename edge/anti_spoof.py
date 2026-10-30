"""
Presento Face Anti-Spoofing Module.
Wraps the OpenVINO Open Model Zoo 'anti-spoof-mn3' binary classifier
running natively on Intel Iris Xe GPU (Intel GPU) to distinguish real faces from spoof attempts.
"""
from __future__ import annotations

import logging
import os
import time
from typing import Tuple
import cv2
import numpy as np
import openvino as ov

logger = logging.getLogger("presento.edge.antispoof")

# Official CelebA-Spoof normalization parameters from Open Model Zoo model.yml
CELEBA_MEAN = np.array([151.2405, 119.5950, 107.8395], dtype=np.float32)  # R, G, B
CELEBA_SCALE = np.array([63.0105, 56.4570, 55.0035], dtype=np.float32)    # R, G, B


def crop_face(
    frame: np.ndarray,
    bbox: tuple[int, int, int, int],
    scale: float = 1.2,
) -> np.ndarray:
    """
    Extract a square face crop with padding to preserve contextual anti-spoofing cues.

    Args:
        frame: Full camera image (BGR uint8).
        bbox: Face bounding box (left, top, right, bottom).
        scale: Expansion factor around face center (default: 1.2).

    Returns:
        Square face crop (BGR uint8) or blank 128x128 image if invalid.
    """
    if frame is None or frame.size == 0 or bbox is None:
        return np.zeros((128, 128, 3), dtype=np.uint8)

    left, top, right, bottom = bbox
    w = right - left
    h = bottom - top
    if w <= 0 or h <= 0:
        return np.zeros((128, 128, 3), dtype=np.uint8)

    cx = (left + right) / 2.0
    cy = (top + bottom) / 2.0
    side = max(w, h) * scale

    orig_h, orig_w = frame.shape[:2]
    x1 = int(round(max(0, cx - side / 2.0)))
    y1 = int(round(max(0, cy - side / 2.0)))
    x2 = int(round(min(orig_w, cx + side / 2.0)))
    y2 = int(round(min(orig_h, cy + side / 2.0)))

    crop = frame[y1:y2, x1:x2]
    if crop.size == 0:
        return np.zeros((128, 128, 3), dtype=np.uint8)

    return crop


class AntiSpoofClassifier:
    """
    Native OpenVINO anti-spoof-mn3 binary face anti-spoofing classifier.
    MobileNetV3 trained on CelebA-Spoof.
    Output: Class 0 = REAL, Class 1 = SPOOF.
    """

    def __init__(
        self,
        model_path: str = "data/models/anti-spoof-mn3.onnx",
        device: str = "GPU",
    ):
        self.model_path = model_path
        self.requested_device = device.upper()

        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Anti-spoof model file not found at: {self.model_path}")

        # Initialize OpenVINO Core and verify hardware
        self.core = ov.Core()
        available_devices = [d.upper() for d in self.core.available_devices]
        logger.info("OpenVINO Core initialized for Anti-Spoof. Available devices: %s", available_devices)

        # Explicit verification: Do NOT silently claim GPU while running CPU
        if self.requested_device == "GPU":
            if "GPU" not in available_devices:
                raise RuntimeError(
                    f"OpenVINO GPU device requested but 'GPU' not found in available devices: {available_devices}"
                )
            self.active_device = "GPU"
        elif self.requested_device == "CPU":
            self.active_device = "CPU"
        else:
            if self.requested_device not in available_devices:
                raise RuntimeError(
                    f"Requested OpenVINO device '{self.requested_device}' is unavailable: {available_devices}"
                )
            self.active_device = self.requested_device

        try:
            self.full_device_name = self.core.get_property(self.active_device, "FULL_DEVICE_NAME")
        except Exception:
            self.full_device_name = self.active_device

        logger.info(
            "Compiling anti-spoof-mn3 '%s' on verified OpenVINO device: %s (%s)...",
            self.model_path,
            self.active_device,
            self.full_device_name,
        )

        t0 = time.perf_counter()
        ov_model = self.core.read_model(self.model_path)
        self.compiled_model = self.core.compile_model(ov_model, self.active_device)
        self.infer_request = self.compiled_model.create_infer_request()
        self.input_tensor_name = self.compiled_model.inputs[0].any_name
        self.output_tensor = self.compiled_model.outputs[0]
        self.load_time_ms = (time.perf_counter() - t0) * 1000.0

        # Warm-up inference
        dummy_input = np.zeros((1, 3, 128, 128), dtype=np.float32)
        t_warm0 = time.perf_counter()
        _ = self.infer_request.infer({self.input_tensor_name: dummy_input})
        self.warmup_time_ms = (time.perf_counter() - t_warm0) * 1000.0

        logger.info(
            "anti-spoof-mn3 compiled on %s in %.1f ms (warmup: %.1f ms)",
            self.full_device_name,
            self.load_time_ms,
            self.warmup_time_ms,
        )
        print(f"[AntiSpoof] Active OpenVINO Device: {self.active_device} -> {self.full_device_name}", flush=True)

    def preprocess(self, face_crop: np.ndarray) -> np.ndarray:
        """
        Preprocess face crop according to CelebA-Spoof MobileNetV3 specification:
        1. Resize BGR crop to (128, 128).
        2. Convert BGR to RGB.
        3. Normalize: (RGB - mean) / scale.
        4. Transpose to NCHW (1, 3, 128, 128) float32.
        """
        if face_crop is None or face_crop.size == 0:
            face_crop = np.zeros((128, 128, 3), dtype=np.uint8)

        if face_crop.shape[:2] != (128, 128):
            resized = cv2.resize(face_crop, (128, 128), interpolation=cv2.INTER_LINEAR)
        else:
            resized = face_crop

        # BGR -> RGB float32
        rgb = resized[:, :, ::-1].astype(np.float32)

        # CelebA-Spoof channel-wise normalization
        normed = (rgb - CELEBA_MEAN) / CELEBA_SCALE

        # HWC -> CHW -> NCHW
        blob = np.transpose(normed, (2, 0, 1))[np.newaxis, ...].astype(np.float32)
        return blob

    def predict(self, face_crop: np.ndarray) -> Tuple[float, float, str, float]:
        """
        Classify face crop as REAL or SPOOF.

        Args:
            face_crop: BGR uint8 face crop image.

        Returns:
            Tuple of:
            - prob_real: float [0.0, 1.0] (Class 0 probability)
            - prob_spoof: float [0.0, 1.0] (Class 1 probability)
            - label: str ("REAL" or "SPOOF")
            - latency_ms: float (inference latency in milliseconds)
        """
        blob = self.preprocess(face_crop)

        t0 = time.perf_counter()
        res = self.infer_request.infer({self.input_tensor_name: blob})
        latency_ms = (time.perf_counter() - t0) * 1000.0

        raw_output = res[self.output_tensor].flatten()

        p_real = float(raw_output[0])
        p_spoof = float(raw_output[1])

        total = p_real + p_spoof
        if total > 0:
            p_real = max(0.0, min(1.0, p_real / total))
            p_spoof = max(0.0, min(1.0, p_spoof / total))
        else:
            p_real, p_spoof = 0.5, 0.5

        label = "REAL" if p_real >= p_spoof else "SPOOF"
        return p_real, p_spoof, label, latency_ms
