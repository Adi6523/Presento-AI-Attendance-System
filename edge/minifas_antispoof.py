"""
Presento MiniFASNet V2 SE Anti-Spoofing Module.

Wraps the MiniFASNet V2 SE (Squeeze-and-Excitation) binary anti-spoofing classifier
running natively on OpenVINO Core (preferring Intel Iris Xe GPU).

Model Provenance & Architecture Specifications:
- Source: https://github.com/facenox/face-antispoof-onnx (models/best/98.20/best_model.onnx)
- Model File: data/models/minifasnet_v2se.onnx
- File Size: 1,912,594 bytes (~1.82 MB)
- SHA256: af2381b88f38769222ed93379e12444e2a50814575de1c46170de570c55a42b6
- Input Dimension: [-1, 3, 128, 128] (NCHW float32, RGB)
- Normalization: Scaled to [0.0, 1.0] via img.astype(np.float32) / 255.0 (NO mean subtraction)
- Face Crop Expansion: 1.5x bounding box expansion with cv2.BORDER_REFLECT_101 padding
- Output Dimension: [-1, 2] raw logits
- Class Mapping:
    Index 0 = REAL
    Index 1 = SPOOF
- Probability: Softmax across logits (p_real = exp(z0)/(exp(z0)+exp(z1)), p_spoof = exp(z1)/(exp(z0)+exp(z1)))
"""
from __future__ import annotations

import logging
import os
import time
from typing import Tuple

import cv2
import numpy as np
import openvino as ov

logger = logging.getLogger("presento.edge.minifas_antispoof")


def crop_face_minifas(
    frame: np.ndarray,
    bbox: tuple[int, int, int, int],
    scale: float = 1.5,
) -> np.ndarray:
    """
    Extract a square face crop from bounding box with expansion and reflection border padding,
    following the reference implementation in facenox/face-antispoof-onnx.

    Args:
        frame: Full camera image (BGR uint8).
        bbox: Face bounding box as (left, top, right, bottom).
        scale: Bounding box expansion factor (default: 1.5).

    Returns:
        Square face crop (BGR uint8) of size (crop_size, crop_size, 3).
    """
    if frame is None or frame.size == 0 or bbox is None:
        return np.zeros((128, 128, 3), dtype=np.uint8)

    left, top, right, bottom = bbox
    w = right - left
    h = bottom - top

    if w <= 0 or h <= 0:
        return np.zeros((128, 128, 3), dtype=np.uint8)

    orig_h, orig_w = frame.shape[:2]
    max_dim = max(w, h)
    center_x = left + w / 2.0
    center_y = top + h / 2.0

    crop_size = int(round(max_dim * scale))
    if crop_size <= 0:
        return np.zeros((128, 128, 3), dtype=np.uint8)

    x = int(round(center_x - crop_size / 2.0))
    y = int(round(center_y - crop_size / 2.0))

    crop_x1 = max(0, x)
    crop_y1 = max(0, y)
    crop_x2 = min(orig_w, x + crop_size)
    crop_y2 = min(orig_h, y + crop_size)

    top_pad = int(max(0, -y))
    left_pad = int(max(0, -x))
    bottom_pad = int(max(0, (y + crop_size) - orig_h))
    right_pad = int(max(0, (x + crop_size) - orig_w))

    if crop_x2 > crop_x1 and crop_y2 > crop_y1:
        cropped = frame[crop_y1:crop_y2, crop_x1:crop_x2, :]
    else:
        cropped = np.zeros((0, 0, 3), dtype=frame.dtype)

    result = cv2.copyMakeBorder(
        cropped,
        top_pad,
        bottom_pad,
        left_pad,
        right_pad,
        cv2.BORDER_REFLECT_101,
    )

    return result


class MiniFASNetV2SEClassifier:
    """
    Native OpenVINO MiniFASNet V2 SE binary face anti-spoofing classifier.
    Architecture: MiniFASNet V2 SE with Squeeze-and-Excitation blocks.
    Input: [1, 3, 128, 128] RGB, [0, 1] normalized.
    Output: Class 0 = REAL, Class 1 = SPOOF.
    """

    def __init__(
        self,
        model_path: str = "data/models/minifasnet_v2se.onnx",
        device: str = "GPU",
        model_img_size: int = 128,
    ):
        self.model_path = model_path
        self.requested_device = device.upper()
        self.model_img_size = model_img_size

        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"MiniFASNet V2 SE model file not found at: {self.model_path}")

        # Initialize OpenVINO Core and verify hardware
        self.core = ov.Core()
        available_devices = [d.upper() for d in self.core.available_devices]
        logger.info("OpenVINO Core initialized for MiniFASNet V2 SE. Available devices: %s", available_devices)

        # Explicit verification: Do NOT silently fall back to CPU if GPU requested
        if self.requested_device == "GPU":
            if "GPU" not in available_devices:
                raise RuntimeError(
                    f"OpenVINO GPU device requested for MiniFASNet but 'GPU' not found in available devices: {available_devices}"
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
            "Compiling MiniFASNet V2 SE '%s' on verified OpenVINO device: %s (%s)...",
            self.model_path,
            self.active_device,
            self.full_device_name,
        )

        t0 = time.perf_counter()
        ov_model = self.core.read_model(self.model_path)

        # Validate input/output shapes
        self._validate_model_structure(ov_model)

        self.compiled_model = self.core.compile_model(ov_model, self.active_device)
        self.infer_request = self.compiled_model.create_infer_request()
        self.input_tensor_name = self.compiled_model.inputs[0].any_name
        self.output_tensor = self.compiled_model.outputs[0]
        self.load_time_ms = (time.perf_counter() - t0) * 1000.0

        # Warm-up inference
        dummy_input = np.zeros((1, 3, self.model_img_size, self.model_img_size), dtype=np.float32)
        t_warm0 = time.perf_counter()
        _ = self.infer_request.infer({self.input_tensor_name: dummy_input})
        self.warmup_time_ms = (time.perf_counter() - t_warm0) * 1000.0

        logger.info(
            "MiniFASNet V2 SE compiled on %s in %.1f ms (warmup: %.1f ms)",
            self.full_device_name,
            self.load_time_ms,
            self.warmup_time_ms,
        )
        print(f"[MiniFASNet] Active OpenVINO Device: {self.active_device} -> {self.full_device_name}", flush=True)

    def _validate_model_structure(self, ov_model: ov.Model):
        """Validate input tensor has 3 channels and 128x128 resolution, output has 2 classes."""
        inputs = ov_model.inputs
        outputs = ov_model.outputs

        if len(inputs) != 1:
            raise ValueError(f"Expected 1 input tensor for MiniFASNet, got {len(inputs)}")
        if len(outputs) != 1:
            raise ValueError(f"Expected 1 output tensor for MiniFASNet, got {len(outputs)}")

        inp_shape = inputs[0].partial_shape
        out_shape = outputs[0].partial_shape

        # Verify channels, height, width (batch dimension can be dynamic)
        if len(inp_shape) != 4 or inp_shape[1] != 3 or inp_shape[2] != 128 or inp_shape[3] != 128:
            raise ValueError(f"Invalid input shape for MiniFASNet: {inp_shape}, expected [-1, 3, 128, 128]")

        if len(out_shape) != 2 or out_shape[1] != 2:
            raise ValueError(f"Invalid output shape for MiniFASNet: {out_shape}, expected [-1, 2]")

    def preprocess(self, face_crop: np.ndarray) -> np.ndarray:
        """
        Preprocess face crop according to MiniFASNet V2 SE specification:
        1. Letterbox resize with reflection border to 128x128.
        2. Convert BGR to RGB.
        3. Normalize to [0.0, 1.0] by dividing by 255.0.
        4. Transpose from HWC to NCHW (1, 3, 128, 128) float32.
        """
        if face_crop is None or face_crop.size == 0:
            face_crop = np.zeros((self.model_img_size, self.model_img_size, 3), dtype=np.uint8)

        old_size = face_crop.shape[:2]
        new_size = self.model_img_size

        if old_size[0] != new_size or old_size[1] != new_size:
            ratio = float(new_size) / max(old_size)
            scaled_shape = (int(round(old_size[1] * ratio)), int(round(old_size[0] * ratio)))

            interpolation = cv2.INTER_LANCZOS4 if ratio > 1.0 else cv2.INTER_AREA
            resized = cv2.resize(face_crop, scaled_shape, interpolation=interpolation)

            delta_w = new_size - scaled_shape[0]
            delta_h = new_size - scaled_shape[1]
            top = delta_h // 2
            bottom = delta_h - top
            left = delta_w // 2
            right = delta_w - left

            padded = cv2.copyMakeBorder(resized, top, bottom, left, right, cv2.BORDER_REFLECT_101)
        else:
            padded = face_crop

        # BGR -> RGB float32 scaled to [0, 1]
        rgb = cv2.cvtColor(padded, cv2.COLOR_BGR2RGB) if padded.shape[2] == 3 else padded
        blob = rgb.transpose(2, 0, 1)[np.newaxis, ...].astype(np.float32) / 255.0
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

        # Numerically stable Softmax across [logit_real, logit_spoof]
        logits = raw_output[:2].astype(np.float64)
        shifted = logits - np.max(logits)
        exp_vals = np.exp(shifted)
        sum_exp = np.sum(exp_vals)

        if sum_exp > 0:
            probs = exp_vals / sum_exp
            p_real = float(probs[0])
            p_spoof = float(probs[1])
        else:
            p_real, p_spoof = 0.5, 0.5

        label = "REAL" if p_real >= p_spoof else "SPOOF"
        return p_real, p_spoof, label, latency_ms
