"""
Presento AI Vision — Inference Backend Selection Layer.

Implements intelligent, conservative hardware/backend selection:
1. Detect Hardware & GPU Vendors
2. Detect Available AI Runtimes (OpenVINO)
3. Test Supported Devices (Intel iGPU / dGPU)
4. Gracefully Fall Back for Unsupported GPUs (NVIDIA, AMD)
5. Select Best Verified Device
6. Format Startup Hardware Reports & Diagnostic Trees
"""
from __future__ import annotations

from dataclasses import dataclass, field
import logging
import os
import sys
from typing import Dict, List, Optional, Tuple

from .config import settings
from .hardware_detector import CPUInfo, GPUInfo, HardwareReport, detect_system_hardware

logger = logging.getLogger("presento.edge.inference_backend")


@dataclass
class InferenceDeviceDecision:
    selected_device: str  # "CPU" or "GPU"
    backend_name: str  # "OpenVINO"
    display_name: str  # e.g. "Intel Iris Xe Graphics" or "12th Gen Intel Core i5-1240P"
    short_device_name: str  # e.g. "Intel Iris Xe" or "CPU"
    device_type: str  # "Integrated GPU", "Discrete GPU", or "CPU"
    status: str  # "READY", "VERIFIED", or "FALLBACK"
    reason: str  # Clear explanation of device selection
    model_device_map: Dict[str, str] = field(default_factory=dict)
    gpu_verified: bool = False
    hardware_report: Optional[HardwareReport] = None

    def get_hud_label(self) -> str:
        """Formatted string for top HUD bar."""
        if self.selected_device == "GPU":
            return f"OpenVINO GPU ({self.short_device_name})"
        return "OpenVINO CPU"

    def format_startup_box(self) -> str:
        """Produces the polished hardware analysis box for terminal launch."""
        lines = [
            "============================================================",
            " PRESENTO AI VISION -- HARDWARE ANALYSIS",
            "============================================================",
            "",
            " CPU",
            " ------------------------------------------------------------",
        ]
        if self.hardware_report:
            c = self.hardware_report.cpu
            freq_str = f" @ {c.frequency_mhz:.0f} MHz" if c.frequency_mhz else ""
            lines.extend([
                f" Vendor       : {c.vendor}",
                f" Model        : {c.model}{freq_str}",
                f" Architecture : {c.architecture}",
                f" Cores        : {c.physical_cores} physical / {c.logical_cores} logical",
            ])
        else:
            lines.append(" CPU Info     : Standard x86_64")

        lines.extend([
            "",
            " GPU",
            " ------------------------------------------------------------",
        ])

        if self.hardware_report and self.hardware_report.gpus:
            for g in self.hardware_report.gpus:
                stat = "VERIFIED" if (self.selected_device == "GPU" and g.vendor == "Intel") else "DETECTED"
                ram_str = f" ({g.adapter_ram_mb} MB VRAM)" if g.adapter_ram_mb else ""
                lines.extend([
                    f" GPU {g.index}        : {g.clean_name}{ram_str}",
                    f" Type         : {g.gpu_type}",
                    f" Vendor       : {g.vendor}",
                ])
                if g.driver_version:
                    lines.append(f" Driver       : {g.driver_version}")
                lines.append(f" Status       : {stat}")
        else:
            lines.extend([
                " GPU          : None detected (Standard Display Adapter)",
                " Type         : N/A",
                " Status       : CPU Fallback Active",
            ])

        lines.extend([
            "",
            " PRESENTO INFERENCE",
            " ------------------------------------------------------------",
            f" Selected     : {self.display_name}",
            f" Backend      : {self.backend_name}",
            f" Device       : {self.selected_device}",
            f" Status       : {self.status}",
        ])
        if self.reason:
            lines.append(f" Reason       : {self.reason}")

        lines.extend([
            "",
            " AI PIPELINE",
            " ------------------------------------------------------------",
        ])
        for model_name, dev in self.model_device_map.items():
            lines.append(f" {model_name:<12} : {dev}")

        lines.extend([
            "",
            "============================================================",
            "",
        ])
        return "\n".join(lines)

    def format_diagnostic_tree(
        self,
        display_fps: float = 0.0,
        det_fps: float = 0.0,
        arc_fps: float = 0.0,
        as_lat: float = 0.0,
    ) -> str:
        """Produces the hierarchical diagnostic tree for the [D] console command."""
        ov_ver = self.hardware_report.openvino_version if self.hardware_report else "Unknown"
        lines = [
            "",
            "============================================================",
            " PRESENTO HARDWARE & AI RUNTIME DIAGNOSTICS",
            "============================================================",
            "",
            "CPU",
        ]
        if self.hardware_report:
            c = self.hardware_report.cpu
            lines.extend([
                f"+-- Vendor: {c.vendor}",
                f"+-- Model: {c.model}",
                f"+-- Architecture: {c.architecture}",
                f"+-- Physical cores: {c.physical_cores}",
                f"\\-- Logical processors: {c.logical_cores}",
            ])
        else:
            lines.append("\\-- Info: Generic CPU")

        lines.append("")
        lines.append("GPU")
        if self.hardware_report and self.hardware_report.gpus:
            for i, g in enumerate(self.hardware_report.gpus):
                is_last_gpu = (i == len(self.hardware_report.gpus) - 1)
                prefix = "\\--" if is_last_gpu else "+--"
                sub_p = "    " if is_last_gpu else "|   "
                lines.extend([
                    f"{prefix} GPU {g.index}: {g.clean_name}",
                    f"{sub_p}+-- Vendor: {g.vendor}",
                    f"{sub_p}+-- Type: {g.gpu_type}",
                    f"{sub_p}+-- Driver: {g.driver_version or 'N/A'}",
                    f"{sub_p}\\-- Runtime status: {'OpenVINO Accelerated' if (self.selected_device == 'GPU' and g.vendor == 'Intel') else 'Detected (Inference on CPU)'}",
                ])
        else:
            lines.append("\\-- No dedicated GPU controllers detected")

        lines.extend([
            "",
            "AI RUNTIME",
            f"+-- OpenVINO version: {ov_ver}",
            f"+-- Selected device: {self.selected_device}",
            f"+-- Device type: {self.device_type}",
            f"\\-- Backend verification: {self.status} - {self.reason}",
            "",
            "MODEL EXECUTION",
            f"+-- SCRFD: OpenVINO {self.model_device_map.get('SCRFD', self.selected_device)}",
            f"+-- ArcFace: OpenVINO {self.model_device_map.get('ArcFace', self.selected_device)}",
            f"+-- MN3: OpenVINO {self.model_device_map.get('Anti-Spoof', self.selected_device)}",
            f"\\-- MiniFASNet: OpenVINO {self.model_device_map.get('MiniFASNet', self.selected_device)}",
            "",
            "PERFORMANCE",
            f"+-- Display FPS: {display_fps:.1f}",
            f"+-- Detection FPS: {det_fps:.1f}",
            f"+-- ArcFace FPS: {arc_fps:.1f}",
            f"\\-- Anti-Spoof latency: {as_lat:.1f}ms",
            "============================================================",
            "",
        ])
        return "\n".join(lines)


def _verify_openvino_gpu(core) -> Tuple[bool, str]:
    """
    Verifies that OpenVINO can query and initialize the Intel GPU.
    Returns (is_verified, device_full_name).
    """
    try:
        if "GPU" not in core.available_devices:
            return False, "GPU device not present in OpenVINO runtime"
        full_name = core.get_property("GPU", "FULL_DEVICE_NAME")
        return True, str(full_name)
    except Exception as exc:
        logger.warning("OpenVINO GPU verification failed: %s", exc)
        return False, str(exc)


def select_inference_device(preferred_device: Optional[str] = None) -> InferenceDeviceDecision:
    """
    Intelligently selects the best verified inference device for Presento.

    Rules:
    1. If user explicitly requests CPU (via preferred_device or OPENVINO_DEVICE=CPU):
       -> OpenVINO CPU
    2. If Intel GPU is detected and verified by OpenVINO:
       -> OpenVINO GPU (with automatic fallback to CPU if initialization fails)
    3. If NVIDIA GPU is detected:
       -> Presento NVIDIA acceleration not configured; conservatively select OpenVINO CPU.
    4. If AMD GPU is detected:
       -> Presento AMD GPU acceleration not supported; conservatively select OpenVINO CPU.
    5. CPU-only systems:
       -> OpenVINO CPU
    """
    report = detect_system_hardware()

    # Determine requested preference
    req = (
        preferred_device
        or os.getenv("OPENVINO_DEVICE")
        or getattr(settings, "openvino_device", "AUTO")
    ).strip().upper()

    default_model_map_cpu = {
        "SCRFD": "CPU",
        "ArcFace": "CPU",
        "Anti-Spoof": "CPU",
        "MiniFASNet": "CPU",
    }
    default_model_map_gpu = {
        "SCRFD": "GPU",
        "ArcFace": "GPU",
        "Anti-Spoof": "GPU",
        "MiniFASNet": "GPU",
    }

    # User explicitly requested CPU
    if req == "CPU":
        return InferenceDeviceDecision(
            selected_device="CPU",
            backend_name="OpenVINO",
            display_name=report.cpu.model,
            short_device_name="CPU",
            device_type="CPU",
            status="READY",
            reason="User configured OPENVINO_DEVICE=CPU",
            model_device_map=default_model_map_cpu,
            gpu_verified=False,
            hardware_report=report,
        )

    if not getattr(report, "openvino_available", True):
        return InferenceDeviceDecision(
            selected_device="CPU",
            backend_name="OpenVINO",
            display_name=report.cpu.model,
            short_device_name="CPU",
            device_type="CPU",
            status="FALLBACK",
            reason="OpenVINO runtime unavailable; using CPU",
            model_device_map=default_model_map_cpu,
            gpu_verified=False,
            hardware_report=report,
        )

    # Initialize OpenVINO Core
    core = None
    ov_gpu_available = False
    try:
        import openvino as ov
        core = ov.Core()
        ov_gpu_available = "GPU" in core.available_devices
    except Exception as exc:
        logger.warning("Could not initialize OpenVINO Core: %s. Using CPU.", exc)

    # Inspect detected GPUs
    intel_gpus = [g for g in report.gpus if g.vendor == "Intel"]
    nvidia_gpus = [g for g in report.gpus if g.vendor == "NVIDIA"]
    amd_gpus = [g for g in report.gpus if g.vendor == "AMD"]

    # Candidate: Intel GPU + OpenVINO GPU support
    if (req in ("GPU", "AUTO") or not req) and intel_gpus and ov_gpu_available and core is not None:
        verified, full_name = _verify_openvino_gpu(core)
        if verified:
            primary_intel = intel_gpus[0]
            short_label = primary_intel.short_name
            dev_type = f"{primary_intel.gpu_type} GPU" if primary_intel.gpu_type != "Unknown" else "GPU"
            return InferenceDeviceDecision(
                selected_device="GPU",
                backend_name="OpenVINO",
                display_name=primary_intel.clean_name,
                short_device_name=short_label,
                device_type=dev_type,
                status="VERIFIED",
                reason="Intel GPU verified with native OpenVINO acceleration",
                model_device_map=default_model_map_gpu,
                gpu_verified=True,
                hardware_report=report,
            )
        else:
            logger.warning("Intel GPU detected but OpenVINO verification failed: %s. Falling back to CPU.", full_name)
            return InferenceDeviceDecision(
                selected_device="CPU",
                backend_name="OpenVINO",
                display_name=report.cpu.model,
                short_device_name="CPU",
                device_type="CPU",
                status="FALLBACK",
                reason=f"Intel GPU initialization failed ({full_name}); safely falling back to CPU",
                model_device_map=default_model_map_cpu,
                gpu_verified=False,
                hardware_report=report,
            )

    # Scenario: NVIDIA GPU detected without verified OpenVINO GPU runtime
    if nvidia_gpus:
        nv_name = nvidia_gpus[0].clean_name
        return InferenceDeviceDecision(
            selected_device="CPU",
            backend_name="OpenVINO",
            display_name=report.cpu.model,
            short_device_name="CPU",
            device_type="CPU",
            status="READY",
            reason=f"NVIDIA GPU detected ({nv_name}); no verified Presento GPU backend configured",
            model_device_map=default_model_map_cpu,
            gpu_verified=False,
            hardware_report=report,
        )

    # Scenario: AMD GPU detected without verified OpenVINO GPU runtime
    if amd_gpus:
        amd_name = amd_gpus[0].clean_name
        return InferenceDeviceDecision(
            selected_device="CPU",
            backend_name="OpenVINO",
            display_name=report.cpu.model,
            short_device_name="CPU",
            device_type="CPU",
            status="READY",
            reason=f"AMD GPU detected ({amd_name}); GPU acceleration not supported in current OpenVINO runtime",
            model_device_map=default_model_map_cpu,
            gpu_verified=False,
            hardware_report=report,
        )

    # Scenario: GPU explicitly requested but no GPU found or exposed
    if req == "GPU":
        return InferenceDeviceDecision(
            selected_device="CPU",
            backend_name="OpenVINO",
            display_name=report.cpu.model,
            short_device_name="CPU",
            device_type="CPU",
            status="FALLBACK",
            reason="GPU requested but no supported OpenVINO GPU found; safely falling back to CPU",
            model_device_map=default_model_map_cpu,
            gpu_verified=False,
            hardware_report=report,
        )

    # Standard CPU system
    return InferenceDeviceDecision(
        selected_device="CPU",
        backend_name="OpenVINO",
        display_name=report.cpu.model,
        short_device_name="CPU",
        device_type="CPU",
        status="READY",
        reason="Standard OpenVINO CPU inference",
        model_device_map=default_model_map_cpu,
        gpu_verified=False,
        hardware_report=report,
    )
