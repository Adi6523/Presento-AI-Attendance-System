"""
Tests for Presento Hardware Detection and Inference Backend Selection.

Validates:
1. CPU Detection: Vendor, model, architecture, core counts, and frequency.
2. GPU Detection & Classification:
   - Intel integrated (Iris Xe, UHD)
   - Intel discrete (Arc A770, etc.)
   - NVIDIA discrete (GeForce RTX, GTX, Laptop GPU)
   - AMD integrated (Radeon 780M, Radeon Graphics)
   - AMD discrete (Radeon RX 6600, etc.)
   - Unknown GPU classification (safely returns 'Unknown', never guesses)
3. Multi-GPU Systems:
   - Intel iGPU + NVIDIA dGPU
   - AMD iGPU + NVIDIA dGPU
4. Inference Backend Selection:
   - Verified Intel GPU -> OpenVINO GPU
   - Unsupported NVIDIA GPU -> Conservative OpenVINO CPU with professional explanation
   - Unsupported AMD GPU -> Conservative OpenVINO CPU with professional explanation
   - Multi-GPU with AMD + NVIDIA -> OpenVINO CPU
   - User-forced CPU (via param or env) -> OpenVINO CPU
   - OpenVINO GPU init failure -> Fallback to CPU
5. Startup Box & Diagnostic Tree Formatting:
   - ASCII-safe, structured, informative, professional.
6. Missing Runtime & Missing Model handling:
   - Graceful degradation when OpenVINO or model files are absent.
"""
from __future__ import annotations

import io
import os
import sys
import unittest
from unittest.mock import MagicMock, patch

from edge.hardware_detector import (
    CPUInfo,
    GPUInfo,
    HardwareReport,
    classify_gpu_type,
    detect_cpu,
    detect_gpus,
    detect_system_hardware,
    _clean_cpu_model,
    _clean_gpu_name,
)
from edge.inference_backend import (
    InferenceDeviceDecision,
    select_inference_device,
    _verify_openvino_gpu,
)
import edge.demo as demo


class TestHardwareDetection(unittest.TestCase):
    """Unit tests for edge/hardware_detector.py."""

    def test_classify_gpu_type_intel_integrated(self):
        """Intel Iris Xe, UHD, and generic Intel Graphics are classified as Integrated."""
        self.assertEqual(classify_gpu_type("Intel(R) Iris(R) Xe Graphics", "Intel"), "Integrated")
        self.assertEqual(classify_gpu_type("Intel(R) UHD Graphics 620", "Intel"), "Integrated")
        self.assertEqual(classify_gpu_type("Intel HD Graphics 530", "Intel Corporation"), "Integrated")

    def test_classify_gpu_type_intel_discrete(self):
        """Intel Arc A-series desktop/discrete GPUs are classified as Discrete."""
        self.assertEqual(classify_gpu_type("Intel(R) Arc(TM) A770 Graphics", "Intel"), "Discrete")
        self.assertEqual(classify_gpu_type("Intel Arc A380 Graphics", "Intel"), "Discrete")

    def test_classify_gpu_type_nvidia_discrete(self):
        """NVIDIA GeForce, RTX, GTX, and Quadro are classified as Discrete."""
        self.assertEqual(classify_gpu_type("NVIDIA GeForce RTX 4060 Laptop GPU", "NVIDIA"), "Discrete")
        self.assertEqual(classify_gpu_type("NVIDIA GeForce GTX 1650", "NVIDIA"), "Discrete")
        self.assertEqual(classify_gpu_type("NVIDIA RTX A2000 Laptop GPU", "NVIDIA"), "Discrete")
        self.assertEqual(classify_gpu_type("Quadro T1000", "NVIDIA"), "Discrete")

    def test_classify_gpu_type_amd_integrated(self):
        """AMD Radeon 780M, 680M, Vega mobile iGPUs are classified as Integrated."""
        self.assertEqual(classify_gpu_type("AMD Radeon 780M Graphics", "AMD"), "Integrated")
        self.assertEqual(classify_gpu_type("AMD Radeon(TM) Graphics", "Advanced Micro Devices, Inc."), "Integrated")
        self.assertEqual(classify_gpu_type("AMD Radeon 680M", "AMD"), "Integrated")

    def test_classify_gpu_type_amd_discrete(self):
        """AMD Radeon RX 6000/7000 dedicated series are classified as Discrete."""
        self.assertEqual(classify_gpu_type("AMD Radeon RX 6700 XT", "AMD"), "Discrete")
        self.assertEqual(classify_gpu_type("Radeon RX 7600M XT", "AMD"), "Discrete")

    def test_classify_gpu_type_unknown(self):
        """Unrecognized display adapters return Unknown rather than guessing."""
        self.assertEqual(classify_gpu_type("Generic Display Controller", "Acme Corp"), "Unknown")
        self.assertEqual(classify_gpu_type("VirtualBox Graphics Adapter", "InnoTek"), "Unknown")
        self.assertEqual(classify_gpu_type("", ""), "Unknown")

    def test_clean_gpu_name_and_cpu_model(self):
        """Validates clean parsing of CPU and GPU brand strings."""
        vendor, clean_cpu, short_cpu = _clean_cpu_model("12th Gen Intel(R) Core(TM) i5-1240P @ 2.11 GHz")
        self.assertEqual(vendor, "Intel")
        self.assertIn("i5-1240P", clean_cpu)

        vendor, clean_gpu, short_gpu = _clean_gpu_name("Intel(R) Iris(R) Xe Graphics")
        self.assertEqual(vendor, "Intel")
        self.assertEqual(clean_gpu, "Intel Iris Xe Graphics")
        self.assertEqual(short_gpu, "Intel Iris Xe")

    def test_detect_cpu_structure(self):
        """detect_cpu returns valid CPUInfo with vendor, model, arch, cores."""
        info = detect_cpu()
        self.assertIsInstance(info, CPUInfo)
        self.assertTrue(len(info.vendor) > 0)
        self.assertTrue(len(info.model) > 0)
        self.assertTrue(len(info.architecture) > 0)
        self.assertGreaterEqual(info.logical_cores, 1)
        self.assertGreaterEqual(info.physical_cores, 1)

    def test_detect_system_hardware_returns_report(self):
        """detect_system_hardware produces a complete HardwareReport."""
        report = detect_system_hardware()
        self.assertIsInstance(report, HardwareReport)
        self.assertIsInstance(report.cpu, CPUInfo)
        self.assertIsInstance(report.gpus, list)


class TestInferenceBackendSelection(unittest.TestCase):
    """Unit tests for edge/inference_backend.py."""

    def setUp(self):
        # Clear OPENVINO_DEVICE environment variable before each test
        os.environ.pop("OPENVINO_DEVICE", None)

    def tearDown(self):
        os.environ.pop("OPENVINO_DEVICE", None)

    def test_forced_cpu_selection(self):
        """When preferred_device='CPU' is passed or OPENVINO_DEVICE='CPU', CPU is selected directly."""
        decision = select_inference_device(preferred_device="CPU")
        self.assertEqual(decision.selected_device, "CPU")
        self.assertEqual(decision.status, "READY")
        self.assertEqual(decision.short_device_name, "CPU")
        self.assertIn("CPU", decision.reason)

    def test_verified_intel_gpu_selection(self):
        """When an Intel GPU is detected and verified by OpenVINO, OpenVINO GPU is selected."""
        fake_cpu = CPUInfo(
            vendor="Intel",
            model="12th Gen Intel Core i5-1240P",
            architecture="x86_64",
            physical_cores=12,
            logical_cores=16,
            frequency_mhz=2100.0,
        )
        fake_gpu = GPUInfo(
            index=0,
            name="Intel(R) Iris(R) Xe Graphics",
            clean_name="Intel Iris Xe",
            vendor="Intel",
            gpu_type="Integrated",
            driver_version="31.0.101.5590",
            adapter_ram_mb=1024,
        )
        fake_report = HardwareReport(
            cpu=fake_cpu,
            gpus=[fake_gpu],
            openvino_available=True,
            openvino_version="2024.0.0",
            openvino_devices=["CPU", "GPU"],
        )

        mock_core = MagicMock()
        mock_core.available_devices = ["CPU", "GPU"]
        mock_core.get_property.return_value = "Intel(R) Iris(R) Xe Graphics (iGPU)"

        with patch("edge.inference_backend.detect_system_hardware", return_value=fake_report), \
             patch("openvino.Core", return_value=mock_core):
            decision = select_inference_device(preferred_device="AUTO")

            self.assertEqual(decision.selected_device, "GPU")
            self.assertEqual(decision.backend_name, "OpenVINO")
            self.assertEqual(decision.status, "VERIFIED")
            self.assertTrue(decision.gpu_verified)
            self.assertEqual(decision.device_type, "Integrated GPU")
            self.assertIn("verified", decision.reason.lower())
            self.assertEqual(decision.model_device_map["SCRFD"], "GPU")
            self.assertEqual(decision.model_device_map["ArcFace"], "GPU")

    def test_nvidia_gpu_conservative_fallback(self):
        """
        When an NVIDIA GPU is detected, Presento does NOT claim NVIDIA acceleration;
        it conservatively falls back to OpenVINO CPU with a professional explanation.
        """
        fake_cpu = CPUInfo(
            vendor="AMD",
            model="AMD Ryzen 7 6800H with Radeon Graphics",
            architecture="x86_64",
            physical_cores=8,
            logical_cores=16,
            frequency_mhz=3200.0,
        )
        fake_gpu = GPUInfo(
            index=0,
            name="NVIDIA GeForce RTX 3060 Laptop GPU",
            clean_name="NVIDIA RTX 3060 Laptop GPU",
            vendor="NVIDIA",
            gpu_type="Discrete",
            driver_version="551.23",
            adapter_ram_mb=6144,
        )
        fake_report = HardwareReport(
            cpu=fake_cpu,
            gpus=[fake_gpu],
            openvino_available=True,
            openvino_version="2024.0.0",
            openvino_devices=["CPU"],
        )

        mock_core = MagicMock()
        mock_core.available_devices = ["CPU"]

        with patch("edge.inference_backend.detect_system_hardware", return_value=fake_report), \
             patch("openvino.Core", return_value=mock_core):
            decision = select_inference_device(preferred_device="AUTO")

            self.assertEqual(decision.selected_device, "CPU")
            self.assertFalse(decision.gpu_verified)
            self.assertEqual(decision.status, "READY")
            self.assertIn("NVIDIA GPU detected", decision.reason)
            self.assertIn("no verified Presento GPU backend", decision.reason)
            self.assertEqual(decision.model_device_map["SCRFD"], "CPU")
            self.assertEqual(decision.model_device_map["ArcFace"], "CPU")

    def test_amd_gpu_conservative_fallback(self):
        """When an AMD GPU is detected, conservative OpenVINO CPU is selected with clear reason."""
        fake_cpu = CPUInfo(
            vendor="AMD",
            model="AMD Ryzen 5 5600G",
            architecture="x86_64",
            physical_cores=6,
            logical_cores=12,
            frequency_mhz=3900.0,
        )
        fake_gpu = GPUInfo(
            index=0,
            name="AMD Radeon Graphics",
            clean_name="AMD Radeon Graphics",
            vendor="AMD",
            gpu_type="Integrated",
            driver_version="30.0.1",
            adapter_ram_mb=512,
        )
        fake_report = HardwareReport(
            cpu=fake_cpu,
            gpus=[fake_gpu],
            openvino_available=True,
            openvino_version="2024.0.0",
            openvino_devices=["CPU"],
        )

        mock_core = MagicMock()
        mock_core.available_devices = ["CPU"]

        with patch("edge.inference_backend.detect_system_hardware", return_value=fake_report), \
             patch("openvino.Core", return_value=mock_core):
            decision = select_inference_device(preferred_device="AUTO")

            self.assertEqual(decision.selected_device, "CPU")
            self.assertFalse(decision.gpu_verified)
            self.assertIn("AMD GPU detected", decision.reason)
            self.assertIn("GPU acceleration not supported", decision.reason)

    def test_multi_gpu_amd_plus_nvidia(self):
        """Dual-GPU laptop with AMD iGPU + NVIDIA dGPU selects OpenVINO CPU without guessing."""
        fake_cpu = CPUInfo(vendor="AMD", model="AMD Ryzen 7 7840HS", architecture="x86_64", physical_cores=8, logical_cores=16)
        gpu0 = GPUInfo(index=0, name="AMD Radeon 780M", clean_name="AMD Radeon 780M", vendor="AMD", gpu_type="Integrated")
        gpu1 = GPUInfo(index=1, name="NVIDIA GeForce RTX 4060 Laptop GPU", clean_name="NVIDIA RTX 4060 Laptop GPU", vendor="NVIDIA", gpu_type="Discrete")

        fake_report = HardwareReport(
            cpu=fake_cpu,
            gpus=[gpu0, gpu1],
            openvino_available=True,
            openvino_version="2024.0.0",
            openvino_devices=["CPU"],
        )

        mock_core = MagicMock()
        mock_core.available_devices = ["CPU"]

        with patch("edge.inference_backend.detect_system_hardware", return_value=fake_report), \
             patch("openvino.Core", return_value=mock_core):
            decision = select_inference_device(preferred_device="AUTO")

            self.assertEqual(decision.selected_device, "CPU")
            self.assertIn("NVIDIA GPU detected", decision.reason)
            # Both GPUs should be in the report
            box = decision.format_startup_box()
            self.assertIn("AMD Radeon 780M", box)
            self.assertIn("NVIDIA RTX 4060 Laptop GPU", box)

    def test_multi_gpu_intel_plus_nvidia(self):
        """Dual-GPU laptop with Intel iGPU + NVIDIA dGPU selects Intel iGPU when OpenVINO verified."""
        fake_cpu = CPUInfo(vendor="Intel", model="Intel Core i7-13700H", architecture="x86_64", physical_cores=14, logical_cores=20)
        gpu0 = GPUInfo(index=0, name="Intel(R) Iris(R) Xe Graphics", clean_name="Intel Iris Xe", vendor="Intel", gpu_type="Integrated")
        gpu1 = GPUInfo(index=1, name="NVIDIA GeForce RTX 4070 Laptop GPU", clean_name="NVIDIA RTX 4070 Laptop GPU", vendor="NVIDIA", gpu_type="Discrete")

        fake_report = HardwareReport(
            cpu=fake_cpu,
            gpus=[gpu0, gpu1],
            openvino_available=True,
            openvino_version="2024.0.0",
            openvino_devices=["CPU", "GPU"],
        )

        mock_core = MagicMock()
        mock_core.available_devices = ["CPU", "GPU"]
        mock_core.get_property.return_value = "Intel(R) Iris(R) Xe Graphics (iGPU)"

        with patch("edge.inference_backend.detect_system_hardware", return_value=fake_report), \
             patch("openvino.Core", return_value=mock_core):
            decision = select_inference_device(preferred_device="AUTO")

            self.assertEqual(decision.selected_device, "GPU")
            self.assertTrue(decision.gpu_verified)
            box = decision.format_startup_box()
            self.assertIn("Intel Iris Xe", box)
            self.assertIn("NVIDIA RTX 4070 Laptop GPU", box)

    def test_intel_gpu_verification_failure_fallback(self):
        """If OpenVINO GPU compilation or initialization check fails, it gracefully falls back to CPU."""
        fake_cpu = CPUInfo(vendor="Intel", model="Intel Core i5-1135G7", architecture="x86_64", physical_cores=4, logical_cores=8)
        fake_gpu = GPUInfo(index=0, name="Intel Iris Xe", clean_name="Intel Iris Xe", vendor="Intel", gpu_type="Integrated")

        fake_report = HardwareReport(
            cpu=fake_cpu,
            gpus=[fake_gpu],
            openvino_available=True,
            openvino_version="2024.0.0",
            openvino_devices=["CPU", "GPU"],
        )

        mock_core = MagicMock()
        mock_core.available_devices = ["CPU", "GPU"]
        # get_property raises exception (e.g. driver missing / OpenCL unavailable)
        mock_core.get_property.side_effect = RuntimeError("OpenCL driver not installed")

        with patch("edge.inference_backend.detect_system_hardware", return_value=fake_report), \
             patch("openvino.Core", return_value=mock_core):
            decision = select_inference_device(preferred_device="AUTO")

            self.assertEqual(decision.selected_device, "CPU")
            self.assertEqual(decision.status, "FALLBACK")
            self.assertIn("falling back to CPU", decision.reason)

    def test_cpu_only_machine(self):
        """Machine with no dedicated GPU detects CPU-only and selects OpenVINO CPU."""
        fake_cpu = CPUInfo(vendor="Intel", model="Intel Core i3-10100", architecture="x86_64", physical_cores=4, logical_cores=8)
        fake_report = HardwareReport(
            cpu=fake_cpu,
            gpus=[],
            openvino_available=True,
            openvino_version="2024.0.0",
            openvino_devices=["CPU"],
        )

        mock_core = MagicMock()
        mock_core.available_devices = ["CPU"]

        with patch("edge.inference_backend.detect_system_hardware", return_value=fake_report), \
             patch("openvino.Core", return_value=mock_core):
            decision = select_inference_device(preferred_device="AUTO")

            self.assertEqual(decision.selected_device, "CPU")
            self.assertEqual(decision.status, "READY")
            self.assertIn("Standard OpenVINO CPU", decision.reason)

    def test_missing_openvino_runtime_handling(self):
        """When OpenVINO runtime is not installed or import fails, it returns a safe CPU decision."""
        fake_cpu = CPUInfo(vendor="Intel", model="Generic Intel CPU", architecture="x86_64", physical_cores=4, logical_cores=4)
        fake_report = HardwareReport(
            cpu=fake_cpu,
            gpus=[],
            openvino_available=False,
            openvino_version=None,
            openvino_devices=[],
        )

        with patch("edge.inference_backend.detect_system_hardware", return_value=fake_report):
            decision = select_inference_device(preferred_device="AUTO")
            self.assertEqual(decision.selected_device, "CPU")
            self.assertEqual(decision.status, "FALLBACK")
            self.assertIn("OpenVINO runtime unavailable", decision.reason)

    def test_format_startup_box_ascii_safe(self):
        """Startup box output must be encodable in standard ASCII / cp1252 without UnicodeEncodeError."""
        fake_cpu = CPUInfo(vendor="Intel", model="Intel Core i5-1240P", architecture="x86_64", physical_cores=12, logical_cores=16, frequency_mhz=2100.0)
        fake_gpu = GPUInfo(index=0, name="Intel Iris Xe", clean_name="Intel Iris Xe", vendor="Intel", gpu_type="Integrated")
        fake_report = HardwareReport(cpu=fake_cpu, gpus=[fake_gpu], openvino_available=True, openvino_version="2024.0.0", openvino_devices=["CPU", "GPU"])

        decision = InferenceDeviceDecision(
            selected_device="GPU",
            backend_name="OpenVINO",
            display_name="Intel Iris Xe Graphics",
            short_device_name="Intel Iris Xe",
            device_type="Integrated GPU",
            status="READY",
            reason="Verified Intel Iris Xe",
            model_device_map={"SCRFD": "GPU", "ArcFace": "GPU", "Anti-Spoof": "GPU", "MiniFASNet": "GPU"},
            gpu_verified=True,
            hardware_report=fake_report,
        )

        box = decision.format_startup_box()
        # Verify it encodes cleanly to ASCII
        encoded = box.encode("ascii", errors="strict")
        self.assertGreater(len(encoded), 100)
        self.assertIn("PRESENTO AI VISION -- HARDWARE ANALYSIS", box)
        self.assertIn("Intel Iris Xe Graphics", box)

    def test_format_diagnostic_tree_ascii_safe(self):
        """Diagnostic tree output must be ASCII safe and display performance metrics."""
        fake_cpu = CPUInfo(vendor="Intel", model="Intel Core i5-1240P", architecture="x86_64", physical_cores=12, logical_cores=16)
        fake_report = HardwareReport(cpu=fake_cpu, gpus=[], openvino_available=True, openvino_version="2024.0.0", openvino_devices=["CPU"])

        decision = InferenceDeviceDecision(
            selected_device="CPU",
            backend_name="OpenVINO",
            display_name="Intel Core i5-1240P",
            short_device_name="CPU",
            device_type="CPU",
            status="READY",
            reason="Standard OpenVINO CPU",
            model_device_map={"SCRFD": "CPU", "ArcFace": "CPU", "Anti-Spoof": "CPU", "MiniFASNet": "CPU"},
            gpu_verified=False,
            hardware_report=fake_report,
        )

        tree = decision.format_diagnostic_tree(display_fps=30.0, det_fps=28.5, arc_fps=25.0, as_lat=32.0)
        encoded = tree.encode("ascii", errors="strict")
        self.assertGreater(len(encoded), 100)
        self.assertIn("PRESENTO HARDWARE & AI RUNTIME DIAGNOSTICS", tree)
        self.assertIn("Display FPS: 30.0", tree)
        self.assertIn("Detection FPS: 28.5", tree)
