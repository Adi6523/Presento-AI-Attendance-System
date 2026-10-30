"""
Presento AI Vision — Portable Hardware Detection Subsystem.

Robustly discovers system architecture, CPU characteristics, and all available
integrated / discrete GPUs on any Windows laptop without requiring proprietary
GPU SDKs or crashing on unknown hardware.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import json
import logging
import os
import platform
import re
import subprocess
import sys
from typing import List, Optional

logger = logging.getLogger("presento.edge.hardware_detector")


@dataclass(frozen=True)
class CPUInfo:
    vendor: str
    model: str
    architecture: str
    physical_cores: int
    logical_cores: int
    short_model: str = ""
    frequency_mhz: Optional[float] = None

    def __post_init__(self):
        if not self.short_model:
            # Automatic fallback for short model
            object.__setattr__(self, "short_model", self.model[:22].strip())


@dataclass(frozen=True)
class GPUInfo:
    index: int
    name: str
    clean_name: str
    vendor: str
    gpu_type: str  # "Integrated", "Discrete", "External", or "Unknown"
    short_name: str = ""
    driver_version: Optional[str] = None
    adapter_ram_mb: Optional[int] = None
    openvino_supported: bool = False

    def __post_init__(self):
        if not self.short_name:
            object.__setattr__(self, "short_name", self.clean_name[:22].strip())

    @property
    def is_integrated(self) -> bool:
        return self.gpu_type == "Integrated"

    @property
    def is_discrete(self) -> bool:
        return self.gpu_type == "Discrete"

    @property
    def is_external(self) -> bool:
        return self.gpu_type == "External"


@dataclass
class HardwareReport:
    cpu: CPUInfo
    gpus: List[GPUInfo] = field(default_factory=list)
    primary_gpu: Optional[GPUInfo] = None
    openvino_available: bool = True
    openvino_devices: List[str] = field(default_factory=list)
    openvino_version: str = "Unknown"
    os_info: str = "Windows"

    def get_compact_hud_labels(self) -> tuple[str, str]:
        """
        Returns (cpu_hud_str, gpu_hud_str) for display on the top HUD bar.
        Example: ("Intel i5-1240P", "Intel Iris Xe (iGPU)")
        """
        cpu_str = self.cpu.short_model
        if not self.gpus:
            gpu_str = "None (CPU Only)"
        else:
            primary = self.primary_gpu or self.gpus[0]
            type_abbr = "iGPU" if primary.gpu_type == "Integrated" else ("dGPU" if primary.gpu_type == "Discrete" else primary.gpu_type)
            gpu_str = f"{primary.short_name} ({type_abbr})"
        return cpu_str, gpu_str


def _clean_string(val: Optional[str]) -> str:
    if not val:
        return ""
    return re.sub(r"\s+", " ", val).strip()


def _clean_cpu_model(raw: str) -> tuple[str, str, str]:
    """
    Parses vendor, clean model name, and short model name from raw CPU string.
    Example:
        '12th Gen Intel(R) Core(TM) i5-1240P' -> ('Intel', 'Intel Core i5-1240P', 'Intel i5-1240P')
        'AMD Ryzen 7 5800H with Radeon Graphics' -> ('AMD', 'AMD Ryzen 7 5800H', 'Ryzen 7 5800H')
    """
    raw_clean = _clean_string(raw)
    vendor = "Unknown"
    if "intel" in raw_clean.lower():
        vendor = "Intel"
    elif "amd" in raw_clean.lower():
        vendor = "AMD"
    elif "qualcomm" in raw_clean.lower() or "snapdragon" in raw_clean.lower() or "arm" in raw_clean.lower():
        vendor = "ARM"
    elif "apple" in raw_clean.lower():
        vendor = "Apple"

    # Clean trademark symbols and trailing text
    model = raw_clean.replace("(R)", "").replace("(TM)", "").replace("@", "").strip()
    model = re.sub(r"with.*Graphics.*$", "", model, flags=re.IGNORECASE).strip()
    model = re.sub(r"\d+\s*-\s*Core\s+Processor", "", model, flags=re.IGNORECASE).strip()
    model = re.sub(r"\b\d+(\.\d+)?\s*GHz\b", "", model, flags=re.IGNORECASE).strip()
    model = re.sub(r"^\d+(st|nd|rd|th)\s+Gen\s+", "", model, flags=re.IGNORECASE).strip()
    model = _clean_string(model)

    # Short model for HUD
    short = model
    if "Intel" in short and "Core" in short:
        # e.g. "Intel Core i5-1240P" -> "Intel i5-1240P"
        short = short.replace("Core ", "")
    elif "AMD" in short and "Ryzen" in short:
        # e.g. "AMD Ryzen 7 5800H" -> "Ryzen 7 5800H"
        short = short.replace("AMD ", "")

    if len(short) > 22:
        short = short[:20] + ".."

    return vendor, model, short


def classify_gpu_type(name: str, vendor: str) -> str:
    """
    Conservatively classifies GPU as Integrated, Discrete, External, or Unknown.
    Does NOT guess if criteria are ambiguous.
    """
    if not name and not vendor:
        return "Unknown"

    lower = name.lower()
    lower_clean = lower.replace("(r)", "").replace("(tm)", "")
    v_clean = vendor.strip().lower()

    if any(k in lower for k in ["external", "egpu", "thunderbolt gpu"]):
        return "External"

    is_intel = "intel" in v_clean or "intel" in lower
    is_nvidia = "nvidia" in v_clean or any(k in lower for k in ["geforce", "rtx", "gtx", "quadro"])
    is_amd = "amd" in v_clean or "advanced micro" in v_clean or "radeon" in lower

    if is_intel:
        if any(k in lower_clean for k in ["arc a7", "arc a5", "arc a3"]):
            return "Discrete"
        if any(k in lower_clean for k in ["iris", "uhd", "hd graphics", "intel graphics", "integrated"]):
            return "Integrated"
        if "arc" in lower_clean and any(k in lower_clean for k in ["graphics", "mobile"]):
            return "Integrated"
        return "Unknown"

    if is_nvidia:
        # Virtually all desktop/laptop NVIDIA GPUs are discrete
        return "Discrete"

    if is_amd:
        if any(k in lower_clean for k in ["rx ", "rx7", "rx6", "radeon pro", "firepro", "discrete"]):
            return "Discrete"
        if any(k in lower_clean for k in ["radeon", "vega", "780m", "680m", "660m", "integrated"]):
            return "Integrated"
        return "Unknown"

    return "Unknown"


def _clean_gpu_name(raw: str) -> tuple[str, str, str]:
    """
    Parses vendor, clean full name, and short name from raw GPU string.
    Example:
        'Intel(R) Iris(R) Xe Graphics' -> ('Intel', 'Intel Iris Xe Graphics', 'Intel Iris Xe')
        'NVIDIA GeForce RTX 4060 Laptop GPU' -> ('NVIDIA', 'NVIDIA GeForce RTX 4060 Laptop GPU', 'RTX 4060 Laptop')
    """
    raw_clean = _clean_string(raw)
    vendor = "Unknown"
    lower = raw_clean.lower()
    if "intel" in lower:
        vendor = "Intel"
    elif "nvidia" in lower or "geforce" in lower or "quadro" in lower or "rtx" in lower:
        vendor = "NVIDIA"
    elif "amd" in lower or "radeon" in lower:
        vendor = "AMD"

    clean = raw_clean.replace("(R)", "").replace("(TM)", "").replace("(iGPU)", "").strip()
    clean = _clean_string(clean)

    # Short name for HUD
    short = clean
    short = short.replace("Graphics", "").strip()
    if "GeForce " in short:
        short = short.replace("GeForce ", "")
    if "NVIDIA " in short and "RTX" in short:
        short = short.replace("NVIDIA ", "")
    if "AMD " in short and "Radeon" in short:
        short = short.replace("AMD ", "")

    short = _clean_string(short)
    if len(short) > 22:
        short = short[:20] + ".."

    return vendor, clean, short


def detect_cpu() -> CPUInfo:
    """Discovers CPU vendor, model, architecture, cores, and frequency."""
    vendor = "Unknown"
    model = platform.processor() or "Unknown CPU"
    arch = platform.machine() or "x86_64"
    if arch.lower() in ("amd64", "x86_64", "em64t"):
        arch = "x86_64"

    # Query Windows Registry for clean hardware processor brand string
    if sys.platform.startswith("win"):
        try:
            import winreg
            with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r"HARDWARE\DESCRIPTION\System\CentralProcessor\0") as key:
                reg_name = winreg.QueryValueEx(key, "ProcessorNameString")[0]
                if reg_name:
                    model = str(reg_name).strip()
        except Exception as exc:
            logger.debug("Could not read CPU from registry: %s", exc)

    vendor, clean_model, short_model = _clean_cpu_model(model)

    # Cores and frequency
    logical_cores = os.cpu_count() or 4
    physical_cores = logical_cores
    freq_mhz = None

    try:
        import psutil
        p_cores = psutil.cpu_count(logical=False)
        if p_cores:
            physical_cores = p_cores
        l_cores = psutil.cpu_count(logical=True)
        if l_cores:
            logical_cores = l_cores
        cfreq = psutil.cpu_freq()
        if cfreq and cfreq.current:
            freq_mhz = round(float(cfreq.current), 1)
    except Exception as exc:
        logger.debug("psutil cpu info query failed: %s", exc)

    return CPUInfo(
        vendor=vendor,
        model=clean_model,
        short_model=short_model,
        architecture=arch,
        physical_cores=physical_cores,
        logical_cores=logical_cores,
        frequency_mhz=freq_mhz,
    )


def _detect_gpus_powershell() -> List[dict]:
    """Query Windows display controllers using PowerShell CIM query."""
    if not sys.platform.startswith("win"):
        return []

    cmd = [
        "powershell",
        "-NoProfile",
        "-NonInteractive",
        "-Command",
        "Get-CimInstance Win32_VideoController | Select-Object Name, AdapterRAM, DriverVersion | ConvertTo-Json -Compress",
    ]
    try:
        res = subprocess.run(cmd, capture_output=True, text=True, timeout=4.0)
        if res.returncode == 0 and res.stdout.strip():
            data = json.loads(res.stdout.strip())
            if isinstance(data, dict):
                return [data]
            elif isinstance(data, list):
                return [d for d in data if isinstance(d, dict)]
    except Exception as exc:
        logger.debug("PowerShell GPU detection failed: %s", exc)
    return []


def _detect_gpus_registry() -> List[dict]:
    """Fallback display adapter discovery via Windows Registry."""
    if not sys.platform.startswith("win"):
        return []

    results = []
    base = r"SYSTEM\CurrentControlSet\Control\Class\{4d36e968-e325-11ce-bfc1-08002be10318}"
    try:
        import winreg
        with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, base) as root:
            idx = 0
            while True:
                try:
                    sub = winreg.EnumKey(root, idx)
                    idx += 1
                    if sub.isdigit():
                        with winreg.OpenKey(root, sub) as sk:
                            try:
                                desc = winreg.QueryValueEx(sk, "DriverDesc")[0]
                                ver = None
                                try:
                                    ver = winreg.QueryValueEx(sk, "DriverVersion")[0]
                                except Exception:
                                    pass
                                if desc and "mirror" not in desc.lower() and "remote" not in desc.lower():
                                    results.append({"Name": str(desc), "DriverVersion": str(ver) if ver else None})
                            except Exception:
                                pass
                except OSError:
                    break
    except Exception as exc:
        logger.debug("Registry GPU query failed: %s", exc)
    return results


def detect_gpus(openvino_available_devices: Optional[List[str]] = None) -> List[GPUInfo]:
    """Discovers and classifies all GPUs present on the system."""
    raw_gpus: List[dict] = _detect_gpus_powershell()
    if not raw_gpus:
        raw_gpus = _detect_gpus_registry()

    gpus: List[GPUInfo] = []
    seen_names = set()

    for idx, item in enumerate(raw_gpus):
        name = item.get("Name")
        if not name or name in seen_names:
            continue
        # Skip virtual/remote desktop display drivers
        if any(v in name.lower() for v in ["virtual display", "remote desktop", "parsec", "spacedesk", "citrix"]):
            continue

        seen_names.add(name)
        driver = item.get("DriverVersion")
        adapter_ram = item.get("AdapterRAM")
        ram_mb = int(adapter_ram // (1024 * 1024)) if isinstance(adapter_ram, (int, float)) and adapter_ram > 0 else None

        vendor, clean_name, short_name = _clean_gpu_name(name)
        gpu_type = classify_gpu_type(clean_name, vendor)

        # Cross-reference with OpenVINO device compatibility
        ov_supported = False
        if vendor == "Intel" and openvino_available_devices and "GPU" in openvino_available_devices:
            ov_supported = True

        gpus.append(
            GPUInfo(
                index=len(gpus),
                name=name,
                clean_name=clean_name,
                short_name=short_name,
                vendor=vendor,
                gpu_type=gpu_type,
                driver_version=driver,
                adapter_ram_mb=ram_mb,
                openvino_supported=ov_supported,
            )
        )

    # If OS detection found nothing, check OpenVINO Core GPU exposure as last resort
    if not gpus and openvino_available_devices and "GPU" in openvino_available_devices:
        try:
            import openvino as ov
            core = ov.Core()
            full_dev_name = core.get_property("GPU", "FULL_DEVICE_NAME")
            vendor, clean_name, short_name = _clean_gpu_name(full_dev_name)
            gpu_type = classify_gpu_type(clean_name, vendor)
            gpus.append(
                GPUInfo(
                    index=0,
                    name=full_dev_name,
                    clean_name=clean_name,
                    short_name=short_name,
                    vendor=vendor,
                    gpu_type=gpu_type,
                    openvino_supported=True,
                )
            )
        except Exception:
            pass

    return gpus


_CACHED_REPORT: Optional[HardwareReport] = None


def detect_system_hardware(force_refresh: bool = False) -> HardwareReport:
    """
    Main hardware detection entry point.
    Discovers CPU, GPUs, and OpenVINO runtime status.
    Results are cached for zero overhead on subsequent calls.
    """
    global _CACHED_REPORT
    if _CACHED_REPORT is not None and not force_refresh:
        return _CACHED_REPORT

    cpu_info = detect_cpu()

    # Query OpenVINO
    ov_devices: List[str] = []
    ov_version = "Unknown"
    try:
        import openvino as ov
        ov_version = getattr(ov, "__version__", "Unknown").split("-")[0]
        core = ov.Core()
        ov_devices = list(core.available_devices)
    except Exception as exc:
        logger.warning("Could not query OpenVINO runtime: %s", exc)

    gpus = detect_gpus(openvino_available_devices=ov_devices)

    # Primary GPU selection (prefer discrete if present, otherwise integrated)
    primary = None
    if gpus:
        discrete = [g for g in gpus if g.gpu_type == "Discrete"]
        if discrete:
            primary = discrete[0]
        else:
            primary = gpus[0]

    os_str = f"{platform.system()} {platform.release()}"

    report = HardwareReport(
        cpu=cpu_info,
        gpus=gpus,
        primary_gpu=primary,
        openvino_devices=ov_devices,
        openvino_version=ov_version,
        os_info=os_str,
    )
    _CACHED_REPORT = report
    return report
