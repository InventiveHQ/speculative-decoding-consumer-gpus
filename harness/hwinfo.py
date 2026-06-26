"""Best-effort, cross-platform hardware capture for community submissions.

The whole point of this benchmark series is comparing real hardware, so every
submission should carry the rig it ran on. `collect()` returns a JSON-serializable
dict describing the machine; `summary()` returns a one-line human string.

Design rules:
  * Never raises. Every probe is wrapped; a missing field is omitted or null.
  * No third-party deps — only the stdlib + whatever CLI tools the OS ships
    (nvidia-smi, powershell, sysctl, lsblk). Missing tools just yield blanks.
  * Backward compatible: keeps the original keys (platform, python, cpu, gpus)
    so existing aggregators keep working, and *adds* cpu_cores, ram,
    motherboard, and storage.

Windows/Linux/macOS are all handled. The richest data comes from Windows
(CIM) and Linux (sysfs/procfs); macOS captures the essentials via sysctl.
"""
import json
import os
import platform
import re
import subprocess

_WIN = platform.system() == "Windows"
_LINUX = platform.system() == "Linux"
_MAC = platform.system() == "Darwin"

# SMBIOS memory-type codes -> friendly name (the ones you'll actually see)
_DDR = {20: "DDR", 21: "DDR2", 24: "DDR3", 26: "DDR4", 34: "DDR5"}


def _run(cmd, timeout=20):
    """Run a command, return stdout (str) or "" on any failure."""
    try:
        return subprocess.check_output(
            cmd, text=True, stderr=subprocess.DEVNULL, timeout=timeout)
    except Exception:
        return ""


def _ps(script, timeout=20):
    """Run a PowerShell snippet, return stdout or ""."""
    return _run(["powershell", "-NoProfile", "-NonInteractive", "-Command", script], timeout)


def _ps_json(script, timeout=20):
    """Run a PowerShell snippet that emits JSON; return the parsed value or None."""
    out = _ps(script, timeout)
    if not out.strip():
        return None
    try:
        return json.loads(out)
    except Exception:
        return None


# --------------------------------------------------------------------------- CPU
def cpu_name():
    try:
        if _WIN:
            out = _ps("(Get-CimInstance Win32_Processor).Name")
            if out.strip():
                return out.strip().splitlines()[0].strip()
        elif _LINUX:
            for line in open("/proc/cpuinfo"):
                if line.lower().startswith("model name"):
                    return line.split(":", 1)[1].strip()
        elif _MAC:
            v = _run(["sysctl", "-n", "machdep.cpu.brand_string"]).strip()
            if v:
                return v
    except Exception:
        pass
    return platform.processor() or platform.machine()


def cpu_cores():
    """Return {'physical': int|None, 'logical': int|None}."""
    logical = os.cpu_count()
    physical = None
    try:
        if _WIN:
            out = _ps("(Get-CimInstance Win32_Processor | "
                      "Measure-Object -Property NumberOfCores -Sum).Sum")
            if out.strip().isdigit():
                physical = int(out.strip())
        elif _LINUX:
            pairs = set()
            phys = core = None
            for line in open("/proc/cpuinfo"):
                if line.startswith("physical id"):
                    phys = line.split(":", 1)[1].strip()
                elif line.startswith("core id"):
                    core = line.split(":", 1)[1].strip()
                elif not line.strip():
                    if phys is not None and core is not None:
                        pairs.add((phys, core))
                    phys = core = None
            if pairs:
                physical = len(pairs)
        elif _MAC:
            out = _run(["sysctl", "-n", "hw.physicalcpu"]).strip()
            if out.isdigit():
                physical = int(out)
    except Exception:
        pass
    return {"physical": physical, "logical": logical}


# --------------------------------------------------------------------------- RAM
def ram_info():
    """Return {'total_gb', 'speed_mhz', 'modules', 'type'} (any may be None)."""
    info = {"total_gb": None, "speed_mhz": None, "modules": None, "type": None}
    try:
        if _WIN:
            d = _ps_json(
                "$m=Get-CimInstance Win32_PhysicalMemory;"
                "[pscustomobject]@{"
                "total=($m|Measure-Object -Property Capacity -Sum).Sum;"
                "speed=($m|Select-Object -First 1).ConfiguredClockSpeed;"
                "modules=@($m).Count;"
                "smbios=($m|Select-Object -First 1).SMBIOSMemoryType}|ConvertTo-Json")
            if d:
                if d.get("total"):
                    info["total_gb"] = round(int(d["total"]) / (1024 ** 3))
                info["speed_mhz"] = d.get("speed") or None
                info["modules"] = d.get("modules") or None
                info["type"] = _DDR.get(d.get("smbios"))
        elif _LINUX:
            for line in open("/proc/meminfo"):
                if line.startswith("MemTotal"):
                    kb = int(re.search(r"\d+", line).group())
                    info["total_gb"] = round(kb / (1024 ** 2))
                    break
        elif _MAC:
            out = _run(["sysctl", "-n", "hw.memsize"]).strip()
            if out.isdigit():
                info["total_gb"] = round(int(out) / (1024 ** 3))
    except Exception:
        pass
    return info


# ------------------------------------------------------------------- motherboard
def motherboard():
    """Return {'vendor', 'model'} or None."""
    try:
        if _WIN:
            d = _ps_json("$b=Get-CimInstance Win32_BaseBoard;"
                         "[pscustomobject]@{vendor=$b.Manufacturer;model=$b.Product}|ConvertTo-Json")
            if d and (d.get("vendor") or d.get("model")):
                return {"vendor": (d.get("vendor") or "").strip() or None,
                        "model": (d.get("model") or "").strip() or None}
        elif _LINUX:
            def rd(p):
                try:
                    return open(p).read().strip()
                except Exception:
                    return ""
            vendor = rd("/sys/class/dmi/id/board_vendor")
            model = rd("/sys/class/dmi/id/board_name")
            if vendor or model:
                return {"vendor": vendor or None, "model": model or None}
        elif _MAC:
            model = _run(["sysctl", "-n", "hw.model"]).strip()
            if model:
                return {"vendor": "Apple", "model": model}
    except Exception:
        pass
    return None


# -------------------------------------------------------------------------- GPUs
def _smi_base():
    out = _run(["nvidia-smi",
                "--query-gpu=name,memory.total,driver_version,compute_cap",
                "--format=csv,noheader"])
    gpus = []
    for line in out.strip().splitlines():
        p = [x.strip() for x in line.split(",")]
        if len(p) >= 4:
            gpus.append({"name": p[0], "memory": p[1], "driver": p[2], "compute_cap": p[3]})
    return gpus


def _smi_pcie():
    """PCIe link details, parallel to _smi_base order. Empty if unsupported."""
    out = _run(["nvidia-smi",
                "--query-gpu=pcie.link.gen.max,pcie.link.gen.gpucurrent,"
                "pcie.link.gen.hostmax,pcie.link.width.current,pcie.link.width.max",
                "--format=csv,noheader"])
    rows = []
    for line in out.strip().splitlines():
        p = [x.strip() for x in line.split(",")]
        if len(p) >= 5:
            def i(x):
                m = re.search(r"\d+", x)
                return int(m.group()) if m else None
            rows.append({"gen_max": i(p[0]), "gen_current": i(p[1]),
                         "gen_host_max": i(p[2]), "width_current": i(p[3]),
                         "width_max": i(p[4])})
    return rows


def gpus():
    """NVIDIA GPUs (with PCIe link), plus Apple Silicon GPU on macOS."""
    base = _smi_base()
    if base:
        pcie = _smi_pcie()
        for i, g in enumerate(base):
            if i < len(pcie):
                g["pcie"] = pcie[i]
        return base
    if _MAC:
        out = _run(["system_profiler", "SPDisplaysDataType"])
        m = re.search(r"Chipset Model:\s*(.+)", out)
        if m:
            return [{"name": m.group(1).strip(), "type": "integrated (Metal)"}]
    return []


# ----------------------------------------------------------------------- storage
def storage():
    """Best-effort list of drive descriptions, e.g. 'Samsung SSD 860 (SATA)'."""
    drives = []
    try:
        if _WIN:
            d = _ps_json(
                "Get-CimInstance -Namespace root\\Microsoft\\Windows\\Storage "
                "MSFT_PhysicalDisk | ForEach-Object { [pscustomobject]@{"
                "name=$_.FriendlyName; bus=$_.BusType; media=$_.MediaType} } | ConvertTo-Json")
            if d:
                if isinstance(d, dict):
                    d = [d]
                bus = {11: "SATA", 17: "NVMe", 7: "USB", 8: "RAID", 9: "iSCSI", 10: "SAS"}
                media = {3: "HDD", 4: "SSD"}
                for x in d:
                    name = (x.get("name") or "").strip()
                    if not name:
                        continue
                    tags = [t for t in (bus.get(x.get("bus")), media.get(x.get("media"))) if t]
                    drives.append(f"{name}" + (f" ({', '.join(tags)})" if tags else ""))
            if not drives:  # fallback: plain disk models
                out = _ps("Get-CimInstance Win32_DiskDrive | "
                          "ForEach-Object { $_.Model }")
                drives = [l.strip() for l in out.splitlines() if l.strip()]
        elif _LINUX:
            out = _run(["lsblk", "-dn", "-o", "MODEL,TRAN,ROTA"])
            for line in out.strip().splitlines():
                parts = line.split()
                if not parts:
                    continue
                rota = parts[-1] if parts[-1] in ("0", "1") else None
                tran = parts[-2] if len(parts) >= 2 else ""
                model = " ".join(parts[:-2]) if len(parts) >= 2 else parts[0]
                tags = [t for t in (tran.upper() if tran else None,
                                    {"0": "SSD", "1": "HDD"}.get(rota)) if t]
                if model:
                    drives.append(model + (f" ({', '.join(tags)})" if tags else ""))
        # macOS storage left out: system_profiler output is verbose and low value here.
    except Exception:
        pass
    return drives


# --------------------------------------------------------------------------- API
def collect():
    """Full hardware snapshot. JSON-serializable; never raises."""
    snap = {
        "platform": platform.platform(),
        "python": platform.python_version(),
        "cpu": cpu_name(),               # kept as a string for backward compat
        "cpu_cores": cpu_cores(),
        "ram": ram_info(),
        "gpus": gpus(),
    }
    mb = motherboard()
    if mb:
        snap["motherboard"] = mb
    st = storage()
    if st:
        snap["storage"] = st
    return snap


def summary(snap=None):
    """One-line human summary, e.g. for console feedback."""
    s = snap or collect()
    bits = []
    cores = s.get("cpu_cores") or {}
    cc = ""
    if cores.get("physical") and cores.get("logical"):
        cc = f" ({cores['physical']}c/{cores['logical']}t)"
    bits.append((s.get("cpu") or "CPU?") + cc)
    ram = s.get("ram") or {}
    if ram.get("total_gb"):
        r = f"{ram['total_gb']}GB"
        if ram.get("type"):
            r += f" {ram['type']}"
        if ram.get("speed_mhz"):
            r += f"-{ram['speed_mhz']}"
        bits.append(r)
    for g in (s.get("gpus") or [])[:2]:
        g1 = g.get("name", "GPU")
        p = g.get("pcie") or {}
        if p.get("gen_host_max") and p.get("width_current"):
            g1 += f" (PCIe {p['gen_host_max']}.0 x{p['width_current']})"
        bits.append(g1)
    bits.append(platform.system())
    return " · ".join(bits)


if __name__ == "__main__":
    snap = collect()
    print(json.dumps(snap, indent=2))
    print("\n" + summary(snap))
