"""
JARVIS TOOLS MODULE
Provides: web search, Python code execution, system info, app launcher, file manager
"""

import subprocess
import sys
import os
import psutil
import datetime
import traceback
import tempfile

# ─── WEB SEARCH ──────────────────────────────────────────────────────────────
def search_web(query: str, max_results: int = 5) -> str:
    """Search DuckDuckGo and return formatted results."""
    try:
        from duckduckgo_search import DDGS
        results = []
        with DDGS() as ddgs:
            for r in ddgs.text(query, max_results=max_results):
                results.append(f"• {r['title']}\n  {r['body']}\n  Source: {r['href']}")
        if results:
            return "\n\n".join(results)
        return "No results found."
    except Exception as e:
        return f"Web search failed: {e}"


# ─── PYTHON CODE SOLVER ───────────────────────────────────────────────────────
BLOCKED_KEYWORDS = [
    "os.remove", "shutil.rmtree", "os.rmdir", "format(", "subprocess.call",
    "os.system(\"del", "os.system(\"rm", "os.system(\"format",
    "__import__('os').system", "eval(", "exec("
]

def run_python_code(code: str, timeout: int = 10) -> str:
    """Safely execute Python code in a subprocess and return output."""
    # Safety check
    for kw in BLOCKED_KEYWORDS:
        if kw in code:
            return f"⛔ Blocked: '{kw}' is not permitted for safety reasons."

    # Write to a temp file
    try:
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py',
                                        delete=False, encoding='utf-8') as f:
            f.write(code)
            tmp_path = f.name

        result = subprocess.run(
            [sys.executable, tmp_path],
            capture_output=True, text=True, timeout=timeout
        )
        output = result.stdout.strip()
        errors = result.stderr.strip()

        if output and errors:
            return f"Output:\n{output}\n\nErrors:\n{errors}"
        elif output:
            return f"Output:\n{output}"
        elif errors:
            return f"Error:\n{errors}"
        else:
            return "Code ran successfully with no output."
    except subprocess.TimeoutExpired:
        return "⚠️ Code execution timed out (10 second limit)."
    except Exception as e:
        return f"Execution failed: {traceback.format_exc()}"
    finally:
        try:
            os.remove(tmp_path)
        except Exception:
            pass


# ─── SYSTEM INFORMATION ───────────────────────────────────────────────────────
def get_system_info() -> str:
    """Return a summary of current system resource usage."""
    lines = []

    # Battery
    battery = psutil.sensors_battery()
    if battery:
        charging = "Charging ⚡" if battery.power_plugged else "On Battery 🔋"
        lines.append(f"Battery: {battery.percent:.0f}% ({charging})")
        if not battery.power_plugged and battery.secsleft > 0:
            mins = battery.secsleft // 60
            lines.append(f"Time remaining: {mins // 60}h {mins % 60}m")

    # CPU
    cpu = psutil.cpu_percent(interval=1)
    lines.append(f"CPU Usage: {cpu}%")

    # RAM
    ram = psutil.virtual_memory()
    lines.append(f"RAM: {ram.percent}% used ({ram.used // (1024**3):.1f} GB / {ram.total // (1024**3):.1f} GB)")

    # Disk
    disk = psutil.disk_usage('C:\\')
    lines.append(f"Disk (C:): {disk.percent}% used ({disk.free // (1024**3):.0f} GB free of {disk.total // (1024**3):.0f} GB)")

    # Time
    now = datetime.datetime.now()
    lines.append(f"Current Time: {now.strftime('%I:%M %p, %A %d %B %Y')}")

    return "\n".join(lines)


# ─── APP LAUNCHER ─────────────────────────────────────────────────────────────
APP_MAP = {
    "notepad":       "notepad.exe",
    "calculator":    "calc.exe",
    "paint":         "mspaint.exe",
    "task manager":  "taskmgr.exe",
    "file explorer": "explorer.exe",
    "explorer":      "explorer.exe",
    "word":          "winword.exe",
    "excel":         "excel.exe",
    "powerpoint":    "powerpnt.exe",
    "chrome":        "chrome.exe",
    "firefox":       "firefox.exe",
    "edge":          "msedge.exe",
    "vs code":       "code.exe",
    "vscode":        "code.exe",
    "visual studio code": "code.exe",
    "cmd":           "cmd.exe",
    "command prompt":"cmd.exe",
    "powershell":    "powershell.exe",
    "spotify":       "spotify.exe",
    "vlc":           "vlc.exe",
    "discord":       "discord.exe",
    "whatsapp":      "whatsapp.exe",
    "zoom":          "zoom.exe",
    "teams":         "teams.exe",
    "settings":      "ms-settings:",
    "control panel": "control.exe",
    "snipping tool": "snippingtool.exe",
}

def open_app(name: str) -> str:
    """Open an application by name."""
    name_clean = name.lower().strip()
    for key, exe in APP_MAP.items():
        if key in name_clean:
            try:
                if exe.startswith("ms-"):
                    os.startfile(exe)
                else:
                    subprocess.Popen(exe, shell=True)
                return f"✅ Opening {key.title()}, Sir."
            except Exception as e:
                return f"❌ Failed to open {key}: {e}"

    # Fallback: try running the name directly
    try:
        subprocess.Popen(name_clean, shell=True)
        return f"✅ Attempting to open '{name_clean}', Sir."
    except Exception:
        return f"❌ I couldn't find '{name}' on your system, Sir."


# ─── FILE MANAGER ─────────────────────────────────────────────────────────────
SAFE_ROOT = os.path.expanduser("~\\Documents")

def _safe_path(path: str) -> str:
    """Resolve path relative to Documents if not absolute."""
    if not os.path.isabs(path):
        return os.path.join(SAFE_ROOT, path)
    return path

def manage_files(action: str, path: str = "", content: str = "") -> str:
    """
    Perform file operations.
    action: 'read' | 'create' | 'list' | 'delete'
    """
    path = _safe_path(path)
    action = action.lower()

    if action == "list":
        target = path if path else SAFE_ROOT
        try:
            items = os.listdir(target)
            if not items:
                return f"📂 '{target}' is empty."
            return f"📂 Contents of {target}:\n" + "\n".join(f"  • {i}" for i in items[:30])
        except Exception as e:
            return f"❌ Cannot list '{target}': {e}"

    elif action == "read":
        try:
            with open(path, 'r', encoding='utf-8') as f:
                text = f.read(3000)
            return f"📄 Content of '{os.path.basename(path)}':\n{text}"
        except Exception as e:
            return f"❌ Cannot read '{path}': {e}"

    elif action == "create":
        try:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(path, 'w', encoding='utf-8') as f:
                f.write(content)
            return f"✅ File created: '{path}'"
        except Exception as e:
            return f"❌ Cannot create file: {e}"

    elif action == "delete":
        # Refuse to delete outside Documents
        if not path.startswith(SAFE_ROOT):
            return "⛔ I can only delete files inside your Documents folder, Sir."
        try:
            os.remove(path)
            return f"🗑️ Deleted '{os.path.basename(path)}'"
        except Exception as e:
            return f"❌ Cannot delete: {e}"

    return f"❓ Unknown file action: '{action}'"
