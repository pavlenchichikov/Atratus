"""Every knob for local Ollama models, one table for the code and the menu.

    python -m core.ollama_settings show
    python -m core.ollama_settings set NUMBER_OR_KEY VALUE      (empty VALUE = default)

The same rules for every local model, no per-model cases: the code reads a
value with get(), the [AN] Models [G] menu lists this table and writes it.

Two scopes. "env" keys live in .env and are read on every call. "server" keys
are read by Ollama and the llama-server it starts, from the user environment
(HKCU), so changing one restarts Ollama; a call in flight fails and is retried.
"""

import os
import subprocess
import sys

ENV_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".env")


def _int(lo=0):
    return lambda v: v.lstrip("-").isdigit() and int(v) >= lo


def _float(v):
    try:
        return float(v) > 0
    except ValueError:
        return False


def _one_of(*vals):
    return lambda v: v in vals


# key, scope, default ("" = Ollama decides), check, description
SETTINGS = (
    ("GTRADE_OLLAMA_KEEP_LOADED", "env", "1", _one_of("0", "1"),
     ("1 = model stays loaded in RAM on the CPU between calls (no reload); "
      "0 = load for each call, unload after it")),
    ("GTRADE_OLLAMA_NUM_GPU", "env", "", _int(),
     ("layers on the GPU when keep loaded is 0, only while nothing trains; "
      "empty = Ollama decides")),
    ("GTRADE_OLLAMA_MIN_FREE_MB", "env", "3500", _int(1),
     "free VRAM in MB before a model may put layers on the GPU"),
    ("GTRADE_OLLAMA_NUM_CTX", "env", "", _int(1024),
     ("context in tokens per call (KV memory grows with it); empty = Ollama's "
      "own (32768 here). Analyst reasoning needs ~24000, wiki ~30000")),
    ("GTRADE_OLLAMA_MMAP", "env", "", _one_of("0", "1"),
     ("1 = weights mapped from the file (Windows may page them out, slower "
      "under pressure); 0 = copied into RAM; empty = Ollama decides (copy on CPU)")),
    ("GTRADE_OLLAMA_RAM_SHARE", "env", "1.0", _float,
     ("share of the model file that must be free in RAM before a load; "
      "below 1 lets a model run partly from the page file")),
    ("GTRADE_OLLAMA_RAM_MARGIN_MB", "env", "2048", _int(),
     "RAM in MB that must stay free on top of the model"),
    ("GTRADE_OLLAMA_RAM_WAIT", "env", "600", _int(),
     "seconds a load waits for that RAM before it is refused"),
    ("GTRADE_TRAIN_RAM_MB", "env", "4000", _int(),
     "free RAM in MB a starting trainer needs; below it a loaded model is unloaded"),
    ("OLLAMA_KV_CACHE_TYPE", "server", "", _one_of("f16", "q8_0", "q4_0"),
     ("KV cache precision: q8_0 = half of f16 (needs flash attention), q4_0 = a "
      "quarter; empty = f16")),
    ("OLLAMA_FLASH_ATTENTION", "server", "", _one_of("0", "1"),
     "flash attention; 1 is required for a q8_0/q4_0 KV cache"),
    ("LLAMA_ARG_CACHE_RAM", "server", "", _int(),
     ("prompt cache in MB beside a loaded model, 0 = off; same answers, only "
      "repeated prompts get slower; empty = 8192")),
    ("LLAMA_ARG_N_CPU_MOE", "server", "", _int(),
     ("MoE models: expert weights of the first N layers stay in RAM, the rest go "
      "to the GPU with the model's layers; gemma 26b IQ3_S at 64k ctx: 27 = "
      "3.8 of 4 GB VRAM, 12 tok/s; empty = Ollama decides; no effect on dense models")),
    ("LLAMA_ARG_THREADS", "server", "", _int(1),
     ("CPU threads per model; 4 = the P-cores only, 15 tok/s vs 12 at the default 6 "
      "on gemma 26b; empty = llama-server's own")),
)
_BY_KEY = {s[0]: s for s in SETTINGS}


def default(key):
    return _BY_KEY[key][2]


def get(key):
    """The value in force for this process: its environment, else the default."""
    v = (os.getenv(key) or "").strip()
    return v if v else default(key)


def _saved(key, scope):
    if scope == "server":
        try:
            import winreg

            with winreg.OpenKey(winreg.HKEY_CURRENT_USER, "Environment") as k:
                return str(winreg.QueryValueEx(k, key)[0])
        except OSError:
            return ""
    from dotenv import dotenv_values

    return (dotenv_values(ENV_PATH).get(key) or "") if os.path.exists(ENV_PATH) else ""


def show():
    for i, (key, scope, dflt, _check, desc) in enumerate(SETTINGS, 1):
        cur = _saved(key, scope)
        print("  [%2d] %-28s = %-6s (default %s)%s" % (
            i, key, cur or "-", dflt or "Ollama's",
            "  *Ollama restarts" if scope == "server" else ""))
        print("        %s" % desc)


def _key_of(which):
    which = str(which).strip()
    if which.isdigit() and 1 <= int(which) <= len(SETTINGS):
        return SETTINGS[int(which) - 1][0]
    return which if which in _BY_KEY else None


def _set_server(key, value):
    import winreg

    with winreg.OpenKey(winreg.HKEY_CURRENT_USER, "Environment", 0, winreg.KEY_SET_VALUE) as k:
        if value:
            winreg.SetValueEx(k, key, 0, winreg.REG_SZ, value)
        else:
            try:
                winreg.DeleteValue(k, key)
            except OSError:
                pass


def restart_ollama():
    """Restart the Ollama tray app with the HKCU values. It inherits this
    process's environment, so the saved server keys are copied in first."""
    for key, scope, *_ in SETTINGS:
        if scope == "server":
            v = _saved(key, scope)
            if v:
                os.environ[key] = v
            else:
                os.environ.pop(key, None)
    for image in ("ollama app.exe", "ollama.exe", "llama-server.exe"):
        subprocess.run(["taskkill", "/im", image, "/f"], capture_output=True, check=False)
    app = os.path.join(os.environ.get("LOCALAPPDATA", ""), "Programs", "Ollama", "ollama app.exe")
    if os.path.exists(app):
        subprocess.Popen([app], close_fds=True, creationflags=getattr(subprocess, "DETACHED_PROCESS", 0))
        return True
    return False


def set_value(which, value):
    """Save one setting; empty value = back to the default. Returns a message."""
    key = _key_of(which)
    if key is None:
        raise ValueError("no setting %r" % which)
    _k, scope, _d, check, _desc = _BY_KEY[key]
    value = str(value).strip()
    if value and not check(value):
        raise ValueError("invalid value %r for %s" % (value, key))
    if scope == "server":
        _set_server(key, value)
        restarted = restart_ollama()
        return "%s = %s; Ollama %s" % (key, value or "default",
                                       "restarted" if restarted else "not found, restart it yourself")
    from dotenv import set_key, unset_key

    if value:
        set_key(ENV_PATH, key, value)
        os.environ[key] = value
    else:
        if _saved(key, scope):
            unset_key(ENV_PATH, key)
        os.environ.pop(key, None)
    return "%s = %s in .env" % (key, value or "default")


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    if argv[:1] == ["show"]:
        show()
        return 0
    if argv[:1] == ["set"] and len(argv) >= 2:
        try:
            print("  " + set_value(argv[1], argv[2] if len(argv) > 2 else ""))
        except ValueError as exc:
            print("  refused: %s" % exc)
            return 1
        return 0
    print(__doc__)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
