"""A numbered model list for the launcher menus, so nobody has to type an exact id.

    python -m core.model_picker list PROVIDER
    python -m core.model_picker pick ANSWER OUTFILE [DEFAULT]

`list` prints the models PROVIDER offers, numbered, and remembers them. The .bat
then reads the answer itself with `set /p` (a Python input() on a redirected
stdin swallowed the rest of the menu's input), and `pick` turns it into an id:
a number picks from the list, other text is taken as the id, empty keeps
DEFAULT. The id goes to OUTFILE for `set /p VAR=<OUTFILE`. A listing that
cannot be fetched leaves typing open.
"""

import json
import os
import sys
import tempfile

# Claude Code takes its aliases or a full id; the API takes full ids only.
CLAUDE_CODE = ["sonnet", "opus", "haiku", "claude-opus-5-5", "claude-sonnet-5",
               "claude-haiku-4-5-20251001", "claude-fable-5-1"]
ANTHROPIC = ["claude-opus-5-5", "claude-sonnet-5", "claude-haiku-4-5-20251001",
             "claude-fable-5-1", "claude-opus-4-8"]
OPENAI = ["gpt-4o", "gpt-4o-mini"]
CLOUD_TAGS = "https://ollama.com/api/tags"
LIST_PATH = os.path.join(tempfile.gettempdir(), "gt_models.json")


def _ollama_local():
    from core.llm_proposer import list_ollama_models

    return list_ollama_models()


def _ollama_cloud():
    import net  # owns the VPN routing; a direct socket does not resolve here

    key = os.getenv("OLLAMA_API_KEY") or ""
    r = net.http_get(CLOUD_TAGS, headers={"Authorization": "Bearer " + key} if key else None,
                     retries=1)
    r.raise_for_status()
    return sorted(m["name"] for m in r.json().get("models", []) if m.get("name"))


def models_for(provider):
    """The ids PROVIDER offers, or [] when they cannot be listed."""
    static = {"claude-code": CLAUDE_CODE, "anthropic": ANTHROPIC, "openai": OPENAI}
    if provider in static:
        return list(static[provider])
    fetch = {"ollama": _ollama_local, "ollama-cloud": _ollama_cloud}.get(provider)
    if fetch is None:
        return []
    try:
        return fetch()
    except Exception as exc:
        print("    (cannot list %s models: %s)" % (provider, str(exc)[:120]))
        return []


def choose(models, answer, default):
    """A number picks from `models`, other text is taken as the id, Enter keeps
    `default`; a number out of range also keeps the default."""
    answer = (answer or "").strip()
    if not answer:
        return default.strip()
    if answer.isdigit():
        i = int(answer)
        return models[i - 1] if 1 <= i <= len(models) else default.strip()
    return answer


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    if len(argv) >= 2 and argv[0] == "list":
        models = models_for(argv[1])
        with open(LIST_PATH, "w", encoding="utf-8") as fh:
            json.dump(models, fh)
        for i, m in enumerate(models, 1):
            print("    [%d] %s" % (i, m))
        return 0
    if len(argv) >= 3 and argv[0] == "pick":
        answer, outfile = argv[1], argv[2]
        default = argv[3] if len(argv) > 3 else ""
        try:
            with open(LIST_PATH, encoding="utf-8") as fh:
                models = json.load(fh)
        except (OSError, ValueError):
            models = []
        with open(outfile, "w", encoding="utf-8") as fh:
            fh.write(choose(models, answer, default))
        return 0
    print(__doc__)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
