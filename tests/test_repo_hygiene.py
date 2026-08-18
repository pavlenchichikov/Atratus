"""Two things that only break in CI, checked on the machine that can fix them.

Both of these have shipped a red build already. Neither is about the code being
wrong: they are about the difference between this checkout, which has local
tools and an older linter, and CI's, which has neither.
"""

import ast
import os
import subprocess

import pytest

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WORKFLOW = os.path.join(BASE, ".github", "workflows", "ci.yml")
PRECOMMIT = os.path.join(BASE, ".pre-commit-config.yaml")


def _git(*args):
    """Tracked paths from git, or None when this is not a usable checkout."""
    try:
        out = subprocess.run(["git", "-C", BASE, *args], capture_output=True,
                             text=True, encoding="utf-8", timeout=60,
                             check=False)
    except (OSError, subprocess.SubprocessError):
        return None
    if out.returncode != 0:
        return None
    return [ln.strip() for ln in out.stdout.splitlines() if ln.strip()]


def test_a_committed_test_never_imports_an_untracked_module():
    """A test for a gitignored tool has to be gitignored too.

    CI checks out the test without the module, and the import fails collection
    for the WHOLE suite, not just that file - one local helper takes every test
    down with it. The .gitignore already says this next to the ab_* companions;
    this is the same rule with a way to notice it before the push.

    In CI the untracked file is simply absent, so this passes trivially there.
    It is written for the checkout that has the file and can still act.
    """
    tracked = _git("ls-files")
    if tracked is None:
        pytest.skip("not a git checkout")
    tracked_set = set(tracked)
    # Repo-root modules only: those are the ones a test imports by bare name.
    local_modules = {f[:-3] for f in os.listdir(BASE)
                     if f.endswith(".py") and os.path.isfile(os.path.join(BASE, f))}
    untracked_modules = {m for m in local_modules if m + ".py" not in tracked_set}

    offences = []
    for rel in tracked:
        if not (rel.startswith("tests/") and rel.endswith(".py")):
            continue
        path = os.path.join(BASE, rel)
        try:
            tree = ast.parse(open(path, encoding="utf-8").read())
        except (OSError, SyntaxError):
            continue
        for node in ast.walk(tree):
            names = []
            if isinstance(node, ast.Import):
                names = [a.name.split(".")[0] for a in node.names]
            elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
                names = [node.module.split(".")[0]]
            for name in names:
                if name in untracked_modules:
                    offences.append("%s:%d imports %s" % (rel, node.lineno, name))

    assert not offences, (
        "these committed tests import modules that are not in Git, so CI fails "
        "collection: %s. Either commit the module or gitignore the test beside "
        "it, as .gitignore does for the ab_* companions." % "; ".join(offences))


def test_the_commit_hook_lints_with_the_version_ci_lints_with():
    """A lint rule set is a version, not a tool.

    ruff 0.12 and 0.16 disagree about real code - how an import block is sorted,
    which rules are on by default - so a green run on the wrong version proves
    nothing about the build. That is not hypothetical: the hook used to take
    whatever ruff was on PATH, this machine had 0.12, CI pinned 0.16, and an
    unsorted import block went through the gate and failed the build.

    Both pins are deliberate, so this only keeps them the same number.
    """
    ci = pre = None
    for line in open(WORKFLOW, encoding="utf-8"):
        if "pip install ruff==" in line:
            ci = line.split("ruff==")[1].strip()
            break
    for line in open(PRECOMMIT, encoding="utf-8"):
        if line.strip().startswith("rev:"):
            pre = line.split("rev:")[1].strip().lstrip("v")
            break
    if not ci or not pre:
        pytest.skip("one of the two files does not pin ruff")
    assert ci == pre, (
        "ci.yml lints with ruff %s and the commit hook with %s: bump both "
        "together, or the gate stops predicting the build" % (ci, pre))
