"""The WhatsApp bridge's JavaScript must parse, its dependency-free modules must import, and every name bridge.js
imports from a local module must exist.

The bridge's own ``node --test`` suites run only in the JS CI job, so a module that Node cannot load (an
upstream merge once left ``const DELIVERY_STATUS_NAMES`` declared twice in bridge_helpers.js) passed every
Python suite and only failed when a live WhatsApp gateway started the bridge. This test runs in the Python
suite. It needs ``node``: without it the bridge cannot run at all, so a missing node fails here, never skips.
"""
import json
import re
import shutil
import subprocess
from pathlib import Path

import pytest

BRIDGE = Path(__file__).resolve().parents[2] / "scripts" / "whatsapp-bridge"
MODULES = sorted(p.name for p in BRIDGE.glob("*.js"))
IMPORT_FROM = re.compile(r"^import\s+(?P<what>[^'\";]*?)\s*from\s*['\"](?P<src>[^'\"]+)['\"]",
                         re.MULTILINE | re.DOTALL)
BUILTINS = {"path", "fs", "crypto", "util", "os", "url", "events", "child_process", "http", "https", "net",
            "stream", "zlib", "buffer", "timers", "readline", "assert", "string_decoder", "fs/promises"}


def imports(module):
    return [(m["what"], m["src"]) for m in IMPORT_FROM.finditer((BRIDGE / module).read_text())]


def dependency_free(module):
    """Imports only Node built-ins and other dependency-free local modules (no node_modules needed)."""
    for _, src in imports(module):
        if src.startswith("./"):
            if not dependency_free(src[2:]):
                return False
        elif src.removeprefix("node:") not in BUILTINS:
            return False
    return True


DEPENDENCY_FREE = [m for m in MODULES if dependency_free(m)]


def node():
    path = shutil.which("node")
    assert path, "node is not on PATH: the WhatsApp bridge cannot run without it"
    return path


def run_module(script, module):
    return subprocess.run([node(), "--input-type=module", "-e", script, (BRIDGE / module).as_uri()],
                          capture_output=True, text=True, timeout=60, cwd=BRIDGE, check=False)


def test_the_bridge_modules_are_found():
    assert "bridge.js" in MODULES and "bridge_helpers.js" in MODULES
    assert "bridge_helpers.js" in DEPENDENCY_FREE and "bridge.js" not in DEPENDENCY_FREE   # bridge.js needs Baileys


@pytest.mark.parametrize("module", MODULES)
def test_every_bridge_module_parses(module):
    done = subprocess.run([node(), "--check", str(BRIDGE / module)],
                          capture_output=True, text=True, timeout=60, check=False)
    assert done.returncode == 0, done.stderr[-2000:]


@pytest.mark.parametrize("module", DEPENDENCY_FREE)
def test_dependency_free_bridge_modules_import(module):
    done = run_module("const m = await import(process.argv[1]);"
                      "if (Object.keys(m).length === 0) { console.error('no exports'); process.exit(3); }"
                      "console.log('ok');", module)
    assert done.returncode == 0 and done.stdout.strip() == "ok", done.stderr[-2000:]


@pytest.mark.parametrize("module", MODULES)
def test_every_name_imported_from_a_local_module_is_exported(module):
    """node --check does not resolve imports: a name a module no longer exports only fails at live startup."""
    for what, src in imports(module):
        if not src.startswith("./") or src[2:] not in DEPENDENCY_FREE:
            continue
        names = re.search(r"\{([^}]*)\}", what)
        wanted = [n.split(" as ")[0].strip() for n in names.group(1).split(",") if n.strip()] if names else []
        done = run_module("const m = await import(process.argv[1]); console.log(JSON.stringify(Object.keys(m)));",
                          src[2:])
        assert done.returncode == 0, done.stderr[-2000:]
        missing = sorted(set(wanted) - set(json.loads(done.stdout)))
        assert not missing, f"{module} imports {missing} from {src}, which does not export them"


def test_the_two_delivery_status_vocabularies_stay_distinct():
    """The per-send status store reports 3 as 'delivery_ack'; the logged delivery_status events say 'delivered'."""
    done = run_module(
        "const m = await import(process.argv[1]);"
        "const store = m.createDeliveryStatusStore(); store.update('a', 3);"
        "const own = (id) => id === 'b';"
        "const ev = m.deliveryStatusEvent({ key: { fromMe: true, id: 'b', remoteJid: '1@s.whatsapp.net' },"
        "  update: { status: 3, messageTimestamp: 5 }, isOwnSend: own });"
        "console.log(JSON.stringify([m.DELIVERY_STATUS_NAMES[3], store.get('a').status, ev.status]));",
        "bridge_helpers.js")
    assert done.returncode == 0, done.stderr[-2000:]
    assert done.stdout.strip() == '["delivery_ack","delivery_ack","delivered"]'
