"""The WhatsApp bridge's JavaScript must parse and its helper module must import.

The bridge's own ``node --test`` suites run only in the JS CI job, so a module that Node cannot load (an
upstream merge once left ``const DELIVERY_STATUS_NAMES`` declared twice in bridge_helpers.js) passed every
Python suite and only failed when a live WhatsApp gateway started the bridge. This test runs in the Python
suite: ``node --check`` on every bridge module, then a real ESM import of the modules that need no
npm dependencies (bridge_helpers.js is imported by bridge.js at startup).
"""
import shutil
import subprocess
from pathlib import Path

import pytest

BRIDGE = Path(__file__).resolve().parents[2] / "scripts" / "whatsapp-bridge"
MODULES = sorted(p.name for p in BRIDGE.glob("*.js"))
# Bridge modules that import only Node built-ins (no node_modules needed), so an import proves they load.
DEPENDENCY_FREE = ["bridge_helpers.js", "outbound_ids.js", "allowlist.js", "owner_message_gate.js"]

NODE = shutil.which("node")
needs_node = pytest.mark.skipif(NODE is None, reason="node is not on PATH (the bridge needs it to run at all)")


def test_the_bridge_modules_are_listed():
    assert "bridge.js" in MODULES and "bridge_helpers.js" in MODULES
    assert set(DEPENDENCY_FREE) <= set(MODULES)


@needs_node
@pytest.mark.parametrize("module", MODULES)
def test_every_bridge_module_parses(module):
    done = subprocess.run([NODE, "--check", str(BRIDGE / module)],
                          capture_output=True, text=True, timeout=60, check=False)
    assert done.returncode == 0, done.stderr[-2000:]


@needs_node
@pytest.mark.parametrize("module", DEPENDENCY_FREE)
def test_dependency_free_bridge_modules_import(module):
    script = (
        "const m = await import(process.argv[1]);"
        "if (Object.keys(m).length === 0) { console.error('no exports'); process.exit(3); }"
        "console.log('ok');"
    )
    done = subprocess.run([NODE, "--input-type=module", "-e", script, (BRIDGE / module).as_uri()],
                          capture_output=True, text=True, timeout=60, cwd=BRIDGE, check=False)
    assert done.returncode == 0 and done.stdout.strip() == "ok", done.stderr[-2000:]


@needs_node
def test_the_two_delivery_status_vocabularies_stay_distinct():
    """The per-send status store reports 3 as 'delivery_ack'; the logged delivery_status events say 'delivered'."""
    script = (
        "const m = await import(process.argv[1]);"
        "const store = m.createDeliveryStatusStore(); store.update('a', 3);"
        "const own = (id) => id === 'b';"
        "const ev = m.deliveryStatusEvent({ key: { fromMe: true, id: 'b', remoteJid: '1@s.whatsapp.net' },"
        "  update: { status: 3, messageTimestamp: 5 }, isOwnSend: own });"
        "console.log(JSON.stringify([m.DELIVERY_STATUS_NAMES[3], store.get('a').status, ev.status]));"
    )
    done = subprocess.run([NODE, "--input-type=module", "-e", script, (BRIDGE / "bridge_helpers.js").as_uri()],
                          capture_output=True, text=True, timeout=60, cwd=BRIDGE, check=False)
    assert done.returncode == 0, done.stderr[-2000:]
    assert done.stdout.strip() == '["delivery_ack","delivery_ack","delivered"]'
