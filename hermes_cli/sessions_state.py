"""``hermes sessions state``: a session's execution state on its owner, and an explicit interrupt.

Fork (launch adapter). Under single-gateway ownership a ``hermes chat`` process is a client: killing it
detaches, and the turn keeps running on the gateway. A spawner (helpdesk asks and requests, Switchboard
intake) that times out or needs capacity must therefore ask the owner, not the process table:

- ``hermes sessions state <id|title>`` prints ``{session_id, running, pending}`` as JSON;
- ``--interrupt`` sends ``session.interrupt`` for the running generation, and ``--wait N`` then polls up
  to N seconds until the owner reports it stopped.

The session is resolved as ``hermes chat --resume`` / ``-c`` resolve it (exact id first, then title).
"""
from __future__ import annotations

import asyncio
import json
import sys
import time

_POLL_SECONDS = 0.5


def _state(snapshot) -> dict:
    pending = [
        {key: row.get(key) for key in ("admission_id", "status", "outcome")}
        for row in snapshot.get("pending") or []
    ]
    return {
        "session_id": snapshot.get("stored_session_id") or snapshot.get("session_id"),
        "running": bool(snapshot.get("running")),
        "pending": pending,
    }


async def _run(args):
    from hermes_cli.sessions_discard import _resume
    from hermes_cli.gateway_client import connect_gateway
    async with connect_gateway() as client:
        snapshot = await _resume(client, args.session)
        out = _state(snapshot)
        if args.interrupt:
            out["interrupted"] = False
            if out["running"]:
                runtime_id = snapshot.get("session_id") or snapshot["stored_session_id"]
                result = await client.rpc("session.interrupt", session_id=runtime_id) or {}
                out["interrupted"] = result.get("status") == "interrupted" or bool(result.get("interrupted"))
                deadline = time.monotonic() + max(0.0, float(args.wait or 0))
                while out["running"] and time.monotonic() < deadline:
                    await asyncio.sleep(_POLL_SECONDS)
                    out.update(_state(await _resume(client, out["session_id"])))
        print(json.dumps(out, sort_keys=True))
        return 0


def add_state_parser(sessions_subparsers) -> None:
    parser = sessions_subparsers.add_parser(
        "state", help="Show whether a session is running on its gateway (JSON); optionally interrupt it",
        description="Prints {session_id, running, pending} for one session as its gateway owns it. A "
            "`hermes chat` client that exits or is killed only detaches: the turn keeps running. "
            "Use --interrupt to stop the running turn, and --wait to wait until it has stopped.")
    parser.add_argument("session", help="Session ID or title")
    parser.add_argument("--interrupt", action="store_true", help="Interrupt the running turn")
    parser.add_argument("--wait", type=float, default=0.0, metavar="SECONDS",
        help="With --interrupt: poll up to this long until the session is no longer running")


def cmd_state(args) -> int:
    from websockets.exceptions import WebSocketException
    from hermes_cli.gateway_client import GatewayClientError
    try:
        return asyncio.run(_run(args))
    except (GatewayClientError, OSError, TimeoutError, WebSocketException) as exc:
        message = str(exc) if isinstance(exc, GatewayClientError) else "Gateway connection/read failed"
        print("Error: " + message, file=sys.stderr)
        return 1
