# Trusted-local run environment

This explicitly operator-allowlisted POSIX local foreground facility is not a sandbox or a malicious same-UID daemonization containment boundary. Default is denied. Only trusted commands may receive credentials. No static credential fallback is supported.

The selected profile must allow each supplied key in `gateway.api_server.run_environment_allowlist`. Supported names are PAPERCLIP_API_KEY, PAPERCLIP_API_URL, PAPERCLIP_RUN_ID, PAPERCLIP_AGENT_ID, PAPERCLIP_COMPANY_ID, PAPERCLIP_TASK_ID and PAPERCLIP_WAKE_REASON. The paired sender supplies the first five and optional task/wake fields. Ordinary terminal approval policy still applies. This source change does not configure a profile.

Authenticate to GET /v1/capabilities using the receiver profile API key. Require features.run_environment enabled=true, protocol=trusted-local-foreground-v1, route=/v1/trusted-local-runs and sessions=fresh-only. POST only that route with environment and required_environment_capability=trusted-local-foreground-v1. The legacy /v1/runs route cannot accept an environment. Legacy receivers do not have the new route, preventing silent unknown-field execution after a preflight/hot-swap race. Idempotent replay retains the acknowledgement.

Credential submissions cannot select a session, response continuation, conversation history, hosted room or X-Hermes-Session-Key. The receiver creates a fresh unpredictable session; existing live-owner admission is unchanged. The paired sender must send full task context, suppress affinity/resume fields and use the ordinary run status/events/stop routes. No existing live-owner lock is bypassed.

Only terminal local foreground subprocesses consume the ContextVar-carried environment. There is no process-global mutation or shell snapshot persistence. Background, PTY and timeout promotion are denied. Literal values are scrubbed at dispatcher observers/results, durable DB/divert projection, queued logs, trajectory exports and API publication. Provider request debug dumps are disabled for credential runs. Authoritative live command arguments are not rewritten. Streamed partial output is suppressed. This is literal redaction, not protection against deliberately encoded exfiltration by trusted commands.

Sibling service control (`tools/sibling_service_policy.py`) is refused in credential runs. The refusal happens before protected-policy admission, approval and the helper sink, so a credential run never reaches the helper's fixed-environment early return. Uncredentialed sibling admission is unchanged.

The run credential scope (`gateway.runtime_context`) is separate from the per-profile terminal policy scope (`tools.terminal_scope`, `TERMINAL_*`). Both may be bound at once; neither reads or writes the other.

Spawn ownership is initialized before registration. Cancellation revokes future spawns and terminates identity-checked ordinary foreground children. Completion forgets reaped processes rather than signaling numeric reused PID/PGIDs. All registered callbacks are attempted even on error; context and ownership unwind via finally. There is no guarantee for daemonized, reparented, deliberately detached descendants, same-UID adversaries, arbitrary plugins or user-directed file writes. Scoped commands do not persist shell cwd/export state.

Closing a local run's scope does not revoke its credential at the issuing service. Operators must separately account for issuer expiry and revocation behavior, review the paired sender, and approve profile enablement and deployment. The synthetic acceptance tests do not establish a production model turn or hostile-process isolation.

The run-authority design is adapted from Acelro Engineering Agent's NousResearch/hermes-agent PR #81976. Integration-specific packaging and deployment policy are outside this core change.
