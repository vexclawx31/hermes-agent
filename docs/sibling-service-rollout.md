# Sibling service rollout admission (policy schema 3)

`tools/sibling_service_policy.py` admits an exact, protected-policy-derived helper argv before the generic terminal guards. Admission is not authority by itself: the normal approval guards still run, and denied or pending approval never executes.

## Schema 2 (unchanged)

`{schema: 2, profile, own_target, helper, targets}` admits `/usr/bin/sudo -n <helper> {status|restart} <name>` for each `targets` name. The sink timeout is `min(caller timeout, 60)` seconds.

## Schema 3 (adds a separate rollout registry)

Schema 3 is schema 2 plus `rollout: {targets: {name: service}, timeout_seconds}`. It admits two more request shapes:

    /usr/bin/sudo -n <helper> apply <plan-sha256> <name>[,<name>...]
    /usr/bin/sudo -n <helper> rollback <plan-sha256> <receipt-sha256> <name>[,<name>...]

Rules:
- **Target set:** names must be distinct members of `rollout.targets`. A mixed, unknown or duplicate set, or a set naming the hosting label, refuses as a whole. The hosting-label exclusion is checked again for every request, not only at policy validation.
- **Consistent registries:** a service may be in both the restart and the rollout registry only under the identical name mapped to the identical service. A name that maps to different services, or a service listed under different names, is refused. A name only in the rollout registry is never restartable, and a name only in the restart registry is never rollout-eligible.
- **Digests:** each must be 64 lowercase hex characters. `apply` takes one; `rollback` takes two.
- **Timeout:** the sink uses the protected `timeout_seconds` (60–3600) as its subprocess bound. The model's `timeout` argument does not shorten it.
- **Outer executor deadline:** every tool call also runs under the agent executor's own deadline: `timeouts.tools.sequential_call` for a lone call, and `timeouts.tools.concurrent_batch` for a parallel batch. The default is 420 s, and approval waits are excluded. When that deadline expires, the executor abandons the tool and reports a generic `timed out` error, while a root operation may still be running. To avoid that, the executor publishes its budget to the tool (`agent.deadline.current_tool_budget()`). A rollout is admitted only if the remaining budget covers `timeout_seconds + 120` s (`SIBLING_ROLLOUT_OUTER_MARGIN_S`). Otherwise it refuses as `blocked` before approval, with no prompt and no subprocess. An unknown budget also refuses; this covers any caller that publishes none. With the default 420 s deadlines, every rollout therefore refuses until both settings are raised (or set to `0`, meaning unbounded).
- **On timeout:** within that guarantee, a sink timeout is reported as `outcome: unknown` (`exit_code: null`), and nothing is retried automatically. The sink stops waiting for `sudo`, but the root-side operation may continue, and its receipts are authoritative. Limits outside the tool executor, such as a whole-turn or whole-run limit owned by a caller, are not inspected. They should not be set below the rollout bound.
- **Credential runs:** refused before admission. A run that carries scoped run credentials (for example a dispatched run with an allowlisted API key) cannot submit a rollout, and only an uncredentialed hosting session can. This stays so until an explicit authority contract supports sibling rollout under run credentials. Isolation is not relaxed implicitly.
- **Unchanged from schema 2:**
  - the hosting-identity and exact-profile checks;
  - the protected helper file check;
  - credential-run refusal before admission;
  - foreground only (no PTY, workdir or background);
  - approval guards and the native foreground finalizer.

### Optional consumers list (schema 3)

`consumers: {names: [...], timeout_seconds}` names non-gateway runtime consumers, such as a dashboard, a notifier or CLI wrappers. These are bare aliases with no launchd service implied. It admits two more request shapes:

    /usr/bin/sudo -n <helper> consumers-apply <plan-sha256> <name>[,<name>...]
    /usr/bin/sudo -n <helper> consumers-rollback <plan-sha256> <receipt-sha256> <name>[,<name>...]

- **Names:** must be distinct, and must not collide with any restart or rollout name or with the hosting profile. A request's set must be drawn wholly from the list; mixed, unknown and duplicate sets refuse.
- **Distinct verbs:** a consumer request can never be read as a rollout, and the reverse is also true.
- **Same rules as rollout:** budget gate, approval, credential-run refusal, foreground-only and finalizer.
- **Older policies:** schema 2, and schema 3 without the key, never admit these requests.

The command text only selects the operation and target set. The helper and its root-side operator must independently verify that the hash-bound plan targets exactly this set and excludes the hosting service.

## Adoption order

Before publishing a schema-3 policy, configure the hosting profile's `timeouts.tools.sequential_call` and `timeouts.tools.concurrent_batch` to at least `rollout.timeout_seconds + 120`. Choose `rollout.timeout_seconds` from measured operation durations. Raising these deadlines also lengthens how long any other wedged tool can hold a turn, which is a deliberate trade-off to record with the change.

A runtime that only knows schema 2 rejects any schema-3 file, so status and restart stop being admitted too, and fail closed. Publish a schema-3 policy only after the hosting gateway runs a core containing this change. If that gateway is rolled back to an older core, restore the schema-2 policy first.
