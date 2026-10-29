# Dev approve-hsm live test

Use this procedure only in Dev to stage a three-approver
`HSM_CEREMONY_TEST_DEV_QUORUM` request. It validates that `submit-hsm` creates
and retrieves the matching token without exposing its `approval_data`, then
extracts the exact token from that ceremony request's verified original input
for the three reviewed PEM fixture SSA approvals. Run only
after the SGN-8811 backend approval routes and Dev policy are deployed.

## Safety contract

- Require `--use_dev` on every command.
- Use exactly `HSM_QUORUM_TOKEN_CREATE_DEV_QUORUM` and
  `HSM_CEREMONY_TEST_DEV_QUORUM`.
- Keep the rendered ceremony template in a mode-`0700` runtime directory with
  mode-`0600` files. `submit-hsm` handles the token result in memory; never
  print, log, paste, download, or otherwise disclose `approval_data`.
- Use the reviewed Dev identities `crabman`, `jschmo`, and `rusty`, in that
  order. Check structured approval readback after each call; exactly the third
  approval must release the ceremony.
- Use the non-multipart submit path. The Dev `SubmitRequestV2` route returned
  HTTP 404 during the verified run; omitting `--multi_part_upload` uses the
  working `SubmitRequest` route.
- Render a deliberately failing ceremony that targets only
  `cu_test_ceremony_02`, searches for a unique missing key, and locks the CU
  out. The request is expected to fail after approval; that failure prevents
  creation or replacement of a real key.

## Prepare protected paths

Run from the nvsec source checkout and substitute the local paths:

```bash
NVSEC_SOURCE=/path/to/nvsec-tool
NVSEC_PYTHON=/path/to/nvsec-python
E2E_ROOT=/path/to/end-to-end-tests
export NVSEC_SKIP_SKILL_INSTALL=1
umask 077
RUNTIME_DIR="$(mktemp -d "${TMPDIR:-/tmp}/nvsec-approve-hsm.XXXXXX")"
chmod 700 "$RUNTIME_DIR"
```

Provision SSA credentials through protected `NVSEC_SSA_CLIENT_ID`,
`NVSEC_SSA_CLIENT_SECRET`, and `NVSEC_SSA_SCOPE` variables or a protected
`[3s-dev]` configuration profile. Do not place secrets in command arguments,
enable shell tracing, or print credential values. No device-code login is
needed. The reviewed Dev job fixtures use scope `SIGNING_HSM_CEREMONY_TEST`;
set `NVSEC_SSA_SCOPE` to that value and ensure it is granted to the client.
The backend remains authoritative for feature enablement, scope,
impersonation, approver eligibility, request state, and both proofs; stop on
any backend rejection.

## Render a ceremony template and submit it

Render the checked-in template. It intentionally contains exactly one complete
`token: __NVSEC_HSM_APPROVAL_DATA__` scalar; do not replace that placeholder.

```bash
MISSING_KEY="nvsec_approve_hsm_missing_$(date +%s)"
"$NVSEC_SOURCE/cli/skills/nvsec-skill/scripts/render-dev-approve-hsm-ceremony.py" \
  --output "$RUNTIME_DIR/find-missing-key.yml" \
  --label "$MISSING_KEY"
```

Submit through the non-multipart route. `submit-hsm` creates the token request,
waits for it, retrieves its result in memory, renders the protected ceremony,
and submits it. Do not add `--request_id_file` or a token-result download.

```bash
PYTHONPATH="$NVSEC_SOURCE${PYTHONPATH:+:$PYTHONPATH}" "$NVSEC_PYTHON" -m cli.nvsec 3s submit-hsm \
  --use_dev \
  --auth ssa \
  --token_job_type HSM_QUORUM_TOKEN_CREATE_DEV_QUORUM \
  --job_type HSM_CEREMONY_TEST_DEV_QUORUM \
  --input_file "$RUNTIME_DIR/find-missing-key.yml" \
  --description 'Dev submit-hsm and approve-hsm test' \
  --timeout 600
```

Record the non-secret `Token request ID: <id>` and `Ceremony request ID: <id>`
lines printed by `submit-hsm` in your terminal for observation and recovery.
Approvals below need only the Ceremony request ID. This is not a local journal; do not write them to a
request-ID file. The command may print a recovery command using
`--token_request_id` if the ceremony submission outcome is unknown. Retrying
that command can create a duplicate ceremony: the ceremonies can share the
token, but only one approved execution consumes it.

```bash
# Replace these examples with the two IDs printed above.
TOKEN_REQUEST_ID=1884001
CEREMONY_REQUEST_ID=1884002
```

Use `request-info` to require the token request to be `Completed`, without
inspecting or downloading its result:

```bash
PYTHONPATH="$NVSEC_SOURCE${PYTHONPATH:+:$PYTHONPATH}" "$NVSEC_PYTHON" -m cli.nvsec 3s request-info \
  --use_dev --auth ssa --request_id "$TOKEN_REQUEST_ID"
```

Require `Awaiting Approval` with an empty `approvals` list before continuing:
If the ceremony is still `Awaiting Validation`, poll this readback within a
ten-minute deadline; do not submit a duplicate ceremony or approve before
validation finishes.

```bash
PYTHONPATH="$NVSEC_SOURCE${PYTHONPATH:+:$PYTHONPATH}" "$NVSEC_PYTHON" -m cli.nvsec 3s request-info \
  --use_dev --auth ssa --request_id "$CEREMONY_REQUEST_ID"
```

## Approve with the three PEM fixtures

The Expect helper supplies the reviewed fixture passphrases to hidden prompts
without putting them on the command line. It invokes `approve-hsm --use_dev
--auth ssa` with both PEM keys and `--impersonate`. Its five-argument form
clears ambient `NVSEC_3S_HSM_TOKEN` and omits `--token_request_id` to verify
ceremony-bound extraction. It prepends the source checkout to existing
`PYTHONPATH`, preserving runtime dependency paths. Run it once per identity:

```bash
APPROVE_FIXTURE="$NVSEC_SOURCE/cli/skills/nvsec-skill/scripts/approve-dev-hsm-sock-puppet.exp"

"$APPROVE_FIXTURE" "$NVSEC_SOURCE" "$NVSEC_PYTHON" "$E2E_ROOT" \
  "$CEREMONY_REQUEST_ID" crabman

PYTHONPATH="$NVSEC_SOURCE${PYTHONPATH:+:$PYTHONPATH}" "$NVSEC_PYTHON" -m cli.nvsec 3s request-info \
  --use_dev --auth ssa --request_id "$CEREMONY_REQUEST_ID"

"$APPROVE_FIXTURE" "$NVSEC_SOURCE" "$NVSEC_PYTHON" "$E2E_ROOT" \
  "$CEREMONY_REQUEST_ID" jschmo

PYTHONPATH="$NVSEC_SOURCE${PYTHONPATH:+:$PYTHONPATH}" "$NVSEC_PYTHON" -m cli.nvsec 3s request-info \
  --use_dev --auth ssa --request_id "$CEREMONY_REQUEST_ID"
```

After the first approval, require `Partially Approved` with only `crabman`.
After the second, require the same status with exactly `crabman` and `jschmo`.
Do not proceed if `rusty` or any other approval is already present. Then run:

```bash
"$APPROVE_FIXTURE" "$NVSEC_SOURCE" "$NVSEC_PYTHON" "$E2E_ROOT" \
  "$CEREMONY_REQUEST_ID" rusty

PYTHONPATH="$NVSEC_SOURCE${PYTHONPATH:+:$PYTHONPATH}" "$NVSEC_PYTHON" -m cli.nvsec 3s request-info \
  --use_dev --auth ssa --request_id "$CEREMONY_REQUEST_ID"
```

Require exactly `crabman`, `jschmo`, and `rusty` in `approvals` and an `Approved`
state in the history before processing. Poll request metadata with a bounded
ten-minute deadline. The deliberately missing-key ceremony must reach `Failed`
for the expected missing-key reason. Inspect only safe metadata and the sign
log; never print the protected rendered input or token result. Check captured
logs/artifacts for protected values without printing matching content.

For an explicit-source recovery check, the helper also accepts the original
six-argument form: source, Python, E2E root, Ceremony request ID, Token request
ID, identity. This adds `--token_request_id` and retains its precedence over
an ambient token. A supplied token must match the ceremony token when comparison
is available; request/hash integrity failures always stop before signing.

## Optional manual YubiKey variant

For a separate manual test, stop after the first two fixture approvals and let
the operator supply the third approval instead of `rusty`. Give the operator
the Ceremony request ID; retain the Token request ID for observation/recovery.
With the default provisioning,
slot `9d` supplies the 3S proof and slot `9c` supplies the HSM proof:

```bash
unset NVSEC_3S_HSM_TOKEN
PYTHONPATH="$NVSEC_SOURCE${PYTHONPATH:+:$PYTHONPATH}" "$NVSEC_PYTHON" -m cli.nvsec 3s approve-hsm \
  --use_dev \
  --auth oidc \
  --request_id "$CEREMONY_REQUEST_ID"
```

The registered-key preflight uses the user-only `Approver/GetPublicKey` route,
so manual YubiKey approvers must use `--auth oidc` when SSA credentials are
configured. Add `--device_code` only when the cached Dev login is not usable. After the
approval, verify that the operator is the third approver. The deliberately
missing-key ceremony should then reach `Failed`; inspect only safe request
metadata and the sign log, never the protected rendered input or token result.

## Cleanup

Remove the protected runtime directory after the final verification or when
abandoning the test:

```bash
rm -rf -- "$RUNTIME_DIR"
```

Resolve and inspect `RUNTIME_DIR` before cleanup. Never use an empty value, a
workspace root, or a broad temporary-directory path as the target.
