---
name: nvsec-skill
description: Lightweight skill for the NVIDIA `nvsec` CLI. Use when working with artifact signing, Security Portal tokens, `nvsec configure`, `nvsec sign`, `nvsec sign-async`, `nvsec sign-logs`, `nvsec 3s`, `nvsec rim`, `nvsec rimv2`, AWS IAM Identity Center via `nvsec aws`, or AWSOS via `nvsec awsos`. Prefer verified local help or installed-source inspection over guessing flags, especially because top-level `nvsec` may fail offline during its startup version check.
---

# NVSEC CLI

Use this skill for operational help around the local `nvsec` tool.

## Quick Start

- Verify the CLI exists with `which nvsec`.
- The default config file is `~/.nvsec.cfg`.
- The default log file is `~/nvsec.log`.
- If `nvsec <command> --help` fails in an offline or sandboxed environment, inspect the installed source instead of inventing flags. Use [references/command-surfaces.md](references/command-surfaces.md).

## Verified Behavior

- Top-level `nvsec` runs a startup version check against `https://api-docs.securityportal.nvidia.com/nvsec/version.txt` for most command paths. In disconnected environments, even `--help` may fail on DNS resolution before the requested command runs.
- `nvsec 3s` is special-cased in the root CLI and can often show help without hitting that version check.
- `nvsec aws` and `nvsec awsos` skip the update warning during normal command execution, but `nvsec aws --help` and `nvsec awsos --help` can still fail before Click resolves group help in offline environments.
- When help is blocked, prefer source inspection under the installed `cli/` package rather than guessing. The reference file lists the relevant modules and verified command surfaces.

## Working Pattern

1. Identify the domain first: `configure`, `sign`, `sign-async`, `sign-logs`, `3s`, `aws`, `awsos`, `rim`, `rimv2`, `version`, or `check-update`.
2. Prefer the narrowest relevant help command if it works.
3. If help fails because of the startup version check, inspect the installed source files listed in [references/command-surfaces.md](references/command-surfaces.md).
4. Before any state-changing operation, confirm the environment, auth mode, profile, file paths, output directory, and destructive intent.

## Configuration

User token flow:

```bash
nvsec configure user --profile default --env prod --token "$NVSEC_TOKEN"
```

Reset a profile:

```bash
nvsec configure reset --profile default
```

Service account flow:

```bash
nvsec configure sa --profile default
```

Starfleet service account flow:

```bash
nvsec configure ssa --profile default --service 3S
```

Verified details:

- `configure user --env` accepts `dev`, `stg`, or `prod`.
- `configure ssa --service` accepts `3S`, `ACCESS-MGMT`, or `CLOUDOS`.
- `configure sa` and `configure ssa` prompt for the remaining credentials rather than taking full non-interactive flags.

## Signing

Synchronous signing:

```bash
nvsec sign \
  --job-name "Windows Authenticode" \
  --description "Release signing" \
  --file ./artifact.exe \
  --out-dir ./signed
```

Trailing key/value job arguments:

```bash
nvsec sign \
  --job-name "Linux Kernel Module" \
  --description "Module signing" \
  --file ./module.ko \
  hashSize 256
```

Async flow:

```bash
nvsec sign-async sign \
  --job-name "Windows Authenticode" \
  --description "Release signing" \
  --file ./artifact.exe

nvsec sign-async job-status --job-id <JOB_ID>

nvsec sign-async download-signature \
  --job-id <JOB_ID> \
  --out-file-name artifact.exe \
  --out-dir ./signed
```

Download signing logs:

```bash
nvsec sign-logs --job-id <JOB_ID> --out-dir ./logs
```

Verified signing flags:

- `--config-file` overrides `~/.nvsec.cfg`.
- `--log-file` overrides the default `~/nvsec.log`.
- `--no-progressbar` is available on `sign` and the async signing/download commands.
- `--no-sign-suffix` is available on `sign` and `sign-async download-signature`.
- Extra job arguments are positional trailing key/value pairs; make sure the count is even before running.

## 3S, AWS, AWSOS, and RIM

- `nvsec 3s` is the safest place to start with live help in restricted environments.
- `nvsec 3s submit --input_file <path>` accepts its existing regular-file input or a local directory. A directory is validated and privately snapshotted before login, packaged as `<directory-name>.zip` with its contents at archive root, and cleaned up after the command exits. Regular-file behavior and the existing parameters, multipart, request-ID, and result-download options are unchanged.
- `nvsec 3s approve --request_id=123,456,789 --private_key <path>` approves explicit request IDs in bulk. `--request_id` may also be repeated. As an alternative input mode, `--request_id_file <path>` accepts `{"request_ids": [123, 456]}` or the single `{"requestId": 123}` file written by `submit --request_id_file`. Duplicate IDs are ignored in first-seen order. The command prepares the private key or YubiKey once, signs and submits each request independently, prints each result immediately, continues after request-level failures, and exits nonzero if any request fails. On retry, fully approved requests are reported as `ALREADY APPROVED — SKIPPED` and do not fail the batch. If interrupted during batch processing, completed approvals remain committed, the in-flight request is reported as `UNKNOWN`, later requests are `NOT ATTEMPTED`, and the command exits 130. These batch retry and interruption semantics apply only to explicit multi-ID input; interactive multi-select retains its existing sequential fail-fast behavior. A single request ID retains the direct-approval behavior. `--request_id`, `--request_id_file`, and `--interactive` are mutually exclusive request-selection modes.
- `nvsec 3s needs-approval`, `nvsec 3s approve --interactive`, and `nvsec 3s user-requests` use their established list APIs by default; add `--paged` only when explicitly opting into snapshot-aware pagination. `--paged` replaces `--use-cursor-api`; the previous name is not accepted. Paged interactive approval keeps cursor/snapshot values internal, caches visited pages, marks the last page, and provides Next/Previous actions; non-interactive commands expose continuation metadata. `--page_size` is the interactive approval pagination control and requires `--paged` on `approve`. Treat cursors as opaque and scoped to the command, API mode, filters, and sort direction that produced them; never reuse a legacy cursor with a paged request, and always pair a continuation with its returned snapshot.
- Multi-word `nvsec 3s` flags use underscores only; previous hyphenated spellings are not accepted. Paging and output flags include `--page_size`, `--snapshot_max_id`, `--scan_batch_size`, `--sort_order`, `--sort_by`, `--sort_direction`, and `--print_table`; the Dev no-quorum flag is `--no_quorum`, not `--no-quorum`.
- `nvsec 3s view-requests` uses the snapshot-aware View Requests API. Its multi-word filters are `--request_id`, `--job_type_name`, `--job_type_display_name`, `--job_type_match_mode`, `--created_after`, and `--created_before`; the old hyphenated forms are not accepted. Its JSON output contains only the full page envelope, including cursor/snapshot continuation metadata; add `--print_table` for a human-readable table. Filename and Description are display-only and are never exposed as filters.
- `nvsec 3s submit-hsm --hsm HSM_A|HSM_B` selects production defaults and creates the token before submitting the ceremony. HSM A maps to `HSM_QUORUM_TOKEN_CREATE_A` + `HSM_CEREMONY_GENERAL_A`; HSM B maps to `HSM_QUORUM_TOKEN_CREATE_B` + `HSM_CEREMONY_GENERAL_B`. Explicit `--job_type` and `--token_job_type` override their respective defaults; known cross-HSM combinations are rejected. `--token_request_id <id>` reuses a known token without creation. Verify deployed Job Types and follow the authoritative HSM automation guide's notification order; its normal production procedure retains separate token creation and reuse. `--hsm` is production-only and cannot be combined with `--use_dev`. Without `--hsm`, supply `--job_type HSM_CEREMONY_*` and exactly one of `--token_job_type`, `--token_request_id`, or `--no_quorum`. Dev quorum retains explicit `HSM_QUORUM_TOKEN_CREATE_DEV_QUORUM` + `HSM_CEREMONY_TEST_DEV_QUORUM` and separate `--use_dev`; no-quorum also requires `--use_dev`.
  - `--input_file` accepts a local `.yml`/`.yaml` file, ZIP content regardless of filename extension, or a directory packaged at ZIP root. `--parameters <JSON>` validates a JSON object, preserves exact selectors, and pins wildcard input.ceremonyFile to its selected concrete path without changing other fields. Kagido accepts omitted or `{}` parameters for legacy behavior; a structured contract must use `schemaVersion: 1`. Accepted `input` fields are `ceremonyId`, `ceremonyFile`, and `requiredPaths`; accepted `output` fields are `includePaths`, `excludePaths`, and `requiredPaths`. Unknown fields and unsupported versions fail in Kagido.
  - To select one YAML from a multi-YAML directory or ZIP, use `--parameters '{"schemaVersion":1,"input":{"ceremonyFile":"ceremonies/dev.yaml"}}'`. Without `input.ceremonyFile`, the package must contain exactly one recursive YAML. A direct YAML input may use a selector that matches its basename. Selectors support `*` within one path component and standalone `**` across directory levels, and must select exactly one regular YAML. Token modes require exactly one complete `token: __NVSEC_HSM_APPROVAL_DATA__` scalar in the selected YAML and replace it with approval data. `--no_quorum --use_dev` skips token creation/retrieval: a present placeholder becomes YAML `null`, and a selected YAML without any placeholder is submitted unchanged, allowing Dev readCreds ceremonies without unlock. Other YAMLs and companions remain unchanged. Kagido validates ceremony schema and actions; 3S controls request release, and SHU/HSM verify supplied approvals. Marker-free input with token options fails before login with guidance to use explicit no-quorum mode. Never substitute the string `None`.
  - The command validates and snapshots the entire source before login, pins wildcard selectors to the selected concrete path, stages source and rendered content only in owner-only temporary storage, leaves the original input unchanged, and removes staging on every exit path. Recovery commands reuse the token request in token mode or retain `--no_quorum` in no-quorum mode; both retain `--parameters` with `<REUSE_ORIGINAL_PARAMETERS>` instead of exposing its value. Restore the original JSON from the same trusted source. Do not use or suggest `--ceremony_file`, source/format flags, `--parameters_file`, `--request_id_file`, a persistent token-result file, a local journal, or any way to print, log, or expose `approval_data`. Reusing a token request can make duplicate ceremonies; they may share that token, but only one approved execution consumes it. Retrying a no-quorum submission can also create a duplicate Dev ceremony.
- `nvsec 3s approve-hsm --request_id <ceremony-id>` downloads that request's original input, verifies its raw SHA-256 against the 3S approval challenge, selects YAML using stored `input.ceremonyFile` when present, and signs the exact `unlock` approval-data token. It never selects a latest global token. The common YubiKey flow keeps slots `9d` (3S) and `9c` (HSM), slot/PEM overrides, separate role-labeled PIN prompts, and both proofs.
  - Explicit `--token_request_id`, `--token`, and detached `--sig` remain mutually exclusive proof sources. Explicit `--token` overrides `NVSEC_3S_HSM_TOKEN`; `--token_request_id` or `--sig` ignores that ambient variable. Clear it for default ceremony extraction. Token sources also attempt ceremony-input verification and decoded-token comparison: mismatches and request/hash integrity failures stop before signing. If the input is unavailable or unparseable, an explicit token source can continue with a comparison-unavailable warning for recovery; it cannot replace a different ceremony token. Default extraction fails with explicit-source recovery guidance. `--sig` bypasses extraction/comparison.
- `nvsec 3s approve-hsm --use_dev --auth ssa` supports automated Dev approvals with PEM-backed fixture keys. It uses bearer-only `RequestSsa` approval routes and preserves optional `--impersonate`; 3S remains authoritative for scope, impersonation, approver, request-state, and proof checks. The CLI rejects SSA approval without `--use_dev`. SSA with a default or explicit YubiKey 3S approver key fails before login with guidance to supply a PEM approver key or select `--auth oidc`; the registered-key preflight is user-only. This restriction applies to the 3S approver key, not the separate HSM administrator key.
- For a repeatable Dev live test of ceremony-bound token extraction and three SSA PEM-fixture approvals, follow [references/dev-approve-hsm-live-test.md](references/dev-approve-hsm-live-test.md). Use its checked-in renderer and fixture-approval helper instead of exposing `approval_data` or reconstructing the ceremony manually.
- `nvsec aws` is the AWS IAM Identity Center flow. The verified high-level commands are `auth`, `list`, `login`, `configure`, `status`, `stop`, and `troubleshoot`.
- `nvsec awsos` is the legacy Security Portal AWS credential flow. The verified entrypoint is `get-creds`.
- `nvsec rim` and `nvsec rimv2` both support `login`, `upload`, `promote`, `download`, `list`, `info`, and `delete`; `rim` also has `set-status`.
- `nvsec rimv2 upload --coev_file <path>` optionally Base64-encodes a CoEV file and sends it with the CoRIM. Omit the option for the existing RIM-only upload; RIM Service validates that CoEV is used only with `--rim_type CORIM`.
- For exact command inventories and offline inspection paths, use [references/command-surfaces.md](references/command-surfaces.md).

## Response Rules

- Do not invent flags or auth flows that have not been verified locally.
- Call out when a failure is caused by the startup version check rather than the requested operation itself.
- Ask for missing metadata before any signing or destructive RIM action.
- Prefer narrow help when it works, and installed-source inspection when it does not.
