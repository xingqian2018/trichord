# NVSEC command surfaces

Use this file when `nvsec --help` or `nvsec <group> --help` is blocked by the root CLI's startup version check.

## Locating the installed source

Common locations on this machine were:

- `~/Library/Python/3.14/lib/python/site-packages/cli/`
- `/opt/homebrew/lib/python3.14/site-packages/cli/`

Portable discovery commands:

```bash
python3 -c 'import cli, pathlib; print(pathlib.Path(cli.__file__).resolve())'
python3 -c 'import cli.nvsec, pathlib; print(pathlib.Path(cli.nvsec.__file__).resolve())'
```

Useful inspection commands:

```bash
rg -n "@cli\\.(command|group)\\(" "$(python3 -c 'import cli, pathlib; print(pathlib.Path(cli.__file__).resolve().parent)')"
sed -n '1,220p' "$(python3 -c 'import cli.nvsec, pathlib; print(pathlib.Path(cli.nvsec.__file__).resolve())')"
```

## Root CLI behavior

- Root module: `cli/nvsec.py`
- Verified top-level commands wired there:
  - `version`
  - `check-update`
  - `configure`
  - `sign`
  - `sign-async`
  - `sign-logs`
  - `awsos`
  - `aws`
  - `3s`
  - `rim`
  - `rimv2`
- The root callback performs a network version check for most invocations.
- `3s` is explicitly special-cased before that setup path.
- `aws` and `awsos` only skip the update warning; they do not avoid the network setup path when group help is resolved from the root command.

## Verified group inventories

`configure` from `cli/commands/configure/configure.py`

- `user`
- `reset`
- `sa`
- `ssa`

`sign-async` from `cli/commands/sign/sign_async.py`

- `sign`
- `job-status`
- `download-signature`

`3s` from live `nvsec 3s --help`

- `adjust-approvers`
- `all-jobtypes`
- `approve`
- `approve-hsm`
- `deny`
- `download-input-file`
- `download-log-file`
- `download-signed-file`
- `get-default-parameters`
- `job-type-creation-request`
- `job-type-info`
- `import-yubikey-approver`
- `login`
- `logout`
- `needs-approval`
- `potential-approvers`
- `public-key`
- `register-hsm-admin`
- `request-info`
- `save-job-type`
- `setup-yubikey-approver`
- `sign-token`
- `submit`
- `submit-hsm`
- `update-public-key`
- `user-jobtypes`
- `user-requests`
- `view-requests`
- `watch`

`aws` from `cli/commands/aws/aws.py`

- `auth`
- `list`
- `login`
- `configure`
- `status`
- `stop`
- `troubleshoot`

`awsos` from `cli/commands/awsos/awsos.py`

- `get-creds`

`rim` from `cli/commands/rim_service/rim_service.py`

- `login`
- `upload`
- `promote`
- `download`
- `list`
- `info`
- `set-status`
- `delete`

`rimv2` from `cli/commands/rim_servicev4/rim_service.py`

- `login`
- `upload`
- `promote`
- `download`
- `list`
- `info`
- `delete`

`rimv2 upload` accepts optional `--coev_file <path>` for a CoEV paired with a CoRIM; the file is Base64-encoded into `coev.document`, and omission leaves the RIM-only payload unchanged.

## High-value verified details

- Config path constant: `~/.nvsec.cfg`
- Default log path from logger setup: `~/nvsec.log`
- `configure ssa --service` choices are currently `3S`, `ACCESS-MGMT`, and `CLOUDOS`
- `sign` and `sign-async sign` take trailing positional key/value job arguments
- `sign-logs` takes `--job-id`, optional `--config-file`, optional `--out-dir`, and optional `--log-file`
- `3s approve` accepts mutually exclusive `--request_id`, `--request_id_file`, or `--interactive` request selection; the file input supports `{"request_ids": [123, 456]}` and `{"requestId": 123}`. Explicit multi-ID results are printed per request; fully approved retries are skipped, and interrupted batches exit 130 after marking the in-flight ID `UNKNOWN` and later IDs `NOT ATTEMPTED`. Interactive multi-select retains its sequential fail-fast behavior
- Multi-word `3s` flags use underscores only; previous hyphenated spellings are not accepted. `--paged` replaces `--use-cursor-api` and enables snapshot-aware pagination; related options include `--page_size`, `--snapshot_max_id`, `--scan_batch_size`, `--sort_order`, `--sort_by`, `--sort_direction`, and `--print_table`
- `3s view-requests` multi-word filters are `--request_id`, `--job_type_name`, `--job_type_display_name`, `--job_type_match_mode`, `--created_after`, and `--created_before`; their previous hyphenated forms are not accepted
- Normal `3s submit --input_file` accepts its existing regular-file input or a local directory. A directory is safely snapshotted before authentication and uploaded as an owner-only `<directory-name>.zip` whose contents are at archive root; staging is removed on every exit path. This does not add ceremony selection or token rendering to normal `submit`
- `3s submit-hsm --hsm HSM_A` defaults to `HSM_QUORUM_TOKEN_CREATE_A` + `HSM_CEREMONY_GENERAL_A`; `--hsm HSM_B` defaults to `HSM_QUORUM_TOKEN_CREATE_B` + `HSM_CEREMONY_GENERAL_B`. The command creates a token then submits the ceremony. Explicit `--token_job_type` and `--job_type` override their defaults; known cross-HSM combinations are rejected. `--token_request_id <id>` reuses a known token without creation. `--hsm` is production-only and incompatible with `--use_dev`. Follow the authoritative production guide's notification sequence, which retains separate token creation/reuse. Without `--hsm`, supply `--job_type` and exactly one of `--token_job_type`, `--token_request_id`, or Dev-only `--no_quorum`; the latter requires `--use_dev`. Dev keeps explicit matching target Job Types and separate API selection through `--use_dev`. `--input_file` accepts a direct `.yml`/`.yaml`, ZIP content regardless of extension, or a directory packaged at ZIP root. `--parameters <JSON>` validates a JSON object, preserves exact selectors, and pins wildcard input.ceremonyFile to its selected concrete path without changing other fields. Kagido accepts omitted or `{}` legacy parameters; a structured contract must use schema version 1, with `input` fields `ceremonyId`, `ceremonyFile`, and `requiredPaths`, and `output` fields `includePaths`, `excludePaths`, and `requiredPaths`. Unknown fields and unsupported versions fail in Kagido. A multi-YAML directory or ZIP selects one ceremony with `--parameters '{"schemaVersion":1,"input":{"ceremonyFile":"ceremonies/dev.yaml"}}'`; without `input.ceremonyFile`, it must contain exactly one recursive YAML. A direct YAML input may use a selector that matches its basename. Selectors support `*` within one path component and standalone `**` across directory levels, and must select exactly one regular YAML. Token modes require exactly one complete `token: __NVSEC_HSM_APPROVAL_DATA__` scalar in the selected YAML and replace it with approval data. No-quorum mode skips token creation/retrieval: a present placeholder becomes YAML null (`token: null`), and a selected YAML without any placeholder is submitted unchanged, allowing Dev readCreds ceremonies without unlock. Other YAMLs and companions remain unchanged. Kagido validates ceremony schema and actions; 3S controls request release, and SHU/HSM verify supplied approvals. Marker-free input with token options fails before login with explicit no-quorum guidance; `token: None` is a string and must not be used. The entire input is validated before authentication, wildcard selectors are pinned to their selected concrete path while other parameters and original source bytes remain unchanged, and staged artifacts use owner-only temporary storage removed on every exit path. Recovery reuses the token request in token mode or retains `--no_quorum` in no-quorum mode, while retaining `--parameters` only as a safe placeholder rather than its value. There are no `--ceremony_file`, source/format flags, `--parameters_file`, persistent token-result file, `--request_id_file`, or local journal. Reused tokens can create duplicate ceremonies even though only one approved execution consumes the token; no-quorum retries can also duplicate a Dev ceremony.
- `3s approve-hsm --request_id <ceremony-id>` verifies the original input's raw SHA-256 against the 3S challenge, selects YAML using stored `input.ceremonyFile` when present, and extracts the exact `unlock` token. It never selects the latest global token. Default keys remain YubiKey slots `9d` (3S) and `9c` (HSM), with slot/PEM overrides and both proofs. Explicit `--token_request_id`, `--token`, and `--sig` are mutually exclusive. Explicit `--token` overrides `NVSEC_3S_HSM_TOKEN`; `--token_request_id` or `--sig` ignores it. Clear the variable for default extraction. Token-source overrides also attempt ceremony-input verification and decoded-token comparison; mismatches and request/hash integrity failures stop before signing. Unavailable/unparseable input permits an explicit token source to continue with a comparison-unavailable recovery warning. Default extraction fails with explicit-source guidance; `--sig` bypasses extraction/comparison.
- `3s approve-hsm --shard <username>:<file>` (repeatable, one per CU; the CU is required) sends a CU-password share as `keyl1:<base64>` RSA-OAEP-SHA256 ciphertext to Kagido KeyL in `hsmApproval.shards`; 3S never receives it in plaintext. `--shard` reads a plaintext share file; `--encrypted_shard <username>:<file>` (also repeatable) first decrypts an RSA-OAEP-SHA256 file encrypted to the approver key (SHU `allocate`/`reset` trustee outputs) with the 3S approver key (YubiKey slot or `--approver_private_key` PEM). Each CU is named once across both options. Shares are normalized from `NN-hex` or `token-NN-hex` in the file contents (after decryption for `--encrypted_shard`); filenames are ignored. Bare hex is rejected because it has no share index. The content index must be 01–255 and must not exceed a manifest’s `total`. File-read diagnostics omit filenames to avoid exposing legacy inline shares. The binding is `kagido.cu-share.v1|<requestId>|<dev|prod>|<username>|<shardSetId|legacy-2>` from the verified ceremony input's `allocate`/`combine` action `manifestFile`; a CU with no such action is rejected. Production uses only the committed KeyL key; `--keyl_public_key` requires `--use_dev`. The share is never printed. A selected ceremony with no `unlock`/`reset` action, approved with shards and no explicit proof source, sends `hsmApproval.signature` as an empty string and skips the HSM-admin key; `reset` tokens are extracted like `unlock`. Action matching is case-insensitive, so uppercase/mixed-case token actions still require signatures.
- `3s approve-hsm` accepts `--auth`, `--client_id`, `--client_secret`, and `--scope` with standard SSA credential precedence. SSA requires `--use_dev` and uses bearer-only `RequestSsa` approval routes. Automated Dev approvals use the existing PEM fixtures; manual YubiKey approvers use `--auth oidc` when ambient SSA credentials are configured. Supply secrets through protected environment/configuration and passphrases through hidden prompts.
