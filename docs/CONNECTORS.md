# Devon connectors

Connectors turn Devon from "here's what you should do" into "done." Each
connector owns one external service: its auth flow, its credential lifecycle
(encrypted vault — never chat, logs, or the command line), a live status
check, and provisioning of whatever the service's API legitimately allows.

## Using connectors

```bash
nm connectors list                          # every connector Devon knows
nm connectors connect --name github         # guided connect flow
nm connectors status --name github          # live status check
nm connectors disconnect --name github      # remove the stored credential
nm connectors provision --name github --kind repo \
    --params-json '{"name":"my-project","private":true}'
```

Connecting needs the vault unlocked:

```bash
export NM_VAULT_PASSPHRASE="your vault passphrase"
```

Secrets can also arrive non-interactively via environment variables (e.g.
`GITHUB_TOKEN`); without a TTY and without the env var, connect fails fast
with a clear message instead of hanging.

## GitHub (reference connector)

Auth is a **fine-grained personal access token**, which only you can create
(the API cannot mint PATs — that one step stays human):

1. Open https://github.com/settings/tokens?type=beta
2. Generate a fine-grained token (Contents: read/write is the usual need)
3. `nm connectors connect --name github` and paste it (or set `GITHUB_TOKEN`)

Then Devon can, on your request:

- create repos, branches, releases
- push code from a local checkout (the token reaches git through a
  one-shot askpass helper — never on the command line or in git config)
- register webhooks and deploy keys
- back up repos: `provision --kind repo_backup` mirror-clones every branch
  and tag, verifies the mirror with `git fsck`, and writes a manifest

```python
from nomorals.connectors import create_connector
gh = create_connector("github", vault)
gh.create_repo("my-project", private=True)
gh.push_code("~/work/my-project", "you/my-project", branch="main")
gh.provision("repo_backup", repo_ref="you/my-project", dest_dir="~/backups")
```

## Adding a service

Subclass `Connector`, decorate with `@register_connector`, and it appears in
`nm connectors list`:

```python
from nomorals.connectors import Connector, register_connector

@register_connector
class MonoConnector(Connector):
    id = "mono"
    name = "Mono"
    description = "Nigerian bank accounts via Mono"
    auth_methods = (AuthMethod.API_KEY,)
    ...
```

Queued next: finance (Mono for Nigeria, Plaid for US/EU), virtual cards,
proxy pool, Nigerian commerce. The framework is service-agnostic so these
slot straight in.

## Human-in-the-loop checkpoints

Some flows hit a step only a human can do — a CAPTCHA, an email-verification
click, a 2FA code, accepting terms. Devon does not bypass these. It
**pauses**: the step is persisted as a checkpoint, you are pinged through the
owner-only delivery channel (like an alarm — it gets through even in quiet
hours), and the flow resumes after *you* personally complete the step. A
CAPTCHA's job is proving a human is involved; you solving it yourself
**satisfies** the check — it is not bypassed.

```bash
nm connectors provision --name github --kind github_account \
  --params-json '{"username": "yourname"}'   # pauses at signup + email steps
nm connectors checkpoint list               # what's waiting on you
nm connectors checkpoint resolve --id <id>  # done — flow continues
nm connectors checkpoint cancel --id <id>   # abandon it
```

Rules: one account per service (a second account flow refuses while a
credential is stored), your own identity only, credentials handed to you and
vault-saved on completion. Checkpoints expire after 24h by default.

## Account policy

Devon provisions whatever a service's API legitimately allows — repos, API
keys, OAuth apps, deploy keys, webhooks, storage buckets — on your request
or standing permission. Provisioned credentials are handed to you and saved
in the vault.

For account *signup* itself, Devon may drive the flow and pause at each
human-verification step for you to complete personally (see checkpoints
above) — one account per service, your own identity. What Devon will never
do: auto-solve CAPTCHAs itself — no solver services, no AI-based bypass, no
verification dodging — and never create fake-identity or bulk accounts. The
human checkpoint is the only path through human verification.
