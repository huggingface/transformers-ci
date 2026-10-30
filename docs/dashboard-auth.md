# GitHub identity and CLI access

The dashboard authenticates users through its existing GitHub App. Authentication
and `/profile` are served by the trace exporter, independently of Grafana.
The profile shows the GitHub account and active browser/CLI sessions, allows
revocation, and provides the confirmation page for `tmic auth login`.

## Configure the GitHub App

Use the **dashboard GitHub App**, rather than Grafana's separate OAuth App:

1. Add `https://dashboard.example.com/auth/callback` as a user
   authorization callback URL on the GitHub App.
2. Under **Organization permissions**, grant **Members: Read-only**, and have an
   organization owner accept the updated permissions on the installation. The
   membership gate cannot read any user's membership without it, and login
   then fails with `github_app_cannot_read_membership`.
3. Copy the App's **client ID** (different from its numeric App ID), and generate
   a client secret. Its existing installation private key is not a client secret.
4. Provision the client secret through your secret manager into the Kubernetes
   Secret named by `secrets.name`, under the key configured by
   `secrets.keys.dashboardGithubClientSecret` (default
   `dashboard-github-client-secret`). Do not put the value in Helm values or Git.
5. Set these deployment values, retaining the persistent run store:

   ```yaml
   traceExporter:
     replicas: 1
     auth:
       enabled: true
       publicUrl: https://dashboard.example.com
       githubClientId: <GitHub App client ID>
       allowedOrganizations: [your-organization]
       allowedUserIds: []
     runStore:
       enabled: true
   ```

The chart routes `/auth` and `/profile` directly to the exporter. The public
origin is fixed configuration: request Host and forwarded headers never determine
OAuth callback destinations. Use HTTPS. Allow egress to **both** `github.com`
(code exchange) and `api.github.com` (identity and memberships), either directly
or through `traceExporter.auth.githubProxy`.

GitHub's [web authorization flow](https://docs.github.com/en/apps/creating-github-apps/authenticating-with-a-github-app/generating-a-user-access-token-for-a-github-app)
uses browser-bound, single-use state and PKCE. The server fetches `/user` and
the authenticated user's membership in each allowed organization, then discards
the GitHub token. The [own-membership endpoint](https://docs.github.com/en/rest/orgs/members#get-an-organization-membership-for-the-authenticated-user)
needs the App's **organization `Members: read` permission** for GitHub App user
tokens: GitHub answers 404 for a non-member (`account_not_allowed`) and 403 when
the App lacks that permission or is blocked (`github_app_cannot_read_membership`).
Private active memberships count; pending invitations do not. GitHub SSO may require an active
organization SSO session during authorization.

The default preserves the previous Hugging Face member login restriction.
`allowedUserIds` grants additional accounts access by immutable numeric GitHub
ID. Setting **both** lists empty explicitly allows all GitHub users to sign in,
including access to existing session-gated LLM dashboard actions; make that
policy decision separately from enabling authentication. Repository write
permission is still checked by Serge for workflow dispatch.

## Sessions and profile

Browser and CLI sessions expire after eight hours. Users sign in again after
expiry; there is no refresh token in this first version. Sessions are opaque
random credentials; only SHA-256 hashes are stored in `auth.sqlite3` on the
run-store volume. GitHub account IDs are the identity keys; a username change
updates the display name without changing account ownership on the next login.

Browser cookies use `Secure`, `HttpOnly`, `SameSite=Lax`, and the `__Host-`
prefix. Profile responses are uncached and cannot be framed. Browser mutations
require the configured Origin and `X-TCI-Action: 1`. CLI bearer tokens cannot
authorize existing browser mutations. Dashboard actions no longer accept a
Grafana session as identity. Grafana can retain its own login for CI Health and
other Grafana administration; it has no role in dashboard authentication.

The profile lists all of the account's active sessions. Revocation takes effect
on the next request. CLI logout revokes the server credential before removing
its local copy. GitHub logout or revocation of the App does not immediately end
an already-issued dashboard session: sessions have their own revocation and
eight-hour expiry. Membership is checked at login. GPU submission must recheck
repository authorization rather than treating a session as a compute grant.

The initial store supports one exporter replica; the chart rejects auth with
multiple replicas. Keep the SQLite file and its WAL sidecars together on the
persistent volume, with access restricted to the exporter. A shared database is
needed before scaling authentication across replicas.

## CLI handoff

```sh
tmic auth login
tmic auth status
tmic auth logout
# For SSH or another dashboard deployment:
tmic auth login --no-browser --dashboard https://dashboard.example.com
```

`tmic` starts a device-style grant, opens the dashboard confirmation page and
prints the code. The user signs in with GitHub, checks the code against their
terminal, and explicitly approves or denies access. An approval belongs to that
account and grant only. The CLI polls with a separate secret, respecting the
poll interval and slowdown response. Grants expire after 15 minutes and can be
exchanged only once. Device secrets and session credentials never appear in URLs.

CLI credentials are scoped to `profile:read mic:read mic:submit`. These are
capabilities for the future MIC job API, not permission to publish expectations,
approve changed baselines or spend GPU resources without repository checks.
This change adds authentication only; it does not add remote GPU job endpoints.

The separate MIC collection service adds `/mic-jobs` behind
`traceExporter.micJobs.enabled`. It reuses this identity and Serge's authorized
workflow dispatch, and is independently disabled by default. See
`transformers-mic/docs/cloud-baselines.md` for its CLI and operator prerequisites.

The CLI stores its eight-hour credential in
`$XDG_CONFIG_HOME/transformers-mic/dashboard-auth.json` (default
`~/.config/transformers-mic/dashboard-auth.json`), atomically written with mode
`0600`. Credentials are separate for each HTTPS dashboard origin. It rejects
insecure credential files and redirects of authenticated requests. It does not
store GitHub tokens or use the local `gh`/`hf` login.

API routes use JSON request bodies:

| Route | Purpose |
| --- | --- |
| `GET /auth/github?next=/profile` | Start browser GitHub authorization |
| `GET /auth/callback` | Consume OAuth state and create a browser session |
| `GET /auth/me` | Read browser or CLI identity; browser also sees session list |
| `POST /auth/device` | Start a grant with `client_id: "tmic"` |
| `GET /auth/cli?user_code=…` | Show the explicit browser confirmation |
| `POST /auth/cli/decision` | Browser approves/denies the displayed code |
| `POST /auth/token` | Exchange an approved device secret for a CLI credential |
| `POST /auth/sessions/revoke` | Browser revokes an owned session |
| `POST /auth/logout` | Revoke the current browser session and clear its cookie |
| `POST /auth/revoke` | CLI revokes its own bearer credential |

Without configured App credentials, sign-in returns `auth_not_configured` and
actions fail closed. Existing Grafana sessions do not bypass that condition.

## Validation

```sh
PYTHONPATH=src pytest -q tests/test_dashboard_auth.py tests/test_wdyt.py \
  tests/test_my_stuff.py tests/test_claims.py tests/test_related_issues.py \
  tests/test_rerun_failed.py
helm lint deploy/helm
```

OAuth tests mock GitHub and use real local HTTP requests for the dashboard.
After provisioning the real App callback/client secret, verify browser login,
CLI approval, profile session listing, and revocation on the deployment.

### Model-scoped baseline inputs

`tmic run MODEL` submits an immutable staged `<name>-mic` snapshot through the
MIC jobs API before final publication. `workspace.repo` must be in
`TCI_MIC_INPUT_NAMESPACES` (default `huggingface-mic`) and `workspace.revision`
must be a full Hub commit SHA. Keep the workflow's `MIC_BASELINE_INPUT_NAMESPACES`
repository variable aligned. Model requests cover both CPU and CUDA, all declared
targets and variants, and exactly one model lock; they do not modify fleet pins.
GitHub repository write access, ownership checks and submission limits still apply.
