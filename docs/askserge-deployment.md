# Deploying the askserge Slack agent

The agent runs in the dashboard trace exporter and delegates investigation to
Serge. No separate agent service or Slack-hosted runtime is needed. This feature
is disabled by default; merging the code does not enable it in production.

## Infra handoff

Prepare these inputs before rollout:

- Admin access to the existing askserge Slack app and permission to reinstall it.
- Slack app ID, bot user ID, workspace ID, bot token, and signing secret.
- Verified mappings from requester Slack user IDs to GitHub logins. These are
  authorization inputs, not names inferred from Slack profiles.
- Existing dashboard-to-Serge service credentials and repository allowlist.
- Durable exporter run storage and exactly one exporter replica.
- Public HTTPS ingress for `/slack/events`, `/slack/actions`, `/slack/commands`;
  these routes validate Slack signatures instead of requiring browser login.
- Exporter HTTPS egress to `slack.com`, directly or through `slack.proxy`, plus
  its existing access to Serge and repository-filtered Tempo.

Do not send tokens in chat or commit them. Add `slack-bot-token` and
`slack-signing-secret` to the dashboard's existing Kubernetes Secret through its
normal secret-management process. Preserve all existing keys.

## Rollout order

1. Publish the Serge and dashboard commits. Wait for Serge CI to publish its
   web, task-runner, and egress images from the same commit. An old task runner
   cannot honor deferred PR publication, so do not enable this with mixed images.
2. Update `serge/env/prod.yaml` in the playbooks workspace with
   `envVars.DASHBOARD_INTERACTIVE_TASKS_ENABLED: "1"`. Retain `TASK_API_ENABLED`,
   provider task opt-in, repository allowlist, dashboard service authentication,
   and existing reproduction/verification settings. From the Serge checkout at
   the published commit, deploy matching images:

   ```bash
   deploy/scripts/deploy.sh \
     --context infra:opensource-aws-use1-prod-54 \
     -n serge -f ../env/prod.yaml --from-head
   ```

3. Use [the Slack manifest](../deploy/slack-app-manifest.yaml) to configure the
   existing app, preserving any unrelated features. Enable `agent_view` and the
   listed scopes; reinstall the app after scope changes. If Slack cannot verify
   the Events URL yet, finish that step after the dashboard rollout.
4. Provision the secrets and merge the following into the existing
   `dashboard/env/private.yaml`; do not replace that file with this fragment:

   ```yaml
   traceExporter:
     replicas: 1
     sourceRevision: PUBLISHED_DASHBOARD_COMMIT_SHA
     runStore:
       enabled: true
     slack:
       enabled: true
       publicUrl: https://transformers-ci.huggingface.cool
       sergePublicUrl: https://serge.huggingface.tech
       appId: A_REPLACE_ME
       botUserId: U_BOT
       teamIds: [T_WORKSPACE]
       actors:
         "T_WORKSPACE:U_REQUESTER": verified-github-login
       proxy: ""
   ```

5. From the dashboard checkout containing the published code and chart, inspect
   the deployment plan, then deploy:

   ```bash
   deploy/scripts/deploy.py --plan -f ../env/private.yaml
   deploy/scripts/deploy.py \
     --context infra:opensource-aws-use1-prod-54 \
     -n transformers-ci -f ../env/private.yaml
   ```

6. Verify the Events URL in Slack, complete the manifest settings and installation,
   and invite askserge into the desired channels. The manifest subscribes to
   `app_mention` and `message.im`; interactive buttons use `/slack/actions`.

## Acceptance checks

- Mention `@askserge look at the traceback: <dashboard failing-test link>`.
  Confirm a private investigation thread, native processing state, phase updates,
  and a proposed fix with verification details.
- Check that the session becomes suspended while awaiting approval and that no
  PR exists before approval. Candidate branches may already exist.
- Test **No PR**, then a separate investigation with **Create PR**. Check ownership
  enforcement and that retrying approval does not create another PR.
- Test yes/no replies inside the correct investigation thread, and both dashboard
  entry points: **Fix it!** and **Deeper investigation**.
- Check `/askserge pr <number>` and a deliberate failed-CI rerun.
- Confirm an unmapped account cannot start a fix or rerun.

## Operations and limitations

Sessions refresh processing status every 20 minutes. Failed status updates retry;
phase cursors and job IDs survive exporter restarts. Command workers queued in
memory are not resumed after a crash: inspect the receipt and Serge job before
resubmitting. A phase message can be repeated if a crash occurs before its cursor
is saved. PR approval is idempotent and expires 24 hours after launch.

The native Stop control is disabled until backend cancellation is implemented.
Approvals do not keep runner pods alive. Candidate branches are retained, and
pending Serge approval records are excluded from ordinary pruning. Verification
depends on the existing task configuration; the bot reports when it was not run.

To disable new Slack traffic, set `traceExporter.slack.enabled: false` and deploy
the dashboard. Disable `DASHBOARD_INTERACTIVE_TASKS_ENABLED` in Serge to reject
new interactive task starts and decisions. Neither setting cancels work already
running. Resolve active investigations before rolling back to older runner images;
do not discard the run-store PVC or downgrade runners while interactive tasks can
still launch.
