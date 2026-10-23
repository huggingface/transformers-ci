/* GitHub identity and session controls. No provider token reaches this page. */
(() => {
  'use strict';
  const $ = id => document.getElementById(id);
  const params = new URLSearchParams(location.search);
  const code = params.get('user_code');
  if (code && /^[a-f0-9]{6}-[a-f0-9]{6}$/i.test(code)) $('user-code').value = code;
  $('sign-in').href = '/auth/github?next=' + encodeURIComponent(location.pathname + location.search);
  function status(text, error = false) { $('status').textContent = text; $('status').classList.toggle('error', error); }
  async function post(path, body = {}) {
    const response = await fetch(path, {method: 'POST', credentials: 'same-origin',
      headers: {'Content-Type': 'application/json', 'X-TCI-Action': '1'}, body: JSON.stringify(body)});
    if (!response.ok) {
      const data = await response.json();
      const messages = {invalid_or_expired_code: 'This code has expired or was already used. Run tmic auth login again.',
        rate_limited: 'Too many attempts. Wait a minute and try again.', unauthorized: 'Your session expired. Sign in again.'};
      throw new Error(messages[data.error] || 'The request could not be completed. Try again.');
    }
  }
  const date = seconds => new Date(seconds * 1000).toLocaleString();
  async function load() {
    const response = await fetch('/auth/me', {credentials: 'same-origin', cache: 'no-store'});
    $('signed-in').hidden = !response.ok;
    $('signed-out').hidden = response.status !== 401;
    if (!response.ok) {
      status(response.status === 401 ? '' : 'GitHub sign-in is not available yet. Please try again later.', response.status !== 401);
      return;
    }
    const user = await response.json();
    $('name').textContent = user.name || user.login;
    $('github').textContent = '@' + user.login;
    $('github').href = user.github_url;
    $('account-id').textContent = 'GitHub account #' + user.id;
    // Same-origin proxy is unnecessary: keep the profile free of third-party requests.
    $('sessions').replaceChildren();
    for (const session of user.sessions) {
      const row = document.createElement('div'); row.className = 'session';
      const details = document.createElement('div');
      const title = document.createElement('strong');
      title.textContent = session.label + (session.id === user.session_id ? ' · This session' : '');
      details.append(title);
      for (const text of ['Connected ' + date(session.created), 'Expires ' + date(session.expires),
        session.kind === 'cli' ? 'Profile and MIC measurement requests' : 'Dashboard browser access']) {
        const line = document.createElement('small'); line.textContent = text; details.append(line);
      }
      const revoke = document.createElement('button'); revoke.type = 'button'; revoke.textContent = 'Revoke';
      revoke.addEventListener('click', () => action(revoke, async () => {
        await post('/auth/sessions/revoke', {session_id: session.id});
        await load(); status('Session revoked.');
      }));
      row.append(details, revoke); $('sessions').append(row);
    }
    status('');
  }
  async function action(button, run) {
    button.disabled = true;
    try { await run(); } catch (error) { status(error.message, true); }
    finally { button.disabled = false; }
  }
  $('logout').addEventListener('click', event => action(event.currentTarget, async () => {
    await post('/auth/logout'); await load(); status('Signed out.');
  }));
  async function decide(button, decision) {
    await action(button, async () => {
      await post('/auth/cli/decision', {user_code: $('user-code').value.trim().toUpperCase(), decision});
      $('user-code').value = ''; history.replaceState(null, '', '/profile');
      status(decision === 'approve' ? 'tmic authorized. Return to your terminal to finish connecting.' : 'CLI access denied.');
      setTimeout(() => load().catch(() => {}), 6000);
    });
  }
  $('authorize').addEventListener('submit', event => { event.preventDefault(); decide(event.submitter, 'approve'); });
  $('deny').addEventListener('click', event => decide(event.currentTarget, 'deny'));
  load().catch(() => status('Could not load your profile. Please try again.', true));
})();
