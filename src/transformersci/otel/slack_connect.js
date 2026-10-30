(() => {
  'use strict';
  const status = document.getElementById('status');
  const button = document.getElementById('confirm');
  const key = 'askserge-link';
  const incoming = location.hash.slice(1);
  if (/^[A-Za-z0-9_-]{43}$/.test(incoming)) sessionStorage.setItem(key, incoming);
  history.replaceState(null, '', '/auth/slack');
  const token = sessionStorage.getItem(key);
  if (!token) { status.textContent = 'Send connect to askserge in Slack to get a connection link.'; return; }
  fetch('/auth/me', {credentials: 'same-origin', cache: 'no-store'}).then(async response => {
    if (response.status === 401) {
      status.textContent = 'Sign in to verify which GitHub account you own.';
      document.getElementById('signin').hidden = false;
      return;
    }
    if (!response.ok) throw new Error('GitHub sign-in is unavailable. Try again later.');
    const user = await response.json();
    status.textContent = 'Signed in as @' + user.login + '. Continue to receive a confirmation button in Slack.';
    button.hidden = false;
  }).catch(error => { status.textContent = error.message; });
  button.addEventListener('click', async () => {
    button.disabled = true;
    try {
      const response = await fetch('/auth/slack/connect', {method: 'POST', credentials: 'same-origin',
        headers: {'Content-Type': 'application/json', 'X-TCI-Action': '1'}, body: JSON.stringify({token})});
      const data = await response.json();
      if (!response.ok) {
        const messages = {invalid_or_expired_link: 'This link expired or was used. Send connect in Slack for a new one.',
          link_already_claimed: 'This link was already used by a different GitHub account. Start again from Slack.',
          managed_account: 'Your account is managed by the administrator.',
          unauthorized: 'Your session expired. Reload this page to sign in again.',
          rate_limited: 'Too many attempts. Wait a minute and retry.'};
        throw new Error(messages[data.error] || 'Could not send the Slack confirmation. Please retry.');
      }
      sessionStorage.removeItem(key);
      status.textContent = 'Return to your askserge Slack DM and click Confirm GitHub connection. Then retry your command.';
      button.hidden = true;
    } catch (error) { status.textContent = error.message; button.disabled = false; }
  });
})();
