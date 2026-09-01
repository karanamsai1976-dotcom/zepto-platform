"""A minimal browser page for the assistant.

The API is the product; this exists because a deployed service whose root path
returns a JSON 404 reads as broken to anyone who opens the link. On Hugging Face
Spaces in particular the app is shown inside an iframe, so the root path is the
first and often only thing a visitor sees.

Deliberately one self-contained string with no build step, no framework, and no
external requests. A CDN reference here would mean the page silently degrades
when the CDN is unreachable, and would add a third party to a service whose
whole point is that it has few dependencies.

It shows what the API returns rather than a polished answer: the intent, the
confidence, and the source documents with their relevance scores. Hiding those
would misrepresent the system, since the interesting part is that it declines
questions it cannot support and says how well the evidence matched.
"""

from __future__ import annotations

LANDING_PAGE = """\
<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Zepto Support Assistant</title>
<style>
  :root {
    color-scheme: light dark;
    --bg: #fbfbfa; --fg: #1a1a18; --muted: #6b6b66;
    --line: #e3e3df; --card: #ffffff; --accent: #6b46c1;
  }
  @media (prefers-color-scheme: dark) {
    :root {
      --bg: #16161a; --fg: #ececf0; --muted: #9a9aa4;
      --line: #2c2c33; --card: #1e1e24; --accent: #a78bfa;
    }
  }
  * { box-sizing: border-box; }
  body {
    margin: 0; padding: 2rem 1rem; background: var(--bg); color: var(--fg);
    font: 15px/1.6 ui-sans-serif, system-ui, -apple-system, "Segoe UI", sans-serif;
  }
  main { max-width: 46rem; margin: 0 auto; }
  h1 { font-size: 1.4rem; margin: 0 0 .25rem; letter-spacing: -.01em; }
  p.lede { color: var(--muted); margin: 0 0 1.5rem; }
  form { display: flex; gap: .5rem; margin-bottom: 1rem; }
  input {
    flex: 1; padding: .7rem .9rem; border: 1px solid var(--line);
    border-radius: 8px; background: var(--card); color: var(--fg); font: inherit;
  }
  input:focus { outline: 2px solid var(--accent); outline-offset: 1px; }
  button {
    padding: .7rem 1.2rem; border: 0; border-radius: 8px; background: var(--accent);
    color: #fff; font: inherit; font-weight: 600; cursor: pointer;
  }
  button:disabled { opacity: .55; cursor: default; }
  .examples { display: flex; flex-wrap: wrap; gap: .4rem; margin-bottom: 1.5rem; }
  .examples button {
    background: transparent; color: var(--muted); border: 1px solid var(--line);
    border-radius: 999px; padding: .3rem .8rem; font-size: .85rem; font-weight: 400;
  }
  .card {
    border: 1px solid var(--line); border-radius: 10px; background: var(--card);
    padding: 1.1rem 1.2rem;
  }
  .answer { margin: 0 0 1rem; }
  .meta { display: flex; flex-wrap: wrap; gap: 1.2rem; font-size: .85rem; color: var(--muted); }
  .badge {
    display: inline-block; padding: .15rem .5rem; border-radius: 5px;
    background: color-mix(in srgb, var(--accent) 15%, transparent); color: var(--accent);
    font-weight: 600; font-size: .8rem;
  }
  table { width: 100%; border-collapse: collapse; margin-top: 1rem; font-size: .85rem; }
  th, td { text-align: left; padding: .35rem .5rem; border-bottom: 1px solid var(--line); }
  th { color: var(--muted); font-weight: 500; }
  td:last-child, th:last-child { text-align: right; font-variant-numeric: tabular-nums; }
  footer { margin-top: 2rem; font-size: .82rem; color: var(--muted); }
  footer a { color: inherit; }
  .error { color: #c0392b; }
  @media (prefers-color-scheme: dark) { .error { color: #ff8a80; } }
</style>
</head>
<body>
<main>
  <h1>Zepto Support Assistant</h1>
  <p class="lede">
    Retrieval-augmented answers over a Zepto policy corpus. Ask a question and it
    quotes the matching policy — or declines, if the corpus does not cover it.
  </p>

  <form id="ask">
    <input id="q" placeholder="How much is the delivery fee?" autocomplete="off" required>
    <button type="submit" id="go">Ask</button>
  </form>

  <div class="examples">
    <button type="button" data-q="Can I return an opened bottle of shampoo?">returns</button>
    <button type="button" data-q="How do I track where my rider is?">tracking</button>
    <button type="button" data-q="Do you have a helpline I can dial?">support</button>
    <button type="button" data-q="Who won the world cup?">out of scope</button>
  </div>

  <div id="out"></div>

  <footer>
    JSON API at <code>POST /ask</code>. Also
    <a href="/health">/health</a>, <a href="/ready">/ready</a>,
    <a href="/metrics">/metrics</a>, <a href="/docs">/docs</a>.
    Confidence is retrieval similarity, not a calibrated probability that the
    answer is correct.
  </footer>
</main>

<script>
const form = document.getElementById('ask');
const input = document.getElementById('q');
const button = document.getElementById('go');
const out = document.getElementById('out');

document.querySelectorAll('.examples button').forEach(b => {
  b.addEventListener('click', () => {
    input.value = b.dataset.q;
    form.requestSubmit();
  });
});

const escape = s => String(s).replace(/[&<>"']/g,
  c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));

form.addEventListener('submit', async event => {
  event.preventDefault();
  const query = input.value.trim();
  if (!query) return;

  button.disabled = true;
  out.innerHTML = '<div class="card"><p class="answer">Thinking…</p></div>';

  try {
    const response = await fetch('ask', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({query})
    });

    if (response.status === 429) {
      out.innerHTML = '<div class="card"><p class="answer error">Rate limited. ' +
        'Try again in a moment.</p></div>';
      return;
    }
    if (!response.ok) {
      out.innerHTML = '<div class="card"><p class="answer error">Request failed (' +
        response.status + ').</p></div>';
      return;
    }

    const data = await response.json();
    const sources = (data.sources || []).map(s =>
      '<tr><td>' + escape(s.document_id) + '</td><td>' +
      s.relevance.toFixed(3) + '</td></tr>').join('');

    out.innerHTML =
      '<div class="card">' +
        '<p class="answer">' + escape(data.answer) + '</p>' +
        '<div class="meta">' +
          '<span class="badge">' + escape(data.intent) + '</span>' +
          '<span>confidence ' + data.confidence.toFixed(3) + '</span>' +
        '</div>' +
        (sources
          ? '<table><thead><tr><th>source</th><th>relevance</th></tr></thead>' +
            '<tbody>' + sources + '</tbody></table>'
          : '') +
      '</div>';
  } catch (err) {
    out.innerHTML = '<div class="card"><p class="answer error">' +
      'Could not reach the service.</p></div>';
  } finally {
    button.disabled = false;
  }
});
</script>
</body>
</html>
"""
