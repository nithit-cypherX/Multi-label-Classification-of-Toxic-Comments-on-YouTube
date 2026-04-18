// content.js — injected into every YouTube page

const API_URL    = 'http://localhost:8000/analyze_batch';
const BATCH_SIZE = 10;

const LABEL_CONFIG = {
  toxic:         { emoji: '🔴', text: 'Toxic',        color: '#ff4444' },
  severe_toxic:  { emoji: '🚨', text: 'Severe Toxic', color: '#cc0000' },
  obscene:       { emoji: '🤬', text: 'Obscene',      color: '#ff6600' },
  threat:        { emoji: '⚠️', text: 'Threat',       color: '#ff0000' },
  insult:        { emoji: '😠', text: 'Insult',       color: '#ff8800' },
  identity_hate: { emoji: '🏴', text: 'Hate Speech',  color: '#9900cc' },
};

// ── Find comment elements ────────────────────────────────────────────────────

function getCommentElements() {
  const selectors = [
    'ytd-comment-renderer #content-text',
    'ytd-comment-view-model #content-text',
    '#comments ytd-comment-thread-renderer #content-text',
    'yt-attributed-string.yt-core-attributed-string',
  ];
  for (const selector of selectors) {
    const els = Array.from(document.querySelectorAll(selector))
      .filter(el => el.textContent.trim().length > 5);
    if (els.length > 0) return els;
  }
  return [];
}

// ── API call ─────────────────────────────────────────────────────────────────

async function analyzeBatch(texts) {
  const res = await fetch(API_URL, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(texts),
  });
  if (!res.ok) throw new Error(`API error ${res.status}`);
  return res.json();
}

// ── Annotate a single comment element ────────────────────────────────────────

function annotateComment(el, result) {
  const container = el.closest('ytd-comment-renderer')
    || el.closest('ytd-comment-view-model')
    || el.parentElement;

  const oldBadge = container.querySelector('.toxic-badge');
  if (oldBadge) oldBadge.remove();
  container.classList.remove('toxic-comment-highlight');

  if (result.is_toxic) {
    container.classList.add('toxic-comment-highlight');
    const badge = document.createElement('div');
    badge.className = 'toxic-badge';
    result.labels.forEach(label => {
      const cfg  = LABEL_CONFIG[label] || { emoji: '⚠️', text: label, color: '#888' };
      const chip = document.createElement('span');
      chip.className   = 'toxic-chip';
      chip.textContent = `${cfg.emoji} ${cfg.text}`;
      chip.style.borderColor = cfg.color;
      chip.style.color       = cfg.color;
      badge.appendChild(chip);
    });
    el.parentNode.insertBefore(badge, el);
  }

  el.dataset.toxicProcessed = Date.now();
}

// ── Clear all annotations ────────────────────────────────────────────────────

function clearAnnotations() {
  document.querySelectorAll('.toxic-comment-highlight')
    .forEach(el => el.classList.remove('toxic-comment-highlight'));
  document.querySelectorAll('.toxic-badge')
    .forEach(el => el.remove());
  document.querySelectorAll('[data-toxic-processed]')
    .forEach(el => delete el.dataset.toxicProcessed);
}

// ── Pretty console table ─────────────────────────────────────────────────────

function logResults(commentEls, results) {
  const toxic = results.filter(r => r.is_toxic);
  const clean = results.filter(r => !r.is_toxic);

  console.log('%c━━━ Toxic Comment Detector Results ━━━', 'color:#ff4444;font-weight:bold;font-size:14px');
  console.log(`%cScanned: ${results.length}  |  🔴 Toxic: ${toxic.length}  |  ✅ Clean: ${clean.length}`,
    'color:#aaa;font-size:12px');
  console.log('');

  // Summary table of all comments
  const tableData = results.map((r, i) => ({
    '#':            i + 1,
    'Comment':      commentEls[i].textContent.trim().substring(0, 50) + '...',
    'Labels':       r.labels.join(', ') || '✅ clean',
    'toxic':        r.probabilities.toxic?.toFixed(4),
    'severe_toxic': r.probabilities.severe_toxic?.toFixed(4),
    'obscene':      r.probabilities.obscene?.toFixed(4),
    'threat':       r.probabilities.threat?.toFixed(4),
    'insult':       r.probabilities.insult?.toFixed(4),
    'identity_hate':r.probabilities.identity_hate?.toFixed(4),
  }));
  console.table(tableData);

  // Detailed breakdown of toxic comments only
  if (toxic.length > 0) {
    console.log('%c━━━ Toxic Comments Detail ━━━', 'color:#ff4444;font-weight:bold');
    toxic.forEach((r, i) => {
      const idx = results.indexOf(r);
      const text = commentEls[idx].textContent.trim().substring(0, 80);
      console.group(`%c🔴 [${idx + 1}] "${text}"`, 'color:#ff8888');
      console.log('%cLabels fired:', 'color:#ffaa44;font-weight:bold', r.labels);
      console.log('%cAll probabilities:', 'color:#aaa');
      Object.entries(r.probabilities).forEach(([label, prob]) => {
        const fired = r.labels.includes(label);
        const bar = '█'.repeat(Math.round(prob * 200)).padEnd(20, '░');
        console.log(
          `%c  ${label.padEnd(15)} ${bar} ${prob.toFixed(4)}${fired ? ' ← FIRED' : ''}`,
          fired ? 'color:#ff4444;font-weight:bold' : 'color:#888'
        );
      });
      console.groupEnd();
    });
  }

  console.log('%c━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━', 'color:#333');
}

// ── Main analyze ─────────────────────────────────────────────────────────────

async function analyzeComments() {
  const commentEls = getCommentElements();

  if (commentEls.length === 0) {
    return {
      total: 0,
      toxic: 0,
      error: 'No comments found. Scroll down until comments are visible, then try again.',
    };
  }

  const texts = commentEls.map(el => el.textContent.trim());
  const allResults = [];

  for (let i = 0; i < commentEls.length; i += BATCH_SIZE) {
    const batchEls   = commentEls.slice(i, i + BATCH_SIZE);
    const batchTexts = texts.slice(i, i + BATCH_SIZE);
    try {
      const results = await analyzeBatch(batchTexts);
      results.forEach((result, j) => {
        annotateComment(batchEls[j], result);
        allResults.push(result);
      });
    } catch (err) {
      return { total: commentEls.length, toxic: allResults.filter(r=>r.is_toxic).length, error: `API error: ${err.message}` };
    }
  }

  // Print detailed stats to console
  logResults(commentEls, allResults);

  const toxicCount = allResults.filter(r => r.is_toxic).length;
  return { total: commentEls.length, toxic: toxicCount };
}

// ── Message listener ─────────────────────────────────────────────────────────

chrome.runtime.onMessage.addListener((message, sender, sendResponse) => {
  if (message.action === 'analyze') {
    analyzeComments()
      .then(result => sendResponse(result))
      .catch(err   => sendResponse({ total: 0, toxic: 0, error: err.message }));
    return true;
  }
  if (message.action === 'clear') {
    clearAnnotations();
    sendResponse({ ok: true });
  }
});