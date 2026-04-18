// popup.js — runs inside the popup window
// When the user clicks Analyze, it sends a message to content.js
// which is already injected into the YouTube page.

const analyzeBtn = document.getElementById('analyzeBtn');
const statusEl   = document.getElementById('status');
const resultsEl  = document.getElementById('results');

// Check if API server is reachable first
async function checkAPI() {
  try {
    const res = await fetch('http://localhost:8000/', { signal: AbortSignal.timeout(2000) });
    return res.ok;
  } catch {
    return false;
  }
}

analyzeBtn.addEventListener('click', async () => {
  analyzeBtn.disabled = true;
  analyzeBtn.textContent = '⏳ Checking server...';
  statusEl.textContent = '';
  resultsEl.innerHTML = '';

  const apiOk = await checkAPI();
  if (!apiOk) {
    statusEl.textContent = '❌ API server not running. Start it with: python api.py';
    analyzeBtn.disabled = false;
    analyzeBtn.textContent = '🔍 Analyze Comments';
    return;
  }

  analyzeBtn.textContent = '⏳ Analyzing...';
  statusEl.textContent = 'Scanning comments...';

  // Get the active tab and send message to content.js
  const [tab] = await chrome.tabs.query({ active: true, currentWindow: true });

  if (!tab.url.includes('youtube.com')) {
    statusEl.textContent = '⚠️ Please open a YouTube video page first.';
    analyzeBtn.disabled = false;
    analyzeBtn.textContent = '🔍 Analyze Comments';
    return;
  }

  // Send analyze command to content script
  chrome.tabs.sendMessage(tab.id, { action: 'analyze' }, (response) => {
    analyzeBtn.disabled = false;
    analyzeBtn.textContent = '🔍 Analyze Comments';

    if (chrome.runtime.lastError || !response) {
      statusEl.textContent = '❌ Could not reach page. Try refreshing YouTube.';
      return;
    }

    const { total, toxic, error } = response;

    if (error) {
      statusEl.textContent = `❌ ${error}`;
      return;
    }

    const clean = total - toxic;
    statusEl.textContent = `Scanned ${total} comment${total !== 1 ? 's' : ''}.`;

    resultsEl.innerHTML = `
      <div class="result-row">
        <span class="result-label">💬 Total scanned</span>
        <span class="result-count count-neutral">${total}</span>
      </div>
      <div class="result-row">
        <span class="result-label">🔴 Toxic detected</span>
        <span class="result-count ${toxic > 0 ? 'count-toxic' : 'count-clean'}">${toxic}</span>
      </div>
      <div class="result-row">
        <span class="result-label">✅ Clean</span>
        <span class="result-count count-clean">${clean}</span>
      </div>
      <button id="clearBtn">✕ Clear highlights</button>
    `;

    document.getElementById('clearBtn').addEventListener('click', () => {
      chrome.tabs.sendMessage(tab.id, { action: 'clear' });
      resultsEl.innerHTML = '';
      statusEl.textContent = 'Highlights cleared.';
    });
  });
});
