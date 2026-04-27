/* ═══════════════════════════════════════════════════════════════════
   Meeting Summariser AI — Frontend App Logic
   ═══════════════════════════════════════════════════════════════════ */

// ── State ─────────────────────────────────────────────────────────

let currentMeetingId = null;
let meetings = {};       // { id: { title, date, attendees } }
let chatHistory = [];    // [{ role, content, traces, query_type }]
let isLoading = false;

// ── DOM refs ──────────────────────────────────────────────────────

const $ = (id) => document.getElementById(id);

const els = {
  sidebar: $('sidebar'),
  meetingList: $('meetingList'),
  noMeetings: $('noMeetings'),
  welcomeScreen: $('welcomeScreen'),
  chatHeader: $('chatHeader'),
  chatTitle: $('chatTitle'),
  chatMeetingId: $('chatMeetingId'),
  quickActions: $('quickActions'),
  chatMessages: $('chatMessages'),
  chatInputArea: $('chatInputArea'),
  chatInput: $('chatInput'),
  sendBtn: $('sendBtn'),
  typingIndicator: $('typingIndicator'),
  ingestModal: $('ingestModal'),
  toastContainer: $('toastContainer'),
};

// ── Init ──────────────────────────────────────────────────────────

document.addEventListener('DOMContentLoaded', () => {
  loadMeetings();

  // Enable/disable send button based on input
  els.chatInput.addEventListener('input', () => {
    els.sendBtn.disabled = !els.chatInput.value.trim();
  });
});

// ── API helpers ───────────────────────────────────────────────────

async function api(path, opts = {}) {
  const res = await fetch(path, {
    headers: { 'Content-Type': 'application/json', ...opts.headers },
    ...opts,
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: res.statusText }));
    throw new Error(err.detail || 'Request failed');
  }
  return res.json();
}

// ── Meetings ──────────────────────────────────────────────────────

async function loadMeetings() {
  try {
    const data = await api('/api/meetings');
    meetings = {};
    data.forEach(m => { meetings[m.id] = m; });
    renderMeetingList();
  } catch (e) {
    console.error('Failed to load meetings:', e);
  }
}

function renderMeetingList() {
  const ids = Object.keys(meetings);

  // Clear existing items (keep the label + no-meetings placeholder)
  const existingItems = els.meetingList.querySelectorAll('.meeting-item');
  existingItems.forEach(el => el.remove());

  if (ids.length === 0) {
    els.noMeetings.style.display = 'block';
    return;
  }

  els.noMeetings.style.display = 'none';

  // Sort by date descending
  ids.sort((a, b) => {
    const da = meetings[a].date || '';
    const db = meetings[b].date || '';
    return db.localeCompare(da);
  });

  const label = els.meetingList.querySelector('.meetings-label');

  ids.forEach(id => {
    const m = meetings[id];
    const item = document.createElement('div');
    item.className = 'meeting-item' + (id === currentMeetingId ? ' active' : '');
    item.id = `meeting-${id}`;
    item.onclick = () => selectMeeting(id);

    const dateStr = m.date ? new Date(m.date).toLocaleDateString('en-US', { month: 'short', day: 'numeric' }) : '';

    item.innerHTML = `
      <span class="meeting-icon">💬</span>
      <div class="meeting-info">
        <div class="meeting-title">${escapeHtml(m.title)}</div>
        <div class="meeting-meta">${dateStr} · ${id.slice(0, 8)}</div>
      </div>
      <button class="meeting-delete" onclick="event.stopPropagation(); deleteMeeting('${id}')" title="Delete">🗑</button>
    `;

    label.insertAdjacentElement('afterend', item);
  });
}

async function selectMeeting(id) {
  currentMeetingId = id;
  const m = meetings[id];

  // Update UI
  els.welcomeScreen.style.display = 'none';
  els.chatHeader.style.display = 'flex';
  els.quickActions.style.display = 'flex';
  els.chatMessages.style.display = 'block';
  els.chatInputArea.style.display = 'block';

  els.chatTitle.textContent = m.title;
  els.chatMeetingId.textContent = id.slice(0, 12);

  // Mark active in sidebar
  document.querySelectorAll('.meeting-item').forEach(el => el.classList.remove('active'));
  const active = document.getElementById(`meeting-${id}`);
  if (active) active.classList.add('active');

  // Load chat history for this meeting
  await loadChatHistory(id);
}

async function loadChatHistory(meetingId) {
  try {
    const data = await api(`/api/meetings/${meetingId}/history`);
    chatHistory = data.messages || [];
    renderChatHistory();
  } catch {
    chatHistory = [];
    renderChatHistory();
  }
}

async function deleteMeeting(id) {
  if (!confirm('Delete this meeting and all its chat history?')) return;
  try {
    await api(`/api/meetings/${id}`, { method: 'DELETE' });
    delete meetings[id];
    if (currentMeetingId === id) {
      currentMeetingId = null;
      showWelcome();
    }
    renderMeetingList();
    showToast('Meeting deleted', 'success');
  } catch (e) {
    showToast('Failed to delete: ' + e.message, 'error');
  }
}

function showWelcome() {
  els.welcomeScreen.style.display = 'flex';
  els.chatHeader.style.display = 'none';
  els.quickActions.style.display = 'none';
  els.chatMessages.style.display = 'none';
  els.chatInputArea.style.display = 'none';
  els.chatMessages.innerHTML = '';
}

// ── Chat ──────────────────────────────────────────────────────────

function renderChatHistory() {
  els.chatMessages.innerHTML = '';
  chatHistory.forEach(msg => appendMessageToDOM(msg));
  scrollToBottom();
}

function appendMessageToDOM(msg) {
  const div = document.createElement('div');
  div.className = `message ${msg.role}`;

  const isUser = msg.role === 'user';
  const avatar = isUser ? '👤' : '✦';
  const sender = isUser ? 'You' : 'AI';

  let bodyHtml = '';
  if (isUser) {
    bodyHtml = `<p>${escapeHtml(msg.content)}</p>`;
  } else {
    bodyHtml = renderMarkdown(msg.content);
  }

  div.innerHTML = `
    <div class="message-header">
      <div class="message-avatar">${avatar}</div>
      <span class="message-sender">${sender}</span>
    </div>
    <div class="message-body">${bodyHtml}</div>
  `;

  // Add trace toggle if traces exist
  if (msg.traces && msg.traces.length > 0) {
    const traceId = 'trace-' + Math.random().toString(36).slice(2, 9);

    const toggleBtn = document.createElement('button');
    toggleBtn.className = 'trace-toggle';
    toggleBtn.innerHTML = '🔍 Agent Trace';
    toggleBtn.onclick = () => {
      const content = document.getElementById(traceId);
      content.classList.toggle('open');
      toggleBtn.innerHTML = content.classList.contains('open') ? '🔍 Hide Trace' : '🔍 Agent Trace';
    };

    const traceDiv = document.createElement('div');
    traceDiv.className = 'trace-content';
    traceDiv.id = traceId;

    const icons = {
      classify_query: '⚡', classify_result: '⚡',
      hyde_expansion: '🔍', expand_and_retrieve: '🔍', retrieval_result: '🔍',
      compress_context: '🗜️', compression_done: '🗜️',
      summary_agent: '📋', summary_done: '📋',
      action_items_agent: '✅', action_items_done: '✅',
      decisions_agent: '🎯', decisions_done: '🎯',
      qa_agent: '💬', qa_done: '💬',
    };

    msg.traces.forEach(t => {
      const icon = icons[t.step] || '○';
      traceDiv.innerHTML += `
        <div class="trace-step">
          <span class="step-icon">${icon}</span>
          <span class="step-name">${escapeHtml(t.step)}</span>
          <span class="step-detail">${escapeHtml(t.detail)}</span>
        </div>
      `;
    });

    div.appendChild(toggleBtn);
    div.appendChild(traceDiv);
  }

  els.chatMessages.appendChild(div);
}

async function sendMessage() {
  const query = els.chatInput.value.trim();
  if (!query || !currentMeetingId || isLoading) return;

  isLoading = true;
  els.chatInput.value = '';
  els.sendBtn.disabled = true;
  autoResize(els.chatInput);

  // Add user message
  const userMsg = { role: 'user', content: query };
  chatHistory.push(userMsg);
  appendMessageToDOM(userMsg);
  scrollToBottom();

  // Show typing indicator
  els.typingIndicator.classList.add('visible');
  scrollToBottom();

  try {
    const result = await api('/api/chat', {
      method: 'POST',
      body: JSON.stringify({ meeting_id: currentMeetingId, query }),
    });

    // Hide typing
    els.typingIndicator.classList.remove('visible');

    const aiMsg = {
      role: 'assistant',
      content: result.content,
      traces: result.traces || [],
      query_type: result.query_type,
    };

    chatHistory.push(aiMsg);
    appendMessageToDOM(aiMsg);
    scrollToBottom();

  } catch (e) {
    els.typingIndicator.classList.remove('visible');

    const errMsg = {
      role: 'assistant',
      content: `❌ Error: ${e.message}`,
      traces: [],
    };
    chatHistory.push(errMsg);
    appendMessageToDOM(errMsg);
    scrollToBottom();
    showToast('Pipeline error: ' + e.message, 'error');
  } finally {
    isLoading = false;
    els.sendBtn.disabled = !els.chatInput.value.trim();
  }
}

function sendQuickAction(query) {
  if (isLoading || !currentMeetingId) return;
  els.chatInput.value = query;
  sendMessage();
}

// ── Input handling ────────────────────────────────────────────────

function handleInputKeydown(e) {
  if (e.key === 'Enter' && !e.shiftKey) {
    e.preventDefault();
    sendMessage();
  }
}

function autoResize(textarea) {
  textarea.style.height = 'auto';
  textarea.style.height = Math.min(textarea.scrollHeight, 120) + 'px';
}

function scrollToBottom() {
  requestAnimationFrame(() => {
    els.chatMessages.scrollTop = els.chatMessages.scrollHeight;
  });
}

// ── Ingest Modal ──────────────────────────────────────────────────

function openIngestModal() {
  els.ingestModal.classList.add('open');
  $('ingestTitle').value = '';
  $('ingestAttendees').value = '';
  $('ingestText').value = '';
  $('ingestFile').value = '';
}

function closeIngestModal() {
  els.ingestModal.classList.remove('open');
}

function switchIngestTab(tab) {
  $('tabPaste').classList.toggle('active', tab === 'paste');
  $('tabUpload').classList.toggle('active', tab === 'upload');
  $('ingestPasteTab').style.display = tab === 'paste' ? 'block' : 'none';
  $('ingestUploadTab').style.display = tab === 'upload' ? 'block' : 'none';
}

async function ingestMeeting() {
  const title = $('ingestTitle').value.trim();
  const attendees = $('ingestAttendees').value.trim();
  const isPaste = $('tabPaste').classList.contains('active');

  const btnIngest = $('btnIngest');
  btnIngest.disabled = true;
  btnIngest.innerHTML = '<span class="loading-spinner"></span> Ingesting...';

  try {
    if (isPaste) {
      const text = $('ingestText').value.trim();
      if (!text) throw new Error('Please paste a transcript');

      const result = await api('/api/meetings/ingest', {
        method: 'POST',
        body: JSON.stringify({
          title: title || 'Untitled Meeting',
          attendees: attendees ? attendees.split(',').map(a => a.trim()) : [],
          text,
        }),
      });

      meetings[result.id] = result;
      renderMeetingList();
      selectMeeting(result.id);
      showToast(`Ingested: ${result.title}`, 'success');

    } else {
      const file = $('ingestFile').files[0];
      if (!file) throw new Error('Please select a file');

      const formData = new FormData();
      formData.append('file', file);
      formData.append('title', title || file.name.replace(/\.[^.]+$/, ''));
      formData.append('attendees', attendees);

      const res = await fetch('/api/meetings/ingest-file', {
        method: 'POST',
        body: formData,
      });

      if (!res.ok) {
        const err = await res.json().catch(() => ({ detail: res.statusText }));
        throw new Error(err.detail || 'Upload failed');
      }

      const result = await res.json();
      meetings[result.id] = result;
      renderMeetingList();
      selectMeeting(result.id);
      showToast(`Ingested: ${result.title}`, 'success');
    }

    closeIngestModal();

  } catch (e) {
    showToast(e.message, 'error');
  } finally {
    btnIngest.disabled = false;
    btnIngest.innerHTML = '⚡ Ingest';
  }
}

// ── Markdown Renderer (lightweight) ──────────────────────────────

function renderMarkdown(text) {
  if (!text) return '<p></p>';

  let html = escapeHtml(text);

  // Headers
  html = html.replace(/^### (.+)$/gm, '<h3>$1</h3>');
  html = html.replace(/^## (.+)$/gm, '<h2>$1</h2>');

  // Bold
  html = html.replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>');

  // Italic
  html = html.replace(/\*(.+?)\*/g, '<em>$1</em>');

  // Inline code
  html = html.replace(/`([^`]+)`/g, '<code>$1</code>');

  // Blockquotes
  html = html.replace(/^&gt;\s?(.+)$/gm, '<blockquote>$1</blockquote>');

  // Unordered lists: collect consecutive "- " lines
  html = html.replace(/((?:^- .+\n?)+)/gm, (match) => {
    const items = match.trim().split('\n').map(l => `<li>${l.replace(/^- /, '')}</li>`).join('');
    return `<ul>${items}</ul>`;
  });

  // Ordered lists
  html = html.replace(/((?:^\d+[.)]\s.+\n?)+)/gm, (match) => {
    const items = match.trim().split('\n').map(l => `<li>${l.replace(/^\d+[.)]\s/, '')}</li>`).join('');
    return `<ol>${items}</ol>`;
  });

  // Horizontal rules
  html = html.replace(/^---$/gm, '<hr>');

  // Paragraphs — wrap bare text lines
  html = html.replace(/^(?!<[a-z])((?!<\/)[^\n]+)$/gm, '<p>$1</p>');

  // Clean up empty paragraphs
  html = html.replace(/<p>\s*<\/p>/g, '');

  return html;
}

// ── Utilities ─────────────────────────────────────────────────────

function escapeHtml(text) {
  const div = document.createElement('div');
  div.textContent = text;
  return div.innerHTML;
}

function showToast(message, type = 'info') {
  const toast = document.createElement('div');
  toast.className = `toast ${type}`;
  toast.textContent = message;
  els.toastContainer.appendChild(toast);
  setTimeout(() => toast.remove(), 4000);
}
