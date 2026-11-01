(() => {
  "use strict";

  // Set this to the URL where your FastAPI backend is running.
  const API_BASE_URL = "http://127.0.0.1:8000";
  const CHAT_ENDPOINT = `${API_BASE_URL.replace(/\/$/, "")}/chat`;

  // sessionStorage keeps the conversation for this browser tab, including refreshes.
  // Closing the tab ends this browser-session persistence.
  const THREAD_KEY = "proxigen_chat_thread_id";
  const HISTORY_KEY = "proxigen_chat_messages";

  const root = document.getElementById("proxigen-chat-root");
  if (!root) return;

  const icon = (name) => {
    const paths = {
      chat: '<path d="M21 11.5a8.4 8.4 0 0 1-.9 3.8 8.5 8.5 0 0 1-7.6 4.7 8.4 8.4 0 0 1-3.8-.9L3 21l1.9-5.7a8.4 8.4 0 0 1-.9-3.8A8.5 8.5 0 0 1 8.7 3.9a8.4 8.4 0 0 1 3.8-.9h.5a8.5 8.5 0 0 1 8 8v.5z"/>',
      close: '<path d="m18 6-12 12M6 6l12 12"/>',
      send: '<path d="m22 2-7 20-4-9-9-4Z"/><path d="M22 2 11 13"/>',
      new: '<path d="M12 5v14M5 12h14"/>'
    };
    return `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">${paths[name] || ""}</svg>`;
  };

  root.innerHTML = `
    <button class="px-launcher" type="button" aria-expanded="false" aria-controls="px-chat-panel">
      ${icon("chat")} <span>Chat with us</span>
    </button>
    <section class="px-panel" id="px-chat-panel" role="dialog" aria-label="Proxigen support chat" aria-modal="false">
      <header class="px-header">
        <div class="px-brand">
          <div class="px-logo" aria-hidden="true">P</div>
          <div class="px-heading">
            <strong>Proxigen Assistant</strong>
            <span>Ask us about our products</span>
          </div>
        </div>
        <div class="px-header-actions">
          <button class="px-icon-btn px-new-chat" type="button" aria-label="Start a new conversation" title="New conversation">${icon("new")}</button>
          <button class="px-icon-btn px-close" type="button" aria-label="Close chat">${icon("close")}</button>
        </div>
      </header>
      <div class="px-messages" aria-live="polite">
        <div class="px-welcome">Hello! How can I help you today? I'll use available Proxigen information to answer your questions.</div>
      </div>
      <div class="px-typing" hidden>Thinking…</div>
      <div class="px-composer-wrap">
        <p class="px-error" role="alert"></p>
        <form class="px-composer">
          <textarea class="px-input" rows="1" maxlength="2000" placeholder="Write your question…" aria-label="Your question"></textarea>
          <button class="px-send" type="submit" aria-label="Send message">${icon("send")}</button>
        </form>
        <p class="px-disclaimer">Please don't share passwords or sensitive personal information.</p>
      </div>
    </section>
  `;

  const launcher = root.querySelector(".px-launcher");
  const panel = root.querySelector(".px-panel");
  const closeButton = root.querySelector(".px-close");
  const newChatButton = root.querySelector(".px-new-chat");
  const messagesEl = root.querySelector(".px-messages");
  const form = root.querySelector(".px-composer");
  const input = root.querySelector(".px-input");
  const sendButton = root.querySelector(".px-send");
  const typing = root.querySelector(".px-typing");
  const errorEl = root.querySelector(".px-error");

  let busy = false;
  let history = readHistory();

  function readHistory() {
    try {
      const value = JSON.parse(sessionStorage.getItem(HISTORY_KEY) || "[]");
      return Array.isArray(value) ? value.filter(item =>
        item && (item.role === "user" || item.role === "assistant") && typeof item.content === "string"
      ) : [];
    } catch {
      return [];
    }
  }

  function persistHistory() {
    try {
      sessionStorage.setItem(HISTORY_KEY, JSON.stringify(history.slice(-100)));
    } catch {
      // Chat still works if browser storage is unavailable or full.
    }
  }

  function getThreadId() {
    let id = null;
    try { id = sessionStorage.getItem(THREAD_KEY); } catch {}
    if (!id) {
      id = (crypto && crypto.randomUUID) ? crypto.randomUUID() :
        "px-" + Date.now().toString(36) + "-" + Math.random().toString(36).slice(2);
      try { sessionStorage.setItem(THREAD_KEY, id); } catch {}
    }
    return id;
  }

  function showError(message) {
    errorEl.textContent = message;
    errorEl.classList.toggle("is-visible", Boolean(message));
  }

  


function cleanAssistantMarkdown(content) {
  let text = String(content ?? "");

  text = text.replace(/\r\n?/g, "\n");
  text = text.replace(/[ \t]+\n/g, "\n");
  text = text.replace(/\n{3,}/g, "\n\n");

  // Remove bold and italic markers while preserving their text.
  text = text.replace(/\*\*\*(.*?)\*\*\*/gs, "$1");
  text = text.replace(/\*\*(.*?)\*\*/gs, "$1");
  text = text.replace(/__(.*?)__/gs, "$1");
  text = text.replace(/(?<!\*)\*([^*\n]+)\*(?!\*)/g, "$1");
  text = text.replace(/(?<!_)_([^_\n]+)_(?!_)/g, "$1");

  // Remove citation labels.
  text = text.replace(
    /\s*\[(?:source|citation|reference)\s*\d+\]/gi,
    ""
  );

  return text.trim();
}

function addMessage(role, content, save = true) {
  const row = document.createElement("div");
  row.className = `px-message ${role}`;

  const bubble = document.createElement("div");
  bubble.className = "px-bubble";

  if (role === "assistant") {
    const cleanContent = cleanAssistantMarkdown(content);

    if (window.marked && window.DOMPurify) {
      const html = window.marked.parse(cleanContent, {
        breaks: false,
        gfm: true
      });

      bubble.innerHTML = window.DOMPurify.sanitize(html);
    } else {
      // Safe fallback if Markdown libraries aren't loaded.
      bubble.textContent = cleanContent;
      bubble.style.whiteSpace = "pre-wrap";
    }

    if (save) {
      history.push({ role, content: cleanContent });
      persistHistory();
    }
  } else {
    // User messages are always plain text.
    bubble.textContent = content;

    if (save) {
      history.push({ role, content });
      persistHistory();
    }
  }

  row.appendChild(bubble);
  messagesEl.appendChild(row);
  messagesEl.scrollTop = messagesEl.scrollHeight;
}



  function renderHistory() {
    messagesEl.innerHTML = "";
    if (!history.length) {
      const welcome = document.createElement("div");
      welcome.className = "px-welcome";
      welcome.textContent = "Hello! How can I help you today? I'll use available Proxigen information to answer your questions.";
      messagesEl.appendChild(welcome);
    } else {
      history.forEach(message => addMessage(message.role, message.content, false));
    }
    messagesEl.scrollTop = messagesEl.scrollHeight;
  }

  function setBusy(value) {
    busy = value;
    typing.hidden = !value;
    sendButton.disabled = value;
    input.disabled = value;
  }

  function openChat() {
    panel.classList.add("is-open");
    launcher.setAttribute("aria-expanded", "true");
    input.focus();
  }

  function closeChat() {
    panel.classList.remove("is-open");
    launcher.setAttribute("aria-expanded", "false");
    launcher.focus();
  }

  async function sendQuestion(question) {
    showError("");
    addMessage("user", question);
    setBusy(true);

    try {
      const response = await fetch(CHAT_ENDPOINT, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          question,
          thread_id: getThreadId()
        })
      });

      let data;
      try { data = await response.json(); } catch { data = null; }

      if (!response.ok) {
        const detail = data && typeof data.detail === "string" ? data.detail : "";
        throw new Error(detail || `The server returned an error (${response.status}).`);
      }

      if (!data || typeof data.answer !== "string") {
        throw new Error("The server response was missing an answer.");
      }

      if (typeof data.thread_id === "string" && data.thread_id) {
        try { sessionStorage.setItem(THREAD_KEY, data.thread_id); } catch {}
      }
      addMessage("assistant", data.answer);
    } catch (error) {
      showError(
        "Couldn't reach the chat service. Check that the backend is running and CORS is configured. " +
        (error instanceof TypeError ? "" : (error.message || ""))
      );
      // Keep the question visible in the conversation. The user can retry by sending it again.
    } finally {
      setBusy(false);
      input.focus();
    }
  }

  launcher.addEventListener("click", () => {
    if (panel.classList.contains("is-open")) closeChat();
    else openChat();
  });
  closeButton.addEventListener("click", closeChat);
  newChatButton.addEventListener("click", () => {
    if (busy) return;
    history = [];
    try {
      sessionStorage.removeItem(THREAD_KEY);
      sessionStorage.removeItem(HISTORY_KEY);
    } catch {}
    getThreadId();
    renderHistory();
    showError("");
    input.focus();
  });

  form.addEventListener("submit", async event => {
    event.preventDefault();
    if (busy) return;
    const question = input.value.trim();
    if (!question) return;
    input.value = "";
    input.style.height = "auto";
    await sendQuestion(question);
  });

  input.addEventListener("input", () => {
    input.style.height = "auto";
    input.style.height = `${Math.min(input.scrollHeight, 100)}px`;
  });

  input.addEventListener("keydown", event => {
    if (event.key === "Enter" && !event.shiftKey) {
      event.preventDefault();
      form.requestSubmit();
    }
  });

  renderHistory();
})();
