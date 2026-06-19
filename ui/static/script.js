// ─── Tab switching ───
document.querySelectorAll(".tab-btn").forEach(btn => {
  btn.addEventListener("click", () => {
    document.querySelectorAll(".tab-btn").forEach(b => b.classList.remove("active"));
    document.querySelectorAll(".tab-content").forEach(c => c.classList.remove("active"));
    btn.classList.add("active");
    document.getElementById("tab-" + btn.dataset.tab).classList.add("active");
  });
});

// ─── Sliders ───
const sliders = [
  { id: "temp", valId: "tempVal", parse: parseFloat },
  { id: "top_k", valId: "top_kVal", parse: parseInt },
  { id: "top_p", valId: "top_pVal", parse: parseFloat },
  { id: "max_new", valId: "max_newVal", parse: parseInt },
];

sliders.forEach(({ id, valId }) => {
  const input = document.getElementById(id);
  const val = document.getElementById(valId);
  if (input && val) {
    const update = () => { val.textContent = input.value; };
    input.addEventListener("input", update);
    update();
  }
});

// ─── Chat ───
const messagesEl = document.getElementById("messages");
const promptEl = document.getElementById("prompt");
const sendBtn = document.getElementById("sendBtn");

function escapeHtml(text) {
  const d = document.createElement("div");
  d.textContent = text;
  return d.innerHTML;
}

function addMessage(text, role) {
  const div = document.createElement("div");
  div.className = `msg msg-${role}`;

  const avatar = role === "user"
    ? '<div class="msg-avatar"><svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M20 21v-2a4 4 0 0 0-4-4H8a4 4 0 0 0-4 4v2"/><circle cx="12" cy="7" r="4"/></svg></div>'
    : '<div class="msg-avatar"><svg width="20" height="20" viewBox="0 0 32 32" fill="none"><rect width="32" height="32" rx="8" fill="url(#g)"/><path d="M10 22V12l6 6 6-6v10" stroke="#fff" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"/><defs><linearGradient id="g" x1="0" y1="0" x2="32" y2="32"><stop stop-color="#6366f1"/><stop offset="1" stop-color="#8b5cf6"/></linearGradient></defs></svg></div>';

  div.innerHTML = `${avatar}<div class="msg-bubble"><p>${escapeHtml(text)}</p></div>`;
  messagesEl.appendChild(div);
  messagesEl.scrollTop = messagesEl.scrollHeight;
}

function showLoading() {
  const div = document.createElement("div");
  div.className = "msg msg-bot";
  div.innerHTML = '<div class="msg-avatar"><svg width="20" height="20" viewBox="0 0 32 32" fill="none"><rect width="32" height="32" rx="8" fill="url(#g)"/><path d="M10 22V12l6 6 6-6v10" stroke="#fff" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"/><defs><linearGradient id="g" x1="0" y1="0" x2="32" y2="32"><stop stop-color="#6366f1"/><stop offset="1" stop-color="#8b5cf6"/></linearGradient></defs></svg></div><div class="msg-bubble loading"><p>Thinking...</p></div>';
  div.id = "loadingMsg";
  messagesEl.appendChild(div);
  messagesEl.scrollTop = messagesEl.scrollHeight;
}

function removeLoading() {
  const el = document.getElementById("loadingMsg");
  if (el) el.remove();
}

async function sendMessage() {
  const prompt = promptEl.value.trim();
  if (!prompt) return;

  addMessage(prompt, "user");
  promptEl.value = "";
  sendBtn.disabled = true;
  showLoading();

  try {
    const res = await fetch("/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        prompt,
        max_new: parseInt(document.getElementById("max_new").value) || 100,
        temp: parseFloat(document.getElementById("temp").value) || 0.5,
        top_k: parseInt(document.getElementById("top_k").value) || 20,
        top_p: parseFloat(document.getElementById("top_p").value) || 0.9,
      }),
    });
    const data = await res.json();
    removeLoading();
    if (data.error) {
      addMessage(`Error: ${escapeHtml(data.error)}`, "bot");
    } else {
      addMessage(escapeHtml(data.response), "bot");
    }
  } catch (err) {
    removeLoading();
    addMessage(`Error: ${escapeHtml(err.message)}`, "bot");
  } finally {
    sendBtn.disabled = false;
    promptEl.focus();
  }
}

sendBtn.addEventListener("click", sendMessage);
promptEl.addEventListener("keydown", (e) => {
  if (e.key === "Enter") sendMessage();
});
