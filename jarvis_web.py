"""
╔══════════════════════════════════════════════════════════════════════════════╗
║   J.A.R.V.I.S. Web Interface                                                 ║
║   Access JARVIS from any device on your home Wi-Fi                           ║
║                                                                               ║
║   Run:   .venv\\Scripts\\python.exe jarvis_web.py                             ║
║   Then open on phone:  http://<YOUR-PC-IP>:5000                              ║
╚══════════════════════════════════════════════════════════════════════════════╝

Find your PC IP:  Run `ipconfig` in PowerShell, look for IPv4 Address.
Example phone URL: http://192.168.1.5:5000
"""

import os
import sys
import time
import datetime
import socket
import threading
import webbrowser

# Force UTF-8 output
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# ─────────────────────────────────────────────────────────────────────────────
# THIRD-PARTY IMPORTS
# ─────────────────────────────────────────────────────────────────────────────
import pyttsx3
import requests as http_requests
import wikipedia
import ollama
import chromadb
from flask import Flask, request, jsonify, render_template_string

# Optional
try:
    import pywhatkit as kit
    WHATSAPP_AVAILABLE = True
except Exception:
    WHATSAPP_AVAILABLE = False

# ══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ══════════════════════════════════════════════════════════════════════════════

NEWSAPI_KEY       = "YOUR_NEWSAPI_KEY"
OLLAMA_MODEL      = "llama3.2:1b"
MAX_HISTORY       = 5
SCRIPT_DIR        = os.path.dirname(os.path.abspath(__file__))
CHROMA_DB_PATH    = os.path.join(SCRIPT_DIR, "jarvis_memory")
CHROMA_COLLECTION = "personal_knowledge"
TTS_RATE          = 190
PORT              = 5000

# ══════════════════════════════════════════════════════════════════════════════
# TTS ENGINE  (speaks through PC speakers)
# ══════════════════════════════════════════════════════════════════════════════

engine = pyttsx3.init()
engine.setProperty("rate", TTS_RATE)
voices = engine.getProperty("voices")
if voices:
    chosen = voices[0]
    for v in voices:
        if "male" in v.name.lower() or "david" in v.name.lower():
            chosen = v
            break
    engine.setProperty("voice", chosen.id)

tts_lock = threading.Lock()   # Prevent concurrent TTS calls from multiple requests

def speak(text: str) -> None:
    """Thread-safe TTS — plays audio through PC speakers."""
    with tts_lock:
        engine.say(text)
        engine.runAndWait()

def speak_async(text: str) -> None:
    """Fire-and-forget TTS so the HTTP response isn't blocked."""
    t = threading.Thread(target=speak, args=(text,), daemon=True)
    t.start()

# ══════════════════════════════════════════════════════════════════════════════
# CHROMADB MEMORY BANK
# ══════════════════════════════════════════════════════════════════════════════

try:
    chroma_client     = chromadb.PersistentClient(path=CHROMA_DB_PATH)
    memory_collection = chroma_client.get_or_create_collection(
        name=CHROMA_COLLECTION,
        metadata={"hnsw:space": "cosine"}
    )
    MEMORY_AVAILABLE = True
    print(f"[BOOT] Memory bank ready. Docs stored: {memory_collection.count()}")
except Exception as e:
    memory_collection = None
    MEMORY_AVAILABLE  = False
    print(f"[BOOT] ChromaDB unavailable: {e}")

# ══════════════════════════════════════════════════════════════════════════════
# CHAT HISTORY & RAG
# ══════════════════════════════════════════════════════════════════════════════

chat_history: list[dict] = []

def update_history(role: str, content: str) -> None:
    chat_history.append({"role": role, "content": content})
    max_msgs = MAX_HISTORY * 2
    if len(chat_history) > max_msgs:
        del chat_history[:len(chat_history) - max_msgs]

def query_memory(text: str, n: int = 3) -> str:
    if not MEMORY_AVAILABLE or not memory_collection or memory_collection.count() == 0:
        return ""
    try:
        res  = memory_collection.query(query_texts=[text], n_results=min(n, memory_collection.count()))
        docs = res.get("documents", [[]])[0]
        return "\n\n".join(f"[Memory {i+1}]: {d}" for i, d in enumerate(docs)) if docs else ""
    except Exception as e:
        print(f"[MEMORY] Query error: {e}")
        return ""

BASE_SYSTEM_PROMPT = (
    "You are J.A.R.V.I.S., an advanced, witty, and highly capable AI assistant. "
    "You speak in a calm, professional, slightly British tone with occasional dry humour. "
    "Keep responses concise (1-3 sentences unless asked for more). "
    "You are running fully locally on the user's machine."
)

def ask_ollama(user_text: str) -> str:
    context = query_memory(user_text)
    system  = BASE_SYSTEM_PROMPT
    if context:
        system += "\n\nPersonal knowledge about the user:\n\n" + context

    update_history("user", user_text)
    messages = [{"role": "system", "content": system}] + chat_history

    try:
        resp  = ollama.chat(model=OLLAMA_MODEL, messages=messages)
        reply = resp["message"]["content"].strip()
    except Exception as e:
        reply = f"Neural core error: {e}"

    update_history("assistant", reply)
    return reply

# ══════════════════════════════════════════════════════════════════════════════
# COMMAND ROUTER  (same logic as jarvis.py)
# ══════════════════════════════════════════════════════════════════════════════

def execute_command(command: str) -> str:
    """
    Routes a text command and returns a response string.
    Also triggers TTS on the PC asynchronously.
    """
    command = command.strip().lower()

    # OS controls
    if any(kw in command for kw in ["shutdown", "power off"]):
        reply = "Initiating shutdown sequence."
        speak_async(reply)
        time.sleep(2)
        os.system("shutdown /s /t 5" if sys.platform.startswith("win") else "shutdown -h +1")
        return reply

    if any(kw in command for kw in ["restart", "reboot"]):
        reply = "Restarting the system."
        speak_async(reply)
        time.sleep(2)
        os.system("shutdown /r /t 5" if sys.platform.startswith("win") else "shutdown -r +1")
        return reply

    if "hibernate" in command:
        reply = "Entering hibernation mode."
        speak_async(reply)
        os.system("shutdown /h" if sys.platform.startswith("win") else "systemctl hibernate")
        return reply

    # World Monitor
    if any(kw in command for kw in ["world news", "global news", "happening around the world", "news update"]):
        return world_monitor_web()

    # YouTube
    if "youtube" in command:
        webbrowser.open("https://www.youtube.com")
        reply = "Opening YouTube on the PC."
        speak_async(reply)
        return reply

    # Wikipedia
    if any(kw in command for kw in ["wikipedia", "who is", "what is", "tell me about", "search"]):
        query = command
        for filler in ["wikipedia", "search for", "search", "tell me about", "who is", "what is", "jarvis"]:
            query = query.replace(filler, "")
        query = query.strip()
        if not query:
            return "What would you like me to search for?"
        return wiki_search(query)

    # Memory save
    if any(kw in command for kw in ["remember this", "save to memory", "note this"]):
        note = command
        for t in ["remember this", "save to memory", "note this", "jarvis"]:
            note = note.replace(t, "")
        note = note.strip(": ").strip()
        if note and MEMORY_AVAILABLE:
            memory_collection.add(documents=[note], ids=[f"mem_{int(time.time())}"])
            reply = "Saved to memory."
            speak_async(reply)
            return reply
        return "Nothing to save, or memory bank is unavailable."

    # Exit
    if any(kw in command for kw in ["goodbye", "exit", "quit"]):
        reply = "Goodbye. Powering down the web interface."
        speak_async(reply)
        return reply

    # Ollama fallback
    reply = ask_ollama(command)
    speak_async(reply)
    return reply


def world_monitor_web() -> str:
    speak_async("Give me a second, let me pull up the global feeds.")
    output_lines = []

    if NEWSAPI_KEY == "YOUR_NEWSAPI_KEY":
        output_lines.append("NewsAPI key not configured. Opening live map.")
    else:
        try:
            url  = f"https://newsapi.org/v2/top-headlines?language=en&pageSize=2&apiKey={NEWSAPI_KEY}"
            resp = http_requests.get(url, timeout=10)
            resp.raise_for_status()
            articles = resp.json().get("articles", [])
            if articles:
                output_lines.append("Top global headlines:")
                for i, a in enumerate(articles[:2], 1):
                    headline = a.get("title", "No title").split(" - ")[0].strip()
                    output_lines.append(f"{i}. {headline}")
                    speak_async(f"Headline {i}: {headline}")
            else:
                output_lines.append("No headlines available.")
        except Exception as e:
            output_lines.append(f"News feed error: {e}")

    webbrowser.open("https://liveuamap.com/")
    output_lines.append("Opening live situational-awareness map on PC.")
    return "\n".join(output_lines)


def wiki_search(query: str) -> str:
    try:
        summary = wikipedia.summary(query, sentences=2, auto_suggest=True)
        speak_async(summary)
        return summary
    except wikipedia.exceptions.DisambiguationError as e:
        reply = f"Multiple results found. Did you mean: {e.options[0]}?"
        speak_async(reply)
        return reply
    except wikipedia.exceptions.PageError:
        return "No Wikipedia page found for that topic."
    except Exception as e:
        return f"Wikipedia error: {e}"


# ══════════════════════════════════════════════════════════════════════════════
# FLASK APP
# ══════════════════════════════════════════════════════════════════════════════

app = Flask(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# HTML — Mobile-first chat UI
# ─────────────────────────────────────────────────────────────────────────────
HTML_PAGE = """
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8"/>
  <meta name="viewport" content="width=device-width, initial-scale=1.0, maximum-scale=1.0"/>
  <title>J.A.R.V.I.S.</title>
  <link href="https://fonts.googleapis.com/css2?family=Rajdhani:wght@400;600;700&family=Inter:wght@300;400;500&display=swap" rel="stylesheet"/>
  <style>
    :root {
      --bg:        #05080f;
      --panel:     #0a0f1e;
      --surface:   #0d1428;
      --border:    #1a2a4a;
      --accent:    #00bfff;
      --accent2:   #0066ff;
      --glow:      rgba(0,191,255,0.18);
      --user-bg:   #0d2240;
      --bot-bg:    #07111f;
      --text:      #c8ddf5;
      --muted:     #4a6080;
      --danger:    #ff4466;
    }

    * { box-sizing: border-box; margin: 0; padding: 0; }

    body {
      background: var(--bg);
      color: var(--text);
      font-family: 'Inter', sans-serif;
      height: 100dvh;
      display: flex;
      flex-direction: column;
      overflow: hidden;
    }

    /* ── Header ── */
    header {
      background: linear-gradient(135deg, #060c1a 0%, #0a1530 100%);
      border-bottom: 1px solid var(--border);
      padding: 14px 20px;
      display: flex;
      align-items: center;
      gap: 14px;
      flex-shrink: 0;
      box-shadow: 0 2px 20px rgba(0,0,0,0.5);
    }

    .logo-ring {
      width: 44px; height: 44px;
      border-radius: 50%;
      border: 2px solid var(--accent);
      box-shadow: 0 0 14px var(--glow), inset 0 0 14px rgba(0,191,255,0.06);
      display: flex; align-items: center; justify-content: center;
      position: relative;
      animation: pulse-ring 3s ease-in-out infinite;
    }
    .logo-ring::after {
      content: '';
      width: 8px; height: 8px;
      border-radius: 50%;
      background: var(--accent);
      box-shadow: 0 0 8px var(--accent);
    }

    @keyframes pulse-ring {
      0%,100% { box-shadow: 0 0 14px var(--glow), inset 0 0 14px rgba(0,191,255,0.06); }
      50%      { box-shadow: 0 0 28px rgba(0,191,255,0.35), inset 0 0 20px rgba(0,191,255,0.1); }
    }

    .header-text h1 {
      font-family: 'Rajdhani', sans-serif;
      font-size: 1.25rem;
      font-weight: 700;
      letter-spacing: 3px;
      color: var(--accent);
      text-shadow: 0 0 12px rgba(0,191,255,0.5);
    }
    .header-text p {
      font-size: 0.68rem;
      color: var(--muted);
      letter-spacing: 1px;
      margin-top: 1px;
    }

    .status-dot {
      margin-left: auto;
      width: 10px; height: 10px;
      border-radius: 50%;
      background: #00ff88;
      box-shadow: 0 0 8px #00ff88;
      animation: blink 2s infinite;
    }
    .status-dot.thinking { background: var(--accent); box-shadow: 0 0 8px var(--accent); }

    @keyframes blink {
      0%,100% { opacity: 1; } 50% { opacity: 0.3; }
    }

    /* ── Chat area ── */
    #chat {
      flex: 1;
      overflow-y: auto;
      padding: 18px 16px;
      display: flex;
      flex-direction: column;
      gap: 14px;
      scroll-behavior: smooth;
    }

    /* Scrollbar */
    #chat::-webkit-scrollbar { width: 4px; }
    #chat::-webkit-scrollbar-track { background: transparent; }
    #chat::-webkit-scrollbar-thumb { background: var(--border); border-radius: 4px; }

    .msg {
      max-width: 88%;
      padding: 12px 16px;
      border-radius: 16px;
      font-size: 0.92rem;
      line-height: 1.55;
      animation: fadeUp 0.25s ease;
      white-space: pre-wrap;
    }

    @keyframes fadeUp {
      from { opacity: 0; transform: translateY(10px); }
      to   { opacity: 1; transform: translateY(0); }
    }

    .msg.user {
      align-self: flex-end;
      background: linear-gradient(135deg, #0d3060, #0a1f45);
      border: 1px solid #1a3a70;
      color: #d0e8ff;
      border-bottom-right-radius: 4px;
    }

    .msg.jarvis {
      align-self: flex-start;
      background: var(--bot-bg);
      border: 1px solid var(--border);
      color: var(--text);
      border-bottom-left-radius: 4px;
      position: relative;
    }
    .msg.jarvis::before {
      content: 'J.A.R.V.I.S.';
      display: block;
      font-family: 'Rajdhani', sans-serif;
      font-size: 0.65rem;
      letter-spacing: 2px;
      color: var(--accent);
      margin-bottom: 6px;
      opacity: 0.8;
    }

    .msg.system {
      align-self: center;
      background: transparent;
      border: 1px solid var(--border);
      color: var(--muted);
      font-size: 0.75rem;
      text-align: center;
      border-radius: 20px;
      padding: 6px 14px;
      letter-spacing: 0.5px;
    }

    /* Typing indicator */
    .typing {
      align-self: flex-start;
      display: flex;
      align-items: center;
      gap: 5px;
      padding: 14px 18px;
      background: var(--bot-bg);
      border: 1px solid var(--border);
      border-radius: 16px;
      border-bottom-left-radius: 4px;
      animation: fadeUp 0.2s ease;
    }
    .typing span {
      width: 7px; height: 7px;
      border-radius: 50%;
      background: var(--accent);
      animation: bounce 1.2s infinite;
    }
    .typing span:nth-child(2) { animation-delay: 0.2s; }
    .typing span:nth-child(3) { animation-delay: 0.4s; }

    @keyframes bounce {
      0%,60%,100% { transform: translateY(0); opacity: 0.5; }
      30%          { transform: translateY(-6px); opacity: 1; }
    }

    /* ── Quick commands ── */
    .quick-btns {
      padding: 8px 16px;
      display: flex;
      gap: 8px;
      overflow-x: auto;
      flex-shrink: 0;
      scrollbar-width: none;
    }
    .quick-btns::-webkit-scrollbar { display: none; }

    .qbtn {
      flex-shrink: 0;
      padding: 7px 14px;
      border-radius: 20px;
      border: 1px solid var(--border);
      background: var(--surface);
      color: var(--muted);
      font-size: 0.75rem;
      cursor: pointer;
      transition: all 0.2s;
      white-space: nowrap;
      font-family: 'Inter', sans-serif;
    }
    .qbtn:hover, .qbtn:active {
      border-color: var(--accent);
      color: var(--accent);
      background: rgba(0,191,255,0.08);
    }

    /* ── Input bar ── */
    .input-bar {
      padding: 12px 16px 20px;
      background: var(--panel);
      border-top: 1px solid var(--border);
      display: flex;
      gap: 10px;
      align-items: center;
      flex-shrink: 0;
    }

    #cmd {
      flex: 1;
      background: var(--surface);
      border: 1px solid var(--border);
      border-radius: 24px;
      padding: 12px 18px;
      color: var(--text);
      font-size: 0.95rem;
      font-family: 'Inter', sans-serif;
      outline: none;
      transition: border-color 0.2s, box-shadow 0.2s;
    }
    #cmd:focus {
      border-color: var(--accent);
      box-shadow: 0 0 0 3px rgba(0,191,255,0.1);
    }
    #cmd::placeholder { color: var(--muted); }

    #send-btn, #mic-btn {
      width: 46px; height: 46px;
      border-radius: 50%;
      border: none;
      cursor: pointer;
      display: flex; align-items: center; justify-content: center;
      font-size: 1.1rem;
      transition: all 0.2s;
      flex-shrink: 0;
    }

    #send-btn {
      background: linear-gradient(135deg, var(--accent2), var(--accent));
      color: #fff;
      box-shadow: 0 2px 12px rgba(0,191,255,0.3);
    }
    #send-btn:active { transform: scale(0.93); }

    #mic-btn {
      background: var(--surface);
      border: 1px solid var(--border);
      color: var(--muted);
    }
    #mic-btn.active {
      background: rgba(255,68,102,0.12);
      border-color: var(--danger);
      color: var(--danger);
      box-shadow: 0 0 12px rgba(255,68,102,0.25);
      animation: mic-pulse 1s infinite;
    }

    @keyframes mic-pulse {
      0%,100% { box-shadow: 0 0 12px rgba(255,68,102,0.25); }
      50%      { box-shadow: 0 0 22px rgba(255,68,102,0.5); }
    }
  </style>
</head>
<body>

<header>
  <div class="logo-ring" id="logo"></div>
  <div class="header-text">
    <h1>J.A.R.V.I.S.</h1>
    <p>JUST A RATHER VERY INTELLIGENT SYSTEM</p>
  </div>
  <div class="status-dot" id="statusDot" title="Online"></div>
</header>

<div id="chat">
  <div class="msg system">Connected to your PC &bull; Commands spoken aloud through speakers</div>
</div>

<!-- Quick command pills -->
<div class="quick-btns">
  <button class="qbtn" onclick="sendCmd('world news')">🌍 World News</button>
  <button class="qbtn" onclick="sendCmd('open youtube')">▶ YouTube</button>
  <button class="qbtn" onclick="sendCmd('what is artificial intelligence')">🔍 Wikipedia</button>
  <button class="qbtn" onclick="sendCmd('tell me a joke')">😄 Joke</button>
  <button class="qbtn" onclick="sendCmd('what time is it')">🕐 Time</button>
  <button class="qbtn" onclick="sendCmd('motivate me')">⚡ Motivate</button>
</div>

<div class="input-bar">
  <button id="mic-btn" title="Voice input">🎙️</button>
  <input id="cmd" type="text" placeholder="Command J.A.R.V.I.S. ..." autocomplete="off" autocorrect="off"/>
  <button id="send-btn" onclick="sendFromInput()" title="Send">&#10148;</button>
</div>

<script>
  const chat     = document.getElementById('chat');
  const input    = document.getElementById('cmd');
  const dot      = document.getElementById('statusDot');
  const micBtn   = document.getElementById('mic-btn');
  let   thinking = null;

  // ── helpers ────────────────────────────────────────────────────────────────
  function addMsg(role, text) {
    if (thinking) { thinking.remove(); thinking = null; }
    const d = document.createElement('div');
    d.className = 'msg ' + role;
    d.textContent = text;
    chat.appendChild(d);
    chat.scrollTop = chat.scrollHeight;
  }

  function showTyping() {
    thinking = document.createElement('div');
    thinking.className = 'typing';
    thinking.innerHTML = '<span></span><span></span><span></span>';
    chat.appendChild(thinking);
    chat.scrollTop = chat.scrollHeight;
  }

  function setThinking(on) {
    dot.className = 'status-dot' + (on ? ' thinking' : '');
  }

  // ── send command ───────────────────────────────────────────────────────────
  async function sendCmd(text) {
    if (!text.trim()) return;
    addMsg('user', text);
    showTyping();
    setThinking(true);

    try {
      const res  = await fetch('/command', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ command: text })
      });
      const data = await res.json();
      addMsg('jarvis', data.reply || 'No response.');
    } catch (e) {
      addMsg('jarvis', 'Connection error — is the PC server running?');
    }

    setThinking(false);
  }

  function sendFromInput() {
    const t = input.value.trim();
    if (!t) return;
    input.value = '';
    sendCmd(t);
  }

  input.addEventListener('keydown', e => { if (e.key === 'Enter') sendFromInput(); });

  // ── voice input (Web Speech API) ───────────────────────────────────────────
  let recognition = null;
  if ('webkitSpeechRecognition' in window || 'SpeechRecognition' in window) {
    const SR = window.SpeechRecognition || window.webkitSpeechRecognition;
    recognition = new SR();
    recognition.lang = 'en-US';
    recognition.interimResults = false;
    recognition.maxAlternatives = 1;

    recognition.onresult = e => {
      const t = e.results[0][0].transcript;
      micBtn.classList.remove('active');
      sendCmd(t);
    };
    recognition.onerror = () => micBtn.classList.remove('active');
    recognition.onend   = () => micBtn.classList.remove('active');
  } else {
    micBtn.title = 'Voice input not supported on this browser';
    micBtn.style.opacity = '0.4';
    micBtn.style.cursor  = 'not-allowed';
  }

  micBtn.addEventListener('click', () => {
    if (!recognition) return;
    micBtn.classList.toggle('active');
    if (micBtn.classList.contains('active')) {
      recognition.start();
    } else {
      recognition.stop();
    }
  });

  // ── greeting on load ───────────────────────────────────────────────────────
  window.addEventListener('load', async () => {
    const res  = await fetch('/greet');
    const data = await res.json();
    addMsg('jarvis', data.greeting);
  });
</script>
</body>
</html>
"""

# ─────────────────────────────────────────────────────────────────────────────
# FLASK ROUTES
# ─────────────────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    """Serve the main chat UI."""
    return render_template_string(HTML_PAGE)


@app.route("/greet")
def greet():
    """Return a time-aware greeting (same logic as wish_me)."""
    hour = datetime.datetime.now().hour
    if 5 <= hour < 12:
        msg = "Good morning, sir. Systems are fully operational."
    elif 12 <= hour < 17:
        msg = "Good afternoon. All systems nominal."
    elif 17 <= hour < 21:
        msg = "Good evening. Ready when you are."
    elif 21 <= hour < 24:
        msg = "Good night, sir. Working late again, I see."
    else:
        msg = ("It is past midnight. May I ask why you are still awake? "
               "Shall I reduce the display brightness and activate a focus-mode playlist?")
    speak_async(msg)
    return jsonify({"greeting": msg})


@app.route("/command", methods=["POST"])
def command():
    """Accept a JSON command and return JARVIS's response."""
    data = request.get_json(force=True)
    cmd  = data.get("command", "").strip()
    if not cmd:
        return jsonify({"reply": "I didn't receive a command."})
    reply = execute_command(cmd)
    return jsonify({"reply": reply})


@app.route("/status")
def status():
    """Health-check endpoint."""
    return jsonify({
        "status":   "online",
        "model":    OLLAMA_MODEL,
        "memory":   MEMORY_AVAILABLE,
        "docs":     memory_collection.count() if MEMORY_AVAILABLE else 0,
        "whatsapp": WHATSAPP_AVAILABLE,
    })


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def get_local_ip() -> str:
    """Detect the PC's local network IP address."""
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception:
        return "127.0.0.1"


if __name__ == "__main__":
    local_ip = get_local_ip()

    print("+----------------------------------------------------------+")
    print("|   J.A.R.V.I.S.  Web Interface                          |")
    print("+----------------------------------------------------------+")
    print(f"|   PC URL    :  http://localhost:{PORT}                   |")
    print(f"|   Phone URL :  http://{local_ip}:{PORT}             |")
    print("|   Open the Phone URL on any device on your Wi-Fi       |")
    print("+----------------------------------------------------------+")

    speak_async(
        f"Web interface is online. "
        f"Open http {local_ip} port {PORT} on your phone to connect."
    )

    # Run Flask — accessible on all network interfaces
    app.run(host="0.0.0.0", port=PORT, debug=False, use_reloader=False)
