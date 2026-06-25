"""
J.A.R.V.I.S. Cloud Edition
===========================
24/7 access from any device — laptop can be OFF.

Deploy to Render.com (free):
  1. Push this folder to GitHub
  2. Connect repo on render.com → New Web Service
  3. Add env var: GROQ_API_KEY = <your key from console.groq.com>
  4. Deploy — get your permanent HTTPS URL
  5. Open on phone from anywhere in the world

Local test:
  pip install flask groq chromadb requests wikipedia
  set GROQ_API_KEY=your_key_here
  python jarvis_cloud.py
"""

import os
import sys
import time
import datetime
import threading
import requests as http_requests
import wikipedia
import chromadb

from flask import Flask, request, jsonify, render_template_string
from groq import Groq

# ══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ══════════════════════════════════════════════════════════════════════════════

GROQ_API_KEY   = os.environ.get("GROQ_API_KEY", "")   # Set as env var on Render
GROQ_MODEL     = "llama3-70b-8192"                     # Free, fast, powerful
NEWSAPI_KEY    = os.environ.get("NEWSAPI_KEY", "")
MAX_HISTORY    = 5
PORT           = int(os.environ.get("PORT", 5000))     # Render sets PORT automatically

# ══════════════════════════════════════════════════════════════════════════════
# GROQ LLM CLIENT
# ══════════════════════════════════════════════════════════════════════════════

groq_client = Groq(api_key=GROQ_API_KEY) if GROQ_API_KEY else None

BASE_SYSTEM_PROMPT = (
    "You are J.A.R.V.I.S., an advanced, witty, and highly capable AI assistant. "
    "You speak in a calm, professional, slightly British tone with occasional dry humour, "
    "inspired by Tony Stark's Friday. "
    "Keep responses concise (1-3 sentences unless asked for more)."
)

# ══════════════════════════════════════════════════════════════════════════════
# CHROMADB  (EphemeralClient — works on cloud with no local disk)
# ══════════════════════════════════════════════════════════════════════════════

try:
    chroma_client     = chromadb.EphemeralClient()
    memory_collection = chroma_client.get_or_create_collection(
        name="personal_knowledge",
        metadata={"hnsw:space": "cosine"}
    )
    MEMORY_AVAILABLE = True
    print("[BOOT] ChromaDB (ephemeral) ready.")
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
    if len(chat_history) > MAX_HISTORY * 2:
        del chat_history[:len(chat_history) - MAX_HISTORY * 2]

def query_memory(text: str) -> str:
    if not MEMORY_AVAILABLE or not memory_collection or memory_collection.count() == 0:
        return ""
    try:
        res  = memory_collection.query(query_texts=[text], n_results=min(3, memory_collection.count()))
        docs = res.get("documents", [[]])[0]
        return "\n\n".join(f"[Memory {i+1}]: {d}" for i, d in enumerate(docs)) if docs else ""
    except Exception:
        return ""

def ask_groq(user_text: str) -> str:
    if not groq_client:
        return "GROQ_API_KEY is not configured. Please add it as an environment variable on Render."

    context = query_memory(user_text)
    system  = BASE_SYSTEM_PROMPT
    if context:
        system += "\n\nPersonal knowledge about the user:\n\n" + context

    update_history("user", user_text)
    messages = [{"role": "system", "content": system}] + chat_history

    try:
        resp  = groq_client.chat.completions.create(
            model=GROQ_MODEL,
            messages=messages,
            max_tokens=256,
            temperature=0.7,
        )
        reply = resp.choices[0].message.content.strip()
    except Exception as e:
        reply = f"Neural core error: {e}"

    update_history("assistant", reply)
    return reply

# ══════════════════════════════════════════════════════════════════════════════
# COMMAND ROUTER
# ══════════════════════════════════════════════════════════════════════════════

def execute_command(command: str) -> dict:
    """
    Returns a dict: { "reply": str, "action": str|None }
    'action' can be "open_url" — the frontend handles opening URLs on the phone.
    """
    command = command.strip().lower()

    # World Monitor
    if any(kw in command for kw in ["world news", "global news", "happening around the world", "news update"]):
        return world_monitor()

    # YouTube
    if "youtube" in command:
        return {"reply": "Opening YouTube now.", "action": "open_url", "url": "https://www.youtube.com"}

    # Wikipedia
    if any(kw in command for kw in ["wikipedia", "who is", "what is", "tell me about", "search"]):
        query = command
        for filler in ["wikipedia", "search for", "search", "tell me about", "who is", "what is", "jarvis"]:
            query = query.replace(filler, "")
        query = query.strip()
        if not query:
            return {"reply": "What would you like me to search for?", "action": None}
        return {"reply": wiki_search(query), "action": None}

    # Memory save
    if any(kw in command for kw in ["remember this", "save to memory", "note this"]):
        note = command
        for t in ["remember this", "save to memory", "note this", "jarvis"]:
            note = note.replace(t, "")
        note = note.strip(": ").strip()
        if note and MEMORY_AVAILABLE:
            memory_collection.add(documents=[note], ids=[f"mem_{int(time.time())}"])
            return {"reply": "Saved to memory.", "action": None}
        return {"reply": "Nothing to save, or memory bank is unavailable.", "action": None}

    # Time
    if "time" in command or "what time" in command:
        now   = datetime.datetime.now().strftime("%I:%M %p")
        reply = f"The current time is {now}."
        return {"reply": reply, "action": None}

    # Date
    if "date" in command or "today" in command:
        today = datetime.datetime.now().strftime("%A, %B %d %Y")
        return {"reply": f"Today is {today}.", "action": None}

    # Groq LLM fallback
    reply = ask_groq(command)
    return {"reply": reply, "action": None}


def world_monitor() -> dict:
    lines = []
    if not NEWSAPI_KEY:
        lines.append("NewsAPI key not configured. Add NEWSAPI_KEY environment variable on Render.")
    else:
        try:
            url  = f"https://newsapi.org/v2/top-headlines?language=en&pageSize=2&apiKey={NEWSAPI_KEY}"
            resp = http_requests.get(url, timeout=10)
            resp.raise_for_status()
            articles = resp.json().get("articles", [])
            if articles:
                lines.append("Top global headlines:")
                for i, a in enumerate(articles[:2], 1):
                    headline = a.get("title", "No title").split(" - ")[0].strip()
                    lines.append(f"{i}. {headline}")
            else:
                lines.append("No headlines available right now.")
        except Exception as e:
            lines.append(f"News feed error: {e}")
    lines.append("Opening the live situational-awareness map.")
    return {
        "reply":  "\n".join(lines),
        "action": "open_url",
        "url":    "https://liveuamap.com/"
    }


def wiki_search(query: str) -> str:
    try:
        return wikipedia.summary(query, sentences=2, auto_suggest=True)
    except wikipedia.exceptions.DisambiguationError as e:
        return f"Multiple results. Did you mean: {e.options[0]}?"
    except wikipedia.exceptions.PageError:
        return "No Wikipedia page found for that topic."
    except Exception as e:
        return f"Wikipedia error: {e}"


# ══════════════════════════════════════════════════════════════════════════════
# HTML — Mobile-first UI with browser TTS (phone speaks the reply aloud)
# ══════════════════════════════════════════════════════════════════════════════

HTML_PAGE = """
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8"/>
  <meta name="viewport" content="width=device-width, initial-scale=1.0, maximum-scale=1.0, user-scalable=no"/>
  <title>J.A.R.V.I.S.</title>
  <meta name="description" content="J.A.R.V.I.S. — Your personal AI assistant, available 24/7"/>
  <meta name="theme-color" content="#05080f"/>
  <link rel="manifest" href="/manifest.json"/>
  <link href="https://fonts.googleapis.com/css2?family=Rajdhani:wght@400;600;700&family=Inter:wght@300;400;500&display=swap" rel="stylesheet"/>
  <style>
    :root {
      --bg:      #05080f;
      --panel:   #0a0f1e;
      --surface: #0d1428;
      --border:  #1a2a4a;
      --accent:  #00bfff;
      --accent2: #0055dd;
      --glow:    rgba(0,191,255,0.18);
      --user-bg: #0d2240;
      --bot-bg:  #07111f;
      --text:    #c8ddf5;
      --muted:   #4a6080;
      --green:   #00ff88;
      --red:     #ff4466;
    }
    *{box-sizing:border-box;margin:0;padding:0;}
    html,body{height:100%;overflow:hidden;}
    body{
      background:var(--bg);color:var(--text);
      font-family:'Inter',sans-serif;
      display:flex;flex-direction:column;
      height:100dvh;
    }

    /* HEADER */
    header{
      background:linear-gradient(135deg,#060c1a,#0a1530);
      border-bottom:1px solid var(--border);
      padding:12px 18px;
      display:flex;align-items:center;gap:12px;
      flex-shrink:0;
      box-shadow:0 2px 24px rgba(0,0,0,0.6);
    }
    .arc-reactor{
      width:42px;height:42px;border-radius:50%;
      border:2px solid var(--accent);
      box-shadow:0 0 16px var(--glow),inset 0 0 16px rgba(0,191,255,0.07);
      display:flex;align-items:center;justify-content:center;
      animation:arc-pulse 3s ease-in-out infinite;
    }
    .arc-reactor::after{
      content:'';width:9px;height:9px;border-radius:50%;
      background:var(--accent);box-shadow:0 0 10px var(--accent);
    }
    @keyframes arc-pulse{
      0%,100%{box-shadow:0 0 16px var(--glow),inset 0 0 16px rgba(0,191,255,0.07);}
      50%    {box-shadow:0 0 30px rgba(0,191,255,0.4),inset 0 0 22px rgba(0,191,255,0.12);}
    }
    .header-text h1{
      font-family:'Rajdhani',sans-serif;font-size:1.2rem;font-weight:700;
      letter-spacing:3px;color:var(--accent);
      text-shadow:0 0 14px rgba(0,191,255,0.5);
    }
    .header-text p{font-size:0.65rem;color:var(--muted);letter-spacing:1px;margin-top:2px;}
    .header-right{margin-left:auto;display:flex;align-items:center;gap:10px;}
    .online-pill{
      display:flex;align-items:center;gap:5px;
      padding:4px 10px;border-radius:20px;
      border:1px solid rgba(0,255,136,0.3);
      background:rgba(0,255,136,0.06);
      font-size:0.65rem;color:var(--green);letter-spacing:1px;
    }
    .dot{width:6px;height:6px;border-radius:50%;background:var(--green);
         box-shadow:0 0 6px var(--green);animation:blink 2s infinite;}
    .dot.thinking{background:var(--accent);box-shadow:0 0 6px var(--accent);}
    @keyframes blink{0%,100%{opacity:1;}50%{opacity:0.2;}}

    /* CHAT */
    #chat{
      flex:1;overflow-y:auto;padding:16px;
      display:flex;flex-direction:column;gap:12px;
    }
    #chat::-webkit-scrollbar{width:3px;}
    #chat::-webkit-scrollbar-thumb{background:var(--border);border-radius:3px;}

    .msg{
      max-width:86%;padding:11px 15px;border-radius:16px;
      font-size:0.9rem;line-height:1.6;
      animation:fadeUp 0.22s ease;white-space:pre-wrap;
    }
    @keyframes fadeUp{from{opacity:0;transform:translateY(8px);}to{opacity:1;transform:translateY(0);}}

    .msg.user{
      align-self:flex-end;
      background:linear-gradient(135deg,#0d2f5e,#0a1e42);
      border:1px solid #1a3870;color:#cfe6ff;
      border-bottom-right-radius:4px;
    }
    .msg.jarvis{
      align-self:flex-start;
      background:var(--bot-bg);border:1px solid var(--border);
      border-bottom-left-radius:4px;
    }
    .msg.jarvis .tag{
      font-family:'Rajdhani',sans-serif;font-size:0.6rem;
      letter-spacing:2px;color:var(--accent);opacity:0.8;
      display:block;margin-bottom:5px;
    }
    .msg.sys{
      align-self:center;border:1px solid var(--border);
      background:transparent;color:var(--muted);
      font-size:0.72rem;border-radius:20px;
      padding:5px 14px;text-align:center;
    }
    .typing{
      align-self:flex-start;display:flex;
      align-items:center;gap:5px;
      padding:13px 16px;background:var(--bot-bg);
      border:1px solid var(--border);
      border-radius:16px;border-bottom-left-radius:4px;
      animation:fadeUp 0.2s ease;
    }
    .typing span{
      width:6px;height:6px;border-radius:50%;
      background:var(--accent);animation:bounce 1.2s infinite;
    }
    .typing span:nth-child(2){animation-delay:.2s;}
    .typing span:nth-child(3){animation-delay:.4s;}
    @keyframes bounce{
      0%,60%,100%{transform:translateY(0);opacity:.5;}
      30%{transform:translateY(-6px);opacity:1;}
    }

    /* QUICK BUTTONS */
    .quick{
      padding:6px 14px;display:flex;gap:7px;
      overflow-x:auto;flex-shrink:0;scrollbar-width:none;
    }
    .quick::-webkit-scrollbar{display:none;}
    .qbtn{
      flex-shrink:0;padding:7px 13px;border-radius:20px;
      border:1px solid var(--border);background:var(--surface);
      color:var(--muted);font-size:0.73rem;cursor:pointer;
      transition:all .2s;white-space:nowrap;
      font-family:'Inter',sans-serif;
    }
    .qbtn:active{border-color:var(--accent);color:var(--accent);
      background:rgba(0,191,255,0.08);}

    /* INPUT BAR */
    .bar{
      padding:10px 14px 18px;
      background:var(--panel);border-top:1px solid var(--border);
      display:flex;gap:8px;align-items:center;flex-shrink:0;
    }
    #cmd{
      flex:1;background:var(--surface);
      border:1px solid var(--border);border-radius:24px;
      padding:11px 16px;color:var(--text);font-size:0.9rem;
      font-family:'Inter',sans-serif;outline:none;
      transition:border-color .2s,box-shadow .2s;
    }
    #cmd:focus{border-color:var(--accent);box-shadow:0 0 0 3px rgba(0,191,255,0.1);}
    #cmd::placeholder{color:var(--muted);}

    .icon-btn{
      width:44px;height:44px;border-radius:50%;border:none;
      cursor:pointer;display:flex;align-items:center;
      justify-content:center;font-size:1.05rem;
      transition:all .2s;flex-shrink:0;
    }
    #send-btn{
      background:linear-gradient(135deg,var(--accent2),var(--accent));
      color:#fff;box-shadow:0 2px 14px rgba(0,191,255,0.3);
    }
    #send-btn:active{transform:scale(.92);}
    #mic-btn{
      background:var(--surface);border:1px solid var(--border);
      color:var(--muted);
    }
    #mic-btn.active{
      background:rgba(255,68,102,.12);border-color:var(--red);
      color:var(--red);animation:mic-pulse 1s infinite;
    }
    #vol-btn{
      background:var(--surface);border:1px solid var(--border);
      color:var(--muted);font-size:.9rem;
    }
    #vol-btn.muted{color:var(--red);border-color:var(--red);}
    @keyframes mic-pulse{
      0%,100%{box-shadow:0 0 10px rgba(255,68,102,.25);}
      50%{box-shadow:0 0 20px rgba(255,68,102,.5);}
    }
  </style>
</head>
<body>

<header>
  <div class="arc-reactor"></div>
  <div class="header-text">
    <h1>J.A.R.V.I.S.</h1>
    <p>CLOUD EDITION &bull; 24/7 ONLINE</p>
  </div>
  <div class="header-right">
    <div class="online-pill">
      <div class="dot" id="dot"></div>
      <span id="status-label">ONLINE</span>
    </div>
  </div>
</header>

<div id="chat">
  <div class="msg sys">24/7 Cloud Mode &bull; Laptop can be OFF &bull; Voice replies play on your phone</div>
</div>

<div class="quick">
  <button class="qbtn" onclick="sendCmd('world news')">🌍 World News</button>
  <button class="qbtn" onclick="sendCmd('open youtube')">▶ YouTube</button>
  <button class="qbtn" onclick="sendCmd('what is machine learning')">🔍 Wikipedia</button>
  <button class="qbtn" onclick="sendCmd('tell me a joke')">😄 Joke</button>
  <button class="qbtn" onclick="sendCmd('what time is it')">🕐 Time</button>
  <button class="qbtn" onclick="sendCmd('motivate me')">⚡ Motivate</button>
  <button class="qbtn" onclick="sendCmd('what is the weather like today')">🌤 Weather</button>
</div>

<div class="bar">
  <button class="icon-btn" id="mic-btn" title="Voice input">🎙️</button>
  <input id="cmd" type="text" placeholder="Command J.A.R.V.I.S. ..." autocomplete="off" autocorrect="off" autocapitalize="off"/>
  <button class="icon-btn" id="vol-btn" title="Toggle voice reply">🔊</button>
  <button class="icon-btn" id="send-btn" onclick="sendFromInput()">&#10148;</button>
</div>

<script>
  const chatEl   = document.getElementById('chat');
  const inputEl  = document.getElementById('cmd');
  const dot      = document.getElementById('dot');
  const statusLb = document.getElementById('status-label');
  const micBtn   = document.getElementById('mic-btn');
  const volBtn   = document.getElementById('vol-btn');

  let ttsEnabled = true;
  let typing     = null;
  let synth      = window.speechSynthesis;
  let voices     = [];

  // Load voices (needed for some browsers)
  function loadVoices() {
    voices = synth.getVoices();
  }
  if (synth.onvoiceschanged !== undefined) synth.onvoiceschanged = loadVoices;
  loadVoices();

  // ── speak on phone ───────────────────────────────────────────────────────
  function speakText(text) {
    if (!ttsEnabled || !synth) return;
    synth.cancel();
    const utt = new SpeechSynthesisUtterance(text);
    utt.rate  = 1.0;
    utt.pitch = 0.9;
    // Prefer an English male voice
    const eng = voices.find(v => v.lang.startsWith('en') && v.name.toLowerCase().includes('male'))
             || voices.find(v => v.lang.startsWith('en'))
             || null;
    if (eng) utt.voice = eng;
    synth.speak(utt);
  }

  // Volume toggle
  volBtn.addEventListener('click', () => {
    ttsEnabled = !ttsEnabled;
    volBtn.textContent  = ttsEnabled ? '🔊' : '🔇';
    volBtn.title        = ttsEnabled ? 'Mute voice reply' : 'Enable voice reply';
    volBtn.classList.toggle('muted', !ttsEnabled);
    if (!ttsEnabled) synth.cancel();
  });

  // ── helpers ──────────────────────────────────────────────────────────────
  function addMsg(role, text) {
    if (typing) { typing.remove(); typing = null; }
    const d = document.createElement('div');
    d.className = 'msg ' + role;
    if (role === 'jarvis') {
      d.innerHTML = '<span class="tag">J.A.R.V.I.S.</span>' +
        text.replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;');
    } else {
      d.textContent = text;
    }
    chatEl.appendChild(d);
    chatEl.scrollTop = chatEl.scrollHeight;
    return d;
  }

  function showTyping() {
    typing = document.createElement('div');
    typing.className = 'typing';
    typing.innerHTML = '<span></span><span></span><span></span>';
    chatEl.appendChild(typing);
    chatEl.scrollTop = chatEl.scrollHeight;
  }

  function setThinking(on) {
    dot.className      = on ? 'dot thinking' : 'dot';
    statusLb.textContent = on ? 'THINKING' : 'ONLINE';
  }

  // ── send command ─────────────────────────────────────────────────────────
  async function sendCmd(text) {
    if (!text.trim()) return;
    addMsg('user', text);
    showTyping();
    setThinking(true);
    try {
      const res  = await fetch('/command', {
        method: 'POST',
        headers: {'Content-Type':'application/json'},
        body: JSON.stringify({command: text})
      });
      const data = await res.json();
      addMsg('jarvis', data.reply || 'No response.');
      speakText(data.reply || '');
      // Handle actions (e.g. open a URL on the phone)
      if (data.action === 'open_url' && data.url) {
        setTimeout(() => window.open(data.url, '_blank'), 800);
      }
    } catch(e) {
      addMsg('jarvis', 'Connection error. Is the server running?');
    }
    setThinking(false);
  }

  function sendFromInput() {
    const t = inputEl.value.trim();
    if (!t) return;
    inputEl.value = '';
    sendCmd(t);
  }

  inputEl.addEventListener('keydown', e => { if(e.key==='Enter') sendFromInput(); });

  // ── microphone (Web Speech API) ──────────────────────────────────────────
  let recognition = null;
  const SR = window.SpeechRecognition || window.webkitSpeechRecognition;
  if (SR) {
    recognition = new SR();
    recognition.lang           = 'en-US';
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
    micBtn.style.opacity = '.4';
    micBtn.style.cursor  = 'not-allowed';
    micBtn.title = 'Voice not supported — use Chrome or Safari';
  }

  micBtn.addEventListener('click', () => {
    if (!recognition) return;
    if (micBtn.classList.contains('active')) {
      recognition.stop();
      micBtn.classList.remove('active');
    } else {
      recognition.start();
      micBtn.classList.add('active');
    }
  });

  // ── greeting on load ─────────────────────────────────────────────────────
  (async () => {
    try {
      const res  = await fetch('/greet');
      const data = await res.json();
      addMsg('jarvis', data.greeting);
      speakText(data.greeting);
    } catch(e) {
      addMsg('jarvis', 'J.A.R.V.I.S. is online. How can I assist?');
    }
  })();
</script>
</body>
</html>
"""

# ══════════════════════════════════════════════════════════════════════════════
# FLASK APP
# ══════════════════════════════════════════════════════════════════════════════

app = Flask(__name__)

@app.route("/")
def index():
    return render_template_string(HTML_PAGE)

@app.route("/greet")
def greet():
    hour = datetime.datetime.now().hour
    if 5 <= hour < 12:
        msg = "Good morning, sir. Systems are fully operational and online."
    elif 12 <= hour < 17:
        msg = "Good afternoon. All systems nominal."
    elif 17 <= hour < 21:
        msg = "Good evening. Ready when you are."
    elif 21 <= hour < 24:
        msg = "Good night, sir. Working late again, I see."
    else:
        msg = "It is past midnight. May I ask why you are still awake?"
    return jsonify({"greeting": msg})

@app.route("/command", methods=["POST"])
def command():
    data = request.get_json(force=True)
    cmd  = data.get("command", "").strip()
    if not cmd:
        return jsonify({"reply": "I didn't receive a command.", "action": None})
    result = execute_command(cmd)
    return jsonify(result)

@app.route("/status")
def status():
    return jsonify({
        "status":  "online",
        "model":   GROQ_MODEL,
        "memory":  MEMORY_AVAILABLE,
        "docs":    memory_collection.count() if MEMORY_AVAILABLE else 0,
        "groq_ok": bool(groq_client),
    })

# PWA manifest (lets phone add JARVIS to home screen like an app)
@app.route("/manifest.json")
def manifest():
    return jsonify({
        "name":             "J.A.R.V.I.S.",
        "short_name":       "JARVIS",
        "start_url":        "/",
        "display":          "standalone",
        "background_color": "#05080f",
        "theme_color":      "#00bfff",
        "description":      "Your personal AI assistant — 24/7 cloud edition",
        "icons": [{"src": "/favicon.ico", "sizes": "any", "type": "image/x-icon"}]
    })

if __name__ == "__main__":
    print("+----------------------------------------------------+")
    print("|  J.A.R.V.I.S. Cloud Edition                       |")
    print("+----------------------------------------------------+")
    if not GROQ_API_KEY:
        print("|  WARNING: GROQ_API_KEY not set!                   |")
        print("|  Get free key: https://console.groq.com           |")
    else:
        print("|  Groq API     : Connected                         |")
    print(f"|  Running on   : http://0.0.0.0:{PORT}               |")
    print("+----------------------------------------------------+")
    app.run(host="0.0.0.0", port=PORT, debug=False, use_reloader=False)
