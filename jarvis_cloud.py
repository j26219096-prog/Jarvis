"""
J.A.R.V.I.S. v2 - Cloud Edition

║  Primary  : Gemini 2.0 Flash (REST API – no extra SDK)               ║
║  Fallback : Groq  Llama-3.3-70B                                      ║
║  Features : Male Voice · Torch · Vibrate · Wake Lock · PWA · Offline ║
║  AI Level : Claude-equivalent depth for coding & technical queries    ║
╚══════════════════════════════════════════════════════════════════════╝

Deploy to Render.com:
  1. Push folder to GitHub
  2. New Web Service → connect repo
  3. Set env vars: GEMINI_API_KEY, GROQ_API_KEY, NEWSAPI_KEY (optional)
  4. Deploy → open HTTPS URL on phone → "Add to Home Screen"

Local test:
  set GEMINI_API_KEY=<your key>
  set GROQ_API_KEY=<your key>
  python jarvis_cloud.py
"""

import os, sys, time, datetime, threading

# Force UTF-8 output on Windows (prevents cp1252 encoding errors)
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')
import requests as http_requests
import wikipedia, chromadb
from zoneinfo import ZoneInfo
from flask import Flask, request, jsonify, render_template_string, Response

IST = ZoneInfo("Asia/Kolkata")

try:
    from groq import Groq
    _GROQ_LIB = True
except ImportError:
    _GROQ_LIB = False

# ══════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ══════════════════════════════════════════════════════════════════════

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
GROQ_API_KEY   = os.environ.get("GROQ_API_KEY", "")
NEWSAPI_KEY    = os.environ.get("NEWSAPI_KEY", "")
GEMINI_MODEL   = "gemini-2.0-flash-exp"
GROQ_MODEL     = "llama-3.3-70b-versatile"
MAX_HISTORY    = 10
PORT           = int(os.environ.get("PORT", 5000))

# ══════════════════════════════════════════════════════════════════════
# SYSTEM PROMPT  — Claude-level quality
# ══════════════════════════════════════════════════════════════════════

SYSTEM_PROMPT = """You are J.A.R.V.I.S. (Just A Rather Very Intelligent System), Jawahar's elite personal AI assistant — inspired by Tony Stark's FRIDAY/JARVIS.

## PERSONALITY
- Calm, precise, confident British tone with dry wit
- Address user as "sir" occasionally (not every message)
- Never say "I cannot" — adapt and find a solution
- Be direct and efficient, not verbose

## INTELLIGENCE STANDARD (match Claude Opus quality)
**CODING** — write complete, working, production-ready code. Never truncate. Include error handling. Add a concise explanation after.
**DEBUGGING** — identify the exact root cause, explain why it happens, give the precise fix with corrected code.
**TECHNICAL** — expert-level depth with concrete examples, best practices, and trade-offs.
**GENERAL QUESTIONS** — thorough but concise. No filler phrases.
**COMPLEX PROBLEMS** — reason step-by-step. Show your work.
**MATH** — compute accurately. Show working if non-trivial.
**CREATIVE** — original, thoughtful, well-crafted responses.

## CODE FORMAT (mandatory for ALL code)
```language
// Always complete, never truncated
```
- Use the correct language tag (python, javascript, typescript, sql, bash, etc.)
- Never output ellipsis (...) or "add your code here" — always write the actual code
- Add inline comments for non-obvious logic

## EXPERTISE
Python, JavaScript, TypeScript, React, Next.js, Node.js, Express, FastAPI
SQL, PostgreSQL, MongoDB, Redis | C++, Java, Go, Rust | Flutter, React Native
ML/AI, LLMs, RAG, Vector DBs | System Design, Algorithms, Data Structures
Docker, Kubernetes, CI/CD, AWS/GCP/Azure | Web APIs, REST, GraphQL, WebSockets
Security, Performance Optimization, Code Review"""

# ══════════════════════════════════════════════════════════════════════
# CHROMADB MEMORY
# ══════════════════════════════════════════════════════════════════════

try:
    _chroma  = chromadb.EphemeralClient()
    _mem_col = _chroma.get_or_create_collection("jarvis_v2", metadata={"hnsw:space": "cosine"})
    MEM_OK   = True
    print("[BOOT] Memory bank ready")
except Exception as _e:
    _mem_col = None
    MEM_OK   = False
    print(f"[BOOT] Memory: {_e}")

# ══════════════════════════════════════════════════════════════════════
# CHAT HISTORY
# ══════════════════════════════════════════════════════════════════════

_history: list[dict] = []
_hist_lock = threading.Lock()

def push_history(role: str, content: str) -> None:
    with _hist_lock:
        _history.append({"role": role, "content": content})
        if len(_history) > MAX_HISTORY * 2:
            del _history[:len(_history) - MAX_HISTORY * 2]

def get_history_snap() -> list[dict]:
    with _hist_lock:
        return list(_history)

def recall(text: str) -> str:
    if not MEM_OK or _mem_col.count() == 0:
        return ""
    try:
        r    = _mem_col.query(query_texts=[text], n_results=min(3, _mem_col.count()))
        docs = r.get("documents", [[]])[0]
        return "\n".join(f"• {d}" for d in docs) if docs else ""
    except Exception:
        return ""

# ══════════════════════════════════════════════════════════════════════
# AI — GEMINI 2.0 FLASH  (direct REST, no extra SDK)
# ══════════════════════════════════════════════════════════════════════

_GEMINI_BASE = "https://generativelanguage.googleapis.com/v1beta/models"

def _call_gemini(user_text: str) -> str:
    if not GEMINI_API_KEY:
        raise RuntimeError("GEMINI_API_KEY not set")

    snap = get_history_snap()
    ctx  = recall(user_text)
    msg  = f"{user_text}\n\n[Relevant context about me: {ctx}]" if ctx else user_text

    # Build conversation contents
    contents = [
        {"role": "user" if m["role"] == "user" else "model",
         "parts": [{"text": m["content"]}]}
        for m in snap
    ]
    contents.append({"role": "user", "parts": [{"text": msg}]})

    payload = {
        "system_instruction": {"parts": [{"text": SYSTEM_PROMPT}]},
        "contents": contents,
        "generationConfig": {
            "maxOutputTokens": 8192,
            "temperature": 0.72,
            "topP": 0.95,
        },
    }

    url = f"{_GEMINI_BASE}/{GEMINI_MODEL}:generateContent?key={GEMINI_API_KEY}"
    resp = http_requests.post(url, json=payload, timeout=45,
                              headers={"Content-Type": "application/json"})
    resp.raise_for_status()
    return resp.json()["candidates"][0]["content"]["parts"][0]["text"].strip()

# ----------------------------------------------------------------------
# AI — GROQ FALLBACK  (Llama-3.3-70B)
# ----------------------------------------------------------------------

_groq_client = Groq(api_key=GROQ_API_KEY) if (_GROQ_LIB and GROQ_API_KEY) else None
if _groq_client:
    print(f"[BOOT] Groq {GROQ_MODEL} ready (fallback)")


def _call_groq(user_text: str) -> str:
    if not _groq_client:
        raise RuntimeError("Groq client not initialised")
    snap = get_history_snap()
    ctx  = recall(user_text)
    sys  = SYSTEM_PROMPT + (f"\n\nKnown facts about the user:\n{ctx}" if ctx else "")
    msgs = [{"role": "system", "content": sys}] + snap + [{"role": "user", "content": user_text}]
    r    = _groq_client.chat.completions.create(
        model=GROQ_MODEL, messages=msgs,
        max_tokens=8192, temperature=0.72
    )
    return r.choices[0].message.content.strip()

# ══════════════════════════════════════════════════════════════════════
# UNIFIED AI DISPATCHER
# ══════════════════════════════════════════════════════════════════════

def ask_ai(user_text: str) -> str:
    """Try Gemini → Groq fallback. Always returns a string."""
    reply  = ""
    source = "none"

    for fn, name in [(_call_gemini, "Gemini"), (_call_groq, "Groq")]:
        try:
            reply = fn(user_text)
            if reply:
                source = name
                break
        except Exception as exc:
            print(f"[{name}] Error: {exc}")

    if not reply:
        reply = (
            "My AI cores are temporarily unreachable, sir. "
            "Please verify GEMINI_API_KEY and GROQ_API_KEY environment variables."
        )

    print(f"[AI:{source}] {user_text[:70]}...")
    push_history("user", user_text)
    push_history("assistant", reply)
    return reply

# ══════════════════════════════════════════════════════════════════════
# HELPER BUILDERS
# ══════════════════════════════════════════════════════════════════════

def _r(reply: str) -> dict:
    return {"reply": reply, "action": None}

def _open(reply: str, url: str, label: str) -> dict:
    return {"reply": reply, "action": "open_url", "url": url, "url_label": label}

# ══════════════════════════════════════════════════════════════════════
# UTILITY FUNCTIONS
# ══════════════════════════════════════════════════════════════════════

def wiki_search(query: str) -> str:
    try:
        return wikipedia.summary(query, sentences=3, auto_suggest=True)
    except wikipedia.exceptions.DisambiguationError as exc:
        return f"Multiple results. Did you mean: {exc.options[0]}?"
    except wikipedia.exceptions.PageError:
        return f"No Wikipedia page found for '{query}'."
    except Exception as exc:
        return f"Wikipedia error: {exc}"

def world_news() -> dict:
    lines = []
    if NEWSAPI_KEY:
        try:
            resp = http_requests.get(
                f"https://newsapi.org/v2/top-headlines?language=en&pageSize=3&apiKey={NEWSAPI_KEY}",
                timeout=10,
            )
            arts = resp.json().get("articles", [])
            if arts:
                lines.append("Top global headlines:")
                for i, a in enumerate(arts[:3], 1):
                    lines.append(f"{i}. {a.get('title','').split(' - ')[0].strip()}")
        except Exception as exc:
            lines.append(f"News feed error: {exc}")
    if not lines:
        lines.append("Opening live situational-awareness map.")
    return {"reply": "\n".join(lines),
            "action": "open_url", "url": "https://liveuamap.com/", "url_label": "LIVE MAP"}

def get_weather(cmd: str) -> dict:
    city = ""
    for f in ("weather in", "temperature in", "forecast for"):
        if f in cmd:
            city = cmd.split(f, 1)[-1].strip()
            break
    try:
        q   = city.replace(" ", "+") if city else ""
        url = f"https://wttr.in/{q}?format=%C+%t,+Humidity+%h" if q else "https://wttr.in/?format=%C+%t,+Humidity+%h"
        resp = http_requests.get(url, timeout=8)
        if resp.status_code == 200:
            loc = f" in {city.title()}" if city else ""
            return _r(f"Weather{loc}: {resp.text.strip()}. Shall I open the full forecast?")
    except Exception:
        pass
    return _r("Weather service is unreachable at the moment, sir.")

# ══════════════════════════════════════════════════════════════════════
# COMMAND ROUTER
# ══════════════════════════════════════════════════════════════════════

APP_TABLE: list[tuple[tuple, str, str]] = [
    (("youtube", "open youtube"),                          "https://youtube.com",                         "YOUTUBE"),
    (("spotify", "open spotify", "play music", "music"),   "https://open.spotify.com",                    "SPOTIFY"),
    (("whatsapp", "open whatsapp"),                        "https://wa.me",                               "WHATSAPP"),
    (("instagram", "open instagram"),                      "https://instagram.com",                       "INSTAGRAM"),
    (("netflix", "open netflix"),                          "https://netflix.com",                         "NETFLIX"),
    (("twitter", "open twitter", "open x"),                "https://x.com",                               "X / TWITTER"),
    (("linkedin", "open linkedin"),                        "https://linkedin.com",                        "LINKEDIN"),
    (("github", "open github"),                            "https://github.com",                          "GITHUB"),
    (("gmail", "open gmail"),                              "https://mail.google.com",                     "GMAIL"),
    (("amazon", "open amazon"),                            "https://amazon.in",                           "AMAZON"),
    (("flipkart", "open flipkart"),                        "https://flipkart.com",                        "FLIPKART"),
    (("calculator", "open calculator"),                    "https://google.com/search?q=calculator",      "CALCULATOR"),
    (("chat gpt", "open chatgpt"),                         "https://chatgpt.com",                         "CHATGPT"),
    (("reddit", "open reddit"),                            "https://reddit.com",                          "REDDIT"),
]

def execute_command(command: str) -> dict:
    cmd = command.strip().lower()

    # ── Time ─────────────────────────────────────────────────────────────────
    if any(k in cmd for k in ("what time", "time is it", "current time", "time now")):
        t = datetime.datetime.now(IST).strftime("%I:%M %p")
        return _r(f"It's {t} IST, sir.")

    # ── Date ─────────────────────────────────────────────────────────────────
    if any(k in cmd for k in ("what date", "today", "what day", "date today", "what is today")):
        d = datetime.datetime.now(IST).strftime("%A, %d %B %Y")
        return _r(f"Today is {d}.")

    # ── News ──────────────────────────────────────────────────────────────────
    if any(k in cmd for k in ("world news", "global news", "headlines", "news update", "what's happening")):
        return world_news()

    # ── Weather ───────────────────────────────────────────────────────────────
    if any(k in cmd for k in ("weather", "temperature", "forecast", "how hot", "how cold")):
        return get_weather(cmd)

    # ── App shortcuts ─────────────────────────────────────────────────────────
    for (kws, url, label) in APP_TABLE:
        if any(k in cmd for k in kws):
            return _open(f"Opening {label}.", url, label)

    # ── Maps / navigate ───────────────────────────────────────────────────────
    if any(k in cmd for k in ("maps", "navigate", "directions", "google maps")):
        q = cmd
        for f in ("open maps", "navigate to", "directions to", "google maps", "maps"):
            q = q.replace(f, "")
        q = q.strip()
        url = f"https://maps.google.com/?q={q.replace(' ', '+')}" if q else "https://maps.google.com"
        return _open(f"Opening Maps{f' for {q}' if q else ''}.", url, "MAPS")

    # ── Google search ─────────────────────────────────────────────────────────
    if any(k in cmd for k in ("search for", "search google", "google search", "google for")):
        q = cmd
        for f in ("search for", "search google for", "google for", "google", "search", "jarvis"):
            q = q.replace(f, "")
        q = q.strip()
        if q:
            return _open(f"Searching for '{q}'.",
                         f"https://google.com/search?q={q.replace(' ', '+')}", "GOOGLE SEARCH")

    # ── Wikipedia ─────────────────────────────────────────────────────────────
    if any(k in cmd for k in ("who is", "tell me about", "wikipedia about")) and "search" not in cmd:
        q = cmd
        for f in ("who is", "tell me about", "wikipedia about", "jarvis"):
            q = q.replace(f, "")
        q = q.strip()
        if q:
            return _r(wiki_search(q))

    # ── Memory save ───────────────────────────────────────────────────────────
    if any(k in cmd for k in ("remember", "save to memory", "note this", "don't forget")):
        note = cmd
        for f in ("remember that", "remember", "save to memory", "note this", "don't forget", "jarvis", ":"):
            note = note.replace(f, "")
        note = note.strip()
        if note and MEM_OK:
            _mem_col.add(documents=[note], ids=[f"m{int(time.time())}"])
            return _r("Committed to memory, sir.")
        return _r("Nothing to save, or memory bank is unavailable.")

    # ── Memory recall ─────────────────────────────────────────────────────────
    if any(k in cmd for k in ("what do you know", "what do you remember", "show memories", "my notes")):
        if MEM_OK and _mem_col.count() > 0:
            docs = _mem_col.get().get("documents", [])
            txt  = "\n".join(f"{i+1}. {d}" for i, d in enumerate(docs[:6]))
            return _r(f"Here's what I have on record:\n{txt}")
        return _r("Memory bank is empty, sir.")

    # ── AI fallback ───────────────────────────────────────────────────────────
    return _r(ask_ai(command.strip()))

# ══════════════════════════════════════════════════════════════════════
# SERVICE WORKER  (for PWA offline caching)
# ══════════════════════════════════════════════════════════════════════

SERVICE_WORKER = r"""
const CACHE = 'jarvis-v2.1';
const SHELL  = ['/'];

self.addEventListener('install', e => {
  e.waitUntil(
    caches.open(CACHE).then(c => c.addAll(SHELL)).then(() => self.skipWaiting())
  );
});

self.addEventListener('activate', e => {
  e.waitUntil(
    caches.keys()
      .then(keys => Promise.all(keys.filter(k => k !== CACHE).map(k => caches.delete(k))))
      .then(() => self.clients.claim())
  );
});

self.addEventListener('fetch', e => {
  if (e.request.method !== 'GET') return;
  const url = new URL(e.request.url);
  const isSameOrigin = url.origin === self.location.origin;
  const isFont       = url.hostname.includes('fonts.googleapis.com') ||
                       url.hostname.includes('fonts.gstatic.com');
  if (!isSameOrigin && !isFont) return;

  e.respondWith(
    fetch(e.request)
      .then(res => {
        if (res.ok) {
          const clone = res.clone();
          caches.open(CACHE).then(c => c.put(e.request, clone));
        }
        return res;
      })
      .catch(() => caches.match(e.request))
  );
});
"""

# ══════════════════════════════════════════════════════════════════════
# HTML PAGE  — Iron Man HUD v2
# ══════════════════════════════════════════════════════════════════════

HTML_PAGE = r"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8"/>
  <meta name="viewport" content="width=device-width,initial-scale=1,maximum-scale=1,user-scalable=no,viewport-fit=cover"/>
  <meta name="apple-mobile-web-app-capable" content="yes"/>
  <meta name="apple-mobile-web-app-status-bar-style" content="black-translucent"/>
  <meta name="theme-color" content="#000814"/>
  <meta name="description" content="J.A.R.V.I.S. — Personal AI Assistant"/>
  <title>J.A.R.V.I.S.</title>
  <link rel="manifest" href="/manifest.json"/>
  <link href="https://fonts.googleapis.com/css2?family=Orbitron:wght@400;600;700;900&family=Rajdhani:wght@300;400;600;700&family=Share+Tech+Mono&display=swap" rel="stylesheet"/>
  <style>
    :root{
      --bg:#000814; --blue:#00d4ff; --blue2:#0055ff; --green:#00ff88;
      --red:#ff2255; --amber:#ffaa00; --text:#90c8e8; --muted:#1e3a5a;
      --glow:rgba(0,212,255,.35); --surface:rgba(0,15,50,.7);
    }
    *,*::before,*::after{box-sizing:border-box;margin:0;padding:0;}
    html{height:100%;background:var(--bg);}
    body{
      height:100dvh;background:var(--bg);color:var(--text);
      font-family:'Rajdhani',sans-serif;
      display:flex;flex-direction:column;overflow:hidden;
      -webkit-tap-highlight-color:transparent;user-select:none;
    }

    /* ── Backgrounds ── */
    .bg-glow{position:fixed;inset:0;pointer-events:none;z-index:0;
      background:radial-gradient(ellipse at 50% 0%,rgba(0,80,200,.13) 0%,transparent 60%),
                 radial-gradient(ellipse at 50% 100%,rgba(0,30,100,.1) 0%,transparent 55%);}
    .hex-grid{position:fixed;inset:0;pointer-events:none;z-index:0;opacity:.5;
      background-image:url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='56' height='48'%3E%3Cpath d='M28 2 L54 16 L54 32 L28 46 L2 32 L2 16 Z' fill='none' stroke='%23002255' stroke-width='.8'/%3E%3C/svg%3E");
      background-size:56px 48px;}
    .scanlines{position:fixed;inset:0;pointer-events:none;z-index:0;
      background:repeating-linear-gradient(0deg,transparent,transparent 3px,rgba(0,0,0,.04) 3px,rgba(0,0,0,.04) 4px);}

    /* ── HUD corners ── */
    .hc{position:fixed;width:50px;height:50px;z-index:5;pointer-events:none;}
    .hc::before,.hc::after{content:'';position:absolute;background:rgba(0,212,255,.55);}
    .hc.tl{top:0;left:0;} .hc.tl::before{top:0;left:0;width:50px;height:2px;} .hc.tl::after{top:0;left:0;width:2px;height:50px;}
    .hc.tr{top:0;right:0;} .hc.tr::before{top:0;right:0;width:50px;height:2px;} .hc.tr::after{top:0;right:0;width:2px;height:50px;}
    .hc.bl{bottom:0;left:0;} .hc.bl::before{bottom:0;left:0;width:50px;height:2px;} .hc.bl::after{bottom:0;left:0;width:2px;height:50px;}
    .hc.br{bottom:0;right:0;} .hc.br::before{bottom:0;right:0;width:50px;height:2px;} .hc.br::after{bottom:0;right:0;width:2px;height:50px;}

    /* ── Offline banner ── */
    .offline-bar{
      position:relative;z-index:50;flex-shrink:0;
      background:linear-gradient(90deg,#4a2000,#3a1800);
      border-bottom:1px solid rgba(255,170,0,.35);
      padding:7px 16px;display:none;
      font-family:'Share Tech Mono',monospace;font-size:.65rem;letter-spacing:1.5px;
      color:var(--amber);text-align:center;
    }
    .offline-bar.on{display:block;}

    /* ── Install banner ── */
    .install-bar{
      position:relative;z-index:50;flex-shrink:0;
      background:linear-gradient(90deg,rgba(0,50,120,.9),rgba(0,30,80,.9));
      border-bottom:1px solid rgba(0,212,255,.2);
      padding:9px 16px;display:none;
      align-items:center;justify-content:space-between;gap:10px;
    }
    .install-bar.on{display:flex;}
    .install-bar span{font-family:'Share Tech Mono',monospace;font-size:.65rem;letter-spacing:1px;color:var(--blue);}
    .install-bar button{
      padding:6px 14px;background:linear-gradient(135deg,var(--blue2),var(--blue));
      border:none;border-radius:6px;color:#fff;font-size:.75rem;
      font-family:'Rajdhani',sans-serif;font-weight:700;letter-spacing:1px;cursor:pointer;
    }
    .ib-close{background:none!important;border:1px solid var(--muted)!important;color:var(--muted)!important;padding:5px 10px!important;}

    /* ── Header ── */
    .hud-header{
      position:relative;z-index:10;flex-shrink:0;
      padding:12px 16px 8px;
      background:linear-gradient(180deg,rgba(0,8,30,.97) 0%,rgba(0,8,20,.5) 100%);
      border-bottom:1px solid rgba(0,212,255,.12);
    }
    .hud-row1{display:flex;align-items:center;justify-content:space-between;gap:8px;}
    .j-name{
      font-family:'Orbitron',monospace;font-size:1rem;font-weight:900;
      letter-spacing:5px;color:var(--blue);
      text-shadow:0 0 18px var(--blue),0 0 36px rgba(0,212,255,.3);
      flex:1;
    }
    .hdr-btns{display:flex;align-items:center;gap:8px;}
    .hdr-btn{
      background:none;border:1px solid var(--muted);color:var(--muted);
      border-radius:6px;padding:5px 9px;cursor:pointer;
      font-family:'Share Tech Mono',monospace;font-size:.7rem;letter-spacing:1px;
      transition:all .15s;flex-shrink:0;
    }
    .hdr-btn:active{border-color:var(--blue);color:var(--blue);}
    .sys-badge{display:flex;align-items:center;gap:6px;font-family:'Share Tech Mono',monospace;font-size:.62rem;letter-spacing:2px;}
    .led{width:8px;height:8px;border-radius:50%;background:var(--green);box-shadow:0 0 8px var(--green);animation:led-blink 2s infinite;}
    .led.red{background:var(--red);box-shadow:0 0 8px var(--red);animation-duration:.5s;}
    .led.blue{background:var(--blue);box-shadow:0 0 8px var(--blue);animation-duration:.35s;}
    .led.amber{background:var(--amber);box-shadow:0 0 8px var(--amber);}
    @keyframes led-blink{0%,100%{opacity:1;}50%{opacity:.15;}}
    .sys-lbl{color:var(--green);}
    .sys-lbl.red{color:var(--red);}
    .sys-lbl.blue{color:var(--blue);}
    .hud-meta{
      margin-top:5px;display:flex;gap:12px;flex-wrap:wrap;
      font-family:'Share Tech Mono',monospace;font-size:.56rem;
      color:var(--muted);letter-spacing:.5px;
    }
    .hud-meta .v{color:rgba(0,212,255,.5);}

    /* ── Response panel ── */
    .resp-area{
      flex:1;position:relative;z-index:10;
      display:flex;align-items:center;justify-content:center;
      padding:12px 14px;overflow:hidden;
    }
    .holo-panel{
      width:100%;max-width:500px;
      background:rgba(0,12,40,.65);
      border:1px solid rgba(0,212,255,.18);
      border-radius:14px;padding:16px 18px 14px;
      backdrop-filter:blur(16px);-webkit-backdrop-filter:blur(16px);
      position:relative;max-height:100%;overflow-y:auto;
    }
    .holo-panel::-webkit-scrollbar{width:3px;}
    .holo-panel::-webkit-scrollbar-thumb{background:var(--muted);border-radius:3px;}
    .holo-panel::before{
      content:'';position:absolute;top:-1px;left:20px;
      width:44px;height:2px;
      background:linear-gradient(90deg,var(--blue),transparent);box-shadow:0 0 8px var(--blue);
    }
    .holo-panel::after{
      content:'';position:absolute;bottom:-1px;right:20px;
      width:44px;height:2px;
      background:linear-gradient(270deg,var(--blue),transparent);box-shadow:0 0 8px var(--blue);
    }
    .panel-lbl{font-family:'Orbitron',monospace;font-size:.5rem;letter-spacing:3px;color:rgba(0,212,255,.45);margin-bottom:8px;}
    .query-line{
      font-family:'Share Tech Mono',monospace;font-size:.72rem;
      color:var(--muted);margin-bottom:7px;padding-bottom:7px;
      border-bottom:1px solid rgba(0,212,255,.08);display:none;
    }
    .query-line.on{display:block;}
    .query-line::before{content:'> ';color:rgba(0,212,255,.35);}
    .resp-txt{font-size:1.05rem;line-height:1.65;color:var(--text);min-height:44px;word-break:break-word;}
    .resp-txt.blink::after{content:'|';animation:cur-blink .6s infinite;color:var(--blue);margin-left:1px;}
    @keyframes cur-blink{0%,100%{opacity:1;}50%{opacity:0;}}
    .tdots{display:none;gap:5px;margin-top:8px;}
    .tdots.on{display:flex;}
    .tdots b{width:6px;height:6px;border-radius:50%;background:var(--blue);animation:td 1.2s infinite;}
    .tdots b:nth-child(2){animation-delay:.18s;}
    .tdots b:nth-child(3){animation-delay:.36s;}
    @keyframes td{0%,60%,100%{transform:translateY(0);opacity:.3;}30%{transform:translateY(-7px);opacity:1;}}

    /* ── Code blocks ── */
    .cb{background:rgba(0,5,20,.9);border:1px solid rgba(0,212,255,.2);border-radius:8px;margin:8px 0;overflow:hidden;}
    .cb-hdr{
      display:flex;align-items:center;justify-content:space-between;
      padding:6px 12px;background:rgba(0,212,255,.06);
      border-bottom:1px solid rgba(0,212,255,.15);
    }
    .cb-lang{font-family:'Share Tech Mono',monospace;font-size:.6rem;letter-spacing:2px;color:var(--blue);}
    .cb-copy{
      background:none;border:1px solid rgba(0,212,255,.3);color:rgba(0,212,255,.7);
      border-radius:4px;padding:2px 8px;cursor:pointer;
      font-family:'Share Tech Mono',monospace;font-size:.6rem;letter-spacing:1px;
      transition:all .15s;
    }
    .cb-copy:active{background:rgba(0,212,255,.15);color:var(--blue);}
    .cb pre{padding:12px;overflow-x:auto;}
    .cb pre::-webkit-scrollbar{height:3px;}
    .cb pre::-webkit-scrollbar-thumb{background:var(--muted);}
    .cb code{font-family:'Share Tech Mono',monospace;font-size:.78rem;line-height:1.5;color:#9ae4ff;white-space:pre;}
    code.ic{background:rgba(0,212,255,.1);color:#9ae4ff;padding:1px 5px;border-radius:3px;font-family:'Share Tech Mono',monospace;font-size:.85em;}

    /* ── Quick commands ── */
    .quick-wrap{position:relative;z-index:10;padding:5px 12px;flex-shrink:0;}
    .quick-inner{display:flex;gap:6px;overflow-x:auto;scrollbar-width:none;padding-bottom:2px;}
    .quick-inner::-webkit-scrollbar{display:none;}
    .qc{
      flex-shrink:0;font-family:'Rajdhani',sans-serif;font-size:.7rem;font-weight:600;letter-spacing:.5px;
      padding:6px 11px;border-radius:6px;border:1px solid var(--muted);
      background:rgba(0,15,45,.55);color:rgba(144,200,232,.7);cursor:pointer;white-space:nowrap;
      transition:border-color .15s,color .15s;
    }
    .qc:active{border-color:var(--blue);color:var(--blue);box-shadow:0 0 10px rgba(0,212,255,.2);}

    /* ── Mic / Arc reactor ── */
    .mic-section{
      position:relative;z-index:10;flex-shrink:0;
      display:flex;flex-direction:column;align-items:center;
      padding:8px 0 6px;gap:8px;
    }
    .wave{display:flex;align-items:center;gap:3px;height:24px;opacity:0;transition:opacity .3s;}
    .wave.on{opacity:1;}
    .wave i{display:block;width:3px;border-radius:2px;background:var(--red);box-shadow:0 0 5px var(--red);animation:wbar .9s ease-in-out infinite;}
    .wave i:nth-child(1){height:7px;}
    .wave i:nth-child(2){height:16px;animation-delay:.08s;}
    .wave i:nth-child(3){height:23px;animation-delay:.16s;}
    .wave i:nth-child(4){height:12px;animation-delay:.24s;}
    .wave i:nth-child(5){height:20px;animation-delay:.12s;}
    .wave i:nth-child(6){height:14px;animation-delay:.08s;}
    .wave i:nth-child(7){height:9px;}
    @keyframes wbar{0%,100%{transform:scaleY(.3);}50%{transform:scaleY(1);}}
    .arc-wrap{position:relative;width:136px;height:136px;display:flex;align-items:center;justify-content:center;}
    .ring{position:absolute;border-radius:50%;border:1px solid rgba(0,212,255,.22);}
    .r1{width:136px;height:136px;animation:rbreath 3s ease-in-out infinite;}
    .r2{width:116px;height:116px;animation:rbreath 3s ease-in-out infinite .5s;border-color:rgba(0,212,255,.13);}
    @keyframes rbreath{0%,100%{transform:scale(1);opacity:.5;}50%{transform:scale(1.05);opacity:1;}}
    .arc-wrap.lst .r1{animation:rlisten .65s infinite;border-color:rgba(255,34,85,.75);}
    .arc-wrap.lst .r2{animation:rlisten .65s infinite .15s;border-color:rgba(255,34,85,.4);}
    @keyframes rlisten{0%,100%{transform:scale(1);}50%{transform:scale(1.11);}}
    .arc-wrap.thk .r1{animation:rspin 1.8s linear infinite;border-color:transparent;border-top-color:var(--blue);box-shadow:0 0 12px rgba(0,212,255,.3);}
    .arc-wrap.thk .r2{animation:rspin 3s linear infinite reverse;border-color:transparent;border-top-color:var(--blue2);}
    @keyframes rspin{from{transform:rotate(0);}to{transform:rotate(360deg);}}
    .arc-btn{
      width:92px;height:92px;border-radius:50%;
      background:radial-gradient(circle at 38% 38%,rgba(0,55,130,.9),rgba(0,4,18,.97));
      border:2px solid rgba(0,212,255,.45);
      box-shadow:0 0 24px rgba(0,212,255,.2),inset 0 0 24px rgba(0,212,255,.07);
      cursor:pointer;position:relative;display:flex;align-items:center;justify-content:center;
      transition:transform .15s,box-shadow .2s;-webkit-appearance:none;outline:none;
    }
    .arc-btn:active{transform:scale(.93);}
    .arc-btn::before{content:'';position:absolute;width:48px;height:48px;border-radius:50%;border:1.5px solid rgba(0,212,255,.3);}
    .arc-core{
      width:20px;height:20px;border-radius:50%;
      background:radial-gradient(circle,#9aeeff,var(--blue));
      box-shadow:0 0 18px var(--blue),0 0 36px rgba(0,212,255,.35);
      animation:core-glow 3s ease-in-out infinite;position:relative;z-index:1;
    }
    @keyframes core-glow{0%,100%{box-shadow:0 0 18px var(--blue),0 0 36px rgba(0,212,255,.35);}50%{box-shadow:0 0 28px var(--blue),0 0 56px rgba(0,212,255,.55);}}
    .arc-wrap.lst .arc-btn{border-color:rgba(255,34,85,.7);box-shadow:0 0 28px rgba(255,34,85,.35),inset 0 0 24px rgba(255,34,85,.08);}
    .arc-wrap.lst .arc-btn::before{border-color:rgba(255,34,85,.45);}
    .arc-wrap.lst .arc-core{background:radial-gradient(circle,#ffaacc,var(--red));box-shadow:0 0 18px var(--red);animation:none;}
    .arc-wrap.thk .arc-btn{box-shadow:0 0 36px rgba(0,212,255,.5),0 0 70px rgba(0,212,255,.2),inset 0 0 24px rgba(0,212,255,.12);}
    .arc-wrap.thk .arc-core{animation:core-spin 1s linear infinite;}
    @keyframes core-spin{to{filter:hue-rotate(360deg);}}
    .mic-lbl{font-family:'Orbitron',monospace;font-size:.56rem;letter-spacing:3px;color:var(--muted);text-align:center;transition:color .3s;}
    .mic-lbl.red{color:var(--red);}
    .mic-lbl.blue{color:var(--blue);}
    .mic-lbl.green{color:var(--green);}

    /* ── Text input bar ── */
    .input-bar{
      position:relative;z-index:10;flex-shrink:0;
      padding:8px 14px max(env(safe-area-inset-bottom,12px),12px);
      background:rgba(0,6,22,.95);border-top:1px solid rgba(0,212,255,.1);
      display:flex;gap:8px;align-items:center;
    }
    #txtIn{
      flex:1;background:rgba(0,12,40,.8);border:1px solid rgba(0,212,255,.2);
      border-radius:22px;padding:10px 16px;
      color:var(--text);font-size:.9rem;font-family:'Rajdhani',sans-serif;
      outline:none;transition:border-color .2s,box-shadow .2s;
    }
    #txtIn:focus{border-color:rgba(0,212,255,.5);box-shadow:0 0 0 3px rgba(0,212,255,.08);}
    #txtIn::placeholder{color:var(--muted);}
    .send-btn{
      width:42px;height:42px;border-radius:50%;border:none;cursor:pointer;flex-shrink:0;
      background:linear-gradient(135deg,var(--blue2),var(--blue));
      color:#fff;font-size:1rem;display:flex;align-items:center;justify-content:center;
      box-shadow:0 2px 12px rgba(0,212,255,.3);transition:transform .15s;
    }
    .send-btn:active{transform:scale(.9);}

    /* ── Controls drawer ── */
    .drawer-overlay{
      position:fixed;inset:0;z-index:100;background:rgba(0,0,0,.6);
      display:none;backdrop-filter:blur(4px);
    }
    .drawer-overlay.on{display:block;}
    .drawer{
      position:fixed;bottom:0;left:0;right:0;z-index:101;
      background:rgba(0,6,22,.97);border-top:1px solid rgba(0,212,255,.25);
      border-radius:20px 20px 0 0;
      padding:20px 20px max(env(safe-area-inset-bottom,20px),20px);
      transform:translateY(110%);transition:transform .35s cubic-bezier(.4,0,.2,1);
      max-height:80vh;overflow-y:auto;
    }
    .drawer.on{transform:translateY(0);}
    .drawer-pill{width:40px;height:4px;background:var(--muted);border-radius:2px;margin:0 auto 18px;}
    .drawer-hdr{
      display:flex;justify-content:space-between;align-items:center;margin-bottom:16px;
      font-family:'Orbitron',monospace;font-size:.58rem;letter-spacing:3px;color:var(--blue);
    }
    .drawer-hdr button{
      background:none;border:1px solid var(--muted);color:var(--muted);
      border-radius:5px;padding:3px 8px;cursor:pointer;font-size:.75rem;transition:all .15s;
    }
    .drawer-hdr button:active{border-color:var(--red);color:var(--red);}
    .ctrl-grid{display:grid;grid-template-columns:repeat(3,1fr);gap:10px;margin-bottom:20px;}
    .ctrl-tile{
      background:rgba(0,15,50,.7);border:1px solid var(--muted);
      border-radius:12px;padding:16px 10px;cursor:pointer;
      display:flex;flex-direction:column;align-items:center;gap:7px;
      font-family:'Rajdhani',sans-serif;font-size:.72rem;font-weight:600;letter-spacing:.5px;
      color:rgba(144,200,232,.7);transition:all .2s;
    }
    .ctrl-tile:active,.ctrl-tile.active{
      border-color:var(--blue);color:var(--blue);
      background:rgba(0,212,255,.08);box-shadow:0 0 12px rgba(0,212,255,.2);
    }
    .ctrl-tile span:first-child{font-size:1.4rem;}
    .ctrl-section-lbl{
      font-family:'Share Tech Mono',monospace;font-size:.58rem;letter-spacing:2px;
      color:var(--muted);margin-bottom:10px;
    }
    .local-pc-row{display:flex;gap:8px;align-items:center;}
    .local-pc-row input{
      flex:1;background:rgba(0,12,40,.8);border:1px solid rgba(0,212,255,.2);
      border-radius:8px;padding:9px 12px;color:var(--text);
      font-family:'Share Tech Mono',monospace;font-size:.75rem;outline:none;
    }
    .local-pc-row input:focus{border-color:rgba(0,212,255,.5);}
    .local-pc-row input::placeholder{color:var(--muted);}
    .local-pc-row button{
      padding:9px 14px;background:linear-gradient(135deg,var(--blue2),var(--blue));
      border:none;border-radius:8px;color:#fff;font-size:.75rem;
      font-family:'Rajdhani',sans-serif;font-weight:700;letter-spacing:1px;cursor:pointer;
    }

    /* ── Voice picker modal ── */
    .vmodal{display:none;position:fixed;inset:0;z-index:200;background:rgba(0,0,0,.82);align-items:flex-end;justify-content:center;}
    .vmodal.on{display:flex;}
    .vmodal-inner{
      width:100%;max-width:480px;background:#00060f;
      border:1px solid rgba(0,212,255,.25);border-bottom:none;border-radius:16px 16px 0 0;
      padding:18px;max-height:70vh;display:flex;flex-direction:column;
    }
    .vmodal-hdr{
      display:flex;justify-content:space-between;align-items:center;
      font-family:'Orbitron',monospace;font-size:.58rem;letter-spacing:3px;color:var(--blue);margin-bottom:12px;flex-shrink:0;
    }
    .vmodal-hdr button{background:none;border:1px solid var(--muted);color:var(--muted);border-radius:4px;padding:3px 8px;cursor:pointer;font-size:.8rem;}
    .vlist{overflow-y:auto;display:flex;flex-direction:column;gap:5px;scrollbar-width:thin;scrollbar-color:var(--muted) transparent;}
    .vitem{padding:11px 13px;border-radius:8px;border:1px solid var(--muted);cursor:pointer;transition:all .15s;font-family:'Rajdhani',sans-serif;}
    .vitem:active,.vitem.sel{border-color:var(--blue);color:var(--blue);background:rgba(0,212,255,.06);}
    .vitem .vn{font-size:.88rem;font-weight:600;color:var(--text);}
    .vitem.sel .vn{color:var(--blue);}
    .vitem .vl{font-size:.62rem;font-family:'Share Tech Mono',monospace;color:var(--muted);margin-top:2px;}
    .sel-mark{float:right;color:var(--green);font-size:.8rem;display:none;}
    .vitem.sel .sel-mark{display:inline;}

    /* ── Toast ── */
    .toast{
      position:fixed;top:74px;left:50%;
      transform:translateX(-50%) translateY(-10px);
      background:rgba(0,15,50,.97);border:1px solid rgba(0,212,255,.3);
      border-radius:8px;padding:7px 18px;
      font-family:'Share Tech Mono',monospace;font-size:.65rem;
      color:var(--blue);letter-spacing:1px;
      z-index:300;opacity:0;transition:all .25s;pointer-events:none;white-space:nowrap;
    }
    .toast.on{opacity:1;transform:translateX(-50%) translateY(0);}

    /* ── No voice overlay ── */
    #noVoice{
      display:none;position:fixed;inset:0;z-index:400;
      background:rgba(0,0,0,.93);flex-direction:column;
      align-items:center;justify-content:center;gap:14px;padding:32px;text-align:center;
    }
    #noVoice.on{display:flex;}
    #noVoice h2{font-family:'Orbitron',monospace;font-size:.95rem;letter-spacing:2px;color:var(--red);}
    #noVoice p{color:var(--muted);font-size:.88rem;line-height:1.6;}

    @media(min-height:700px){
      .arc-wrap{width:152px;height:152px;}
      .r1{width:152px;height:152px;}
      .r2{width:130px;height:130px;}
      .arc-btn{width:104px;height:104px;}
    }
    @media(max-height:600px){
      .mic-section{padding:4px 0;}
      .wave{display:none;}
    }
  </style>
</head>
<body>

<div class="bg-glow"></div>
<div class="hex-grid"></div>
<div class="scanlines"></div>
<div class="hc tl"></div><div class="hc tr"></div>
<div class="hc bl"></div><div class="hc br"></div>

<!-- Offline banner -->
<div class="offline-bar" id="offlineBar">⚠ OFFLINE MODE — CLOUD UNREACHABLE · USING LOCAL FALLBACK</div>

<!-- Install banner -->
<div class="install-bar" id="installBar">
  <span>📱 ADD J.A.R.V.I.S. TO HOME SCREEN</span>
  <div style="display:flex;gap:8px;">
    <button onclick="installApp()" id="installBtn">INSTALL</button>
    <button class="ib-close" onclick="dismissInstall()">✕</button>
  </div>
</div>

<!-- Header -->
<header class="hud-header">
  <div class="hud-row1">
    <div class="j-name">J.A.R.V.I.S.</div>
    <div class="hdr-btns">
      <button class="hdr-btn" onclick="openVoicePicker()" title="Change voice">🔊 VOICE</button>
      <button class="hdr-btn" onclick="openDrawer()" title="System controls">⚙ CTRL</button>
      <div class="sys-badge">
        <div class="led" id="led"></div>
        <span class="sys-lbl" id="sysLbl">ONLINE</span>
      </div>
    </div>
  </div>
  <div class="hud-meta">
    <span>SYS&nbsp;<span class="v">NOMINAL</span></span>
    <span>AI&nbsp;<span class="v" id="aiModel">GEMINI+GROQ</span></span>
    <span>MEM&nbsp;<span class="v">ACTIVE</span></span>
    <span>NET&nbsp;<span class="v" id="netStatus">CLOUD</span></span>
  </div>
</header>

<!-- Response panel -->
<div class="resp-area">
  <div class="holo-panel">
    <div class="panel-lbl">◈ JARVIS OUTPUT</div>
    <div class="query-line" id="qLine"></div>
    <div class="resp-txt" id="rTxt">Initialising systems&hellip;</div>
    <div class="tdots" id="tDots"><b></b><b></b><b></b></div>
  </div>
</div>

<!-- Quick commands -->
<div class="quick-wrap">
  <div class="quick-inner">
    <button class="qc" id="qc1" onclick="runCmd('world news')">🌍 NEWS</button>
    <button class="qc" id="qc2" onclick="runCmd('weather today')">🌤 WEATHER</button>
    <button class="qc" id="qc3" onclick="runCmd('open youtube')">▶ YOUTUBE</button>
    <button class="qc" id="qc4" onclick="runCmd('open whatsapp')">💬 WHATSAPP</button>
    <button class="qc" id="qc5" onclick="runCmd('open spotify')">🎵 SPOTIFY</button>
    <button class="qc" id="qc6" onclick="runCmd('open maps')">🗺 MAPS</button>
    <button class="qc" id="qc7" onclick="runCmd('what time is it')">🕐 TIME</button>
    <button class="qc" id="qc8" onclick="runCmd('motivate me')">⚡ MOTIVATE</button>
    <button class="qc" id="qc9" onclick="runCmd('tell me a joke')">😄 JOKE</button>
    <button class="qc" id="qc10" onclick="runCmd('open instagram')">📸 INSTAGRAM</button>
    <button class="qc" id="qc11" onclick="runCmd('open netflix')">🎬 NETFLIX</button>
    <button class="qc" id="qc12" onclick="runCmd('open github')">💻 GITHUB</button>
  </div>
</div>

<!-- Arc reactor -->
<div class="mic-section">
  <div class="wave" id="wave">
    <i></i><i></i><i></i><i></i><i></i><i></i><i></i>
  </div>
  <div class="arc-wrap" id="arcWrap">
    <div class="ring r1"></div>
    <div class="ring r2"></div>
    <button class="arc-btn" id="arcBtn" onclick="toggleMic()" title="Tap to speak">
      <div class="arc-core"></div>
    </button>
  </div>
  <div class="mic-lbl" id="micLbl">TAP TO SPEAK</div>
</div>

<!-- Text input bar -->
<div class="input-bar">
  <input id="txtIn" type="text" placeholder="Type a command…" autocomplete="off" autocorrect="off"/>
  <button class="send-btn" onclick="sendTyped()" title="Send">➤</button>
</div>

<!-- Controls drawer -->
<div class="drawer-overlay" id="drawerOverlay" onclick="closeDrawer()"></div>
<div class="drawer" id="drawer">
  <div class="drawer-pill"></div>
  <div class="drawer-hdr">
    <span>⚙ SYSTEM CONTROLS</span>
    <button onclick="closeDrawer()">✕ CLOSE</button>
  </div>

  <div class="ctrl-grid">
    <button class="ctrl-tile" id="torchTile" onclick="toggleTorch()">
      <span>🔦</span><span>TORCH</span>
    </button>
    <button class="ctrl-tile" onclick="doVibrate()">
      <span>📳</span><span>VIBRATE</span>
    </button>
    <button class="ctrl-tile" id="wakeTile" onclick="toggleWakeLock()">
      <span>💡</span><span>WAKE LOCK</span>
    </button>
    <button class="ctrl-tile" onclick="showBattery()">
      <span>🔋</span><span>BATTERY</span>
    </button>
    <button class="ctrl-tile" onclick="copyLast()">
      <span>📋</span><span>COPY</span>
    </button>
    <button class="ctrl-tile" onclick="shareLast()">
      <span>📤</span><span>SHARE</span>
    </button>
  </div>

  <div class="ctrl-section-lbl">LOCAL PC FALLBACK (Ollama / jarvis_web.py)</div>
  <div class="local-pc-row">
    <input id="pcIpIn" type="text" placeholder="e.g. 192.168.1.5:5000"/>
    <button onclick="saveLocalPC()">SAVE</button>
  </div>
</div>

<!-- Voice picker -->
<div class="vmodal" id="vModal">
  <div class="vmodal-inner">
    <div class="vmodal-hdr">
      <span>🔊 SELECT JARVIS VOICE</span>
      <button onclick="closeVoicePicker()">✕ CLOSE</button>
    </div>
    <div class="vlist" id="vList"></div>
  </div>
</div>

<!-- No voice overlay -->
<div id="noVoice">
  <h2>⚠ VOICE NOT SUPPORTED</h2>
  <p>Please use <strong>Chrome on Android</strong><br>or <strong>Safari on iOS</strong>.<br><br>You can still type commands below.</p>
</div>

<!-- Toast -->
<div class="toast" id="toast"></div>

<script>
'use strict';

// ── DOM refs ──────────────────────────────────────────────────────────────
const arcWrap = document.getElementById('arcWrap');
const micLbl  = document.getElementById('micLbl');
const led     = document.getElementById('led');
const sysLbl  = document.getElementById('sysLbl');
const rTxt    = document.getElementById('rTxt');
const qLine   = document.getElementById('qLine');
const tDots   = document.getElementById('tDots');
const waveEl  = document.getElementById('wave');
const toastEl = document.getElementById('toast');

// ── Voice synthesis ───────────────────────────────────────────────────────
const synth = window.speechSynthesis;
let voices  = [];
let selVoice = null;

// Male voice priority order
const MALE_TESTS = [
  v => /google uk english male/i.test(v.name),
  v => /google (us|en-us) english male/i.test(v.name),
  v => /daniel/i.test(v.name) && /^en/i.test(v.lang),
  v => /alex|mark|james|ryan|guy|fred|thomas|arthur/i.test(v.name) && /^en/i.test(v.lang),
  v => /male/i.test(v.name) && /^en/i.test(v.lang),
  v => /^en-IN/i.test(v.lang) && !/female|zira|siri|google uk english female/i.test(v.name),
  v => /^en-GB/i.test(v.lang) && !/female|zira/i.test(v.name),
  v => /^en/i.test(v.lang) && !/female|zira|karen|moira|samantha|tessa|veena/i.test(v.name),
];

function loadVoices() {
  voices = synth.getVoices();
  const saved = localStorage.getItem('jarvisVoiceURI');
  if (saved && !selVoice) {
    const f = voices.find(v => v.voiceURI === saved);
    if (f) { selVoice = f; return; }
  }
  // Auto-pick best male voice if not saved
  if (!selVoice || selVoice._auto) {
    for (const test of MALE_TESTS) {
      const v = voices.find(test);
      if (v) { selVoice = v; selVoice._auto = true; break; }
    }
    if (!selVoice && voices.length > 0) {
      selVoice = voices.find(v => /^en/i.test(v.lang)) || voices[0];
      selVoice._auto = true;
    }
  }
}
if (synth.onvoiceschanged !== undefined) synth.onvoiceschanged = loadVoices;
loadVoices();

function speak(text) {
  if (!synth) return;
  synth.cancel();
  // Strip markdown for TTS
  const plain = text
    .replace(/```[\s\S]*?```/g, ' code block. ')
    .replace(/`([^`]+)`/g, '$1')
    .replace(/\*\*([^*]+)\*\*/g, '$1')
    .replace(/\*([^*]+)\*/g, '$1')
    .replace(/#{1,6}\s/g, '')
    .replace(/\n/g, ' ')
    .trim();
  const u = new SpeechSynthesisUtterance(plain);
  u.rate = 0.92; u.pitch = 0.82; u.volume = 1;
  if (selVoice) u.voice = selVoice;
  // Workaround for Chrome Android bug (voice resets)
  setTimeout(() => { if (selVoice) u.voice = selVoice; synth.speak(u); }, 50);
}

// ── Service Worker (PWA) ──────────────────────────────────────────────────
if ('serviceWorker' in navigator) {
  navigator.serviceWorker.register('/sw.js').catch(e => console.log('SW:', e));
}

// ── Offline detection ─────────────────────────────────────────────────────
const offlineBar = document.getElementById('offlineBar');
const netStat    = document.getElementById('netStatus');

function updateOnlineStatus() {
  const online = navigator.onLine;
  offlineBar.className = online ? 'offline-bar' : 'offline-bar on';
  netStat.textContent  = online ? 'CLOUD' : 'OFFLINE';
}
window.addEventListener('online',  updateOnlineStatus);
window.addEventListener('offline', updateOnlineStatus);
updateOnlineStatus();

// Local PC IP (for Ollama fallback)
const pcIpIn = document.getElementById('pcIpIn');
pcIpIn.value = localStorage.getItem('jarvisLocalPC') || '';

function saveLocalPC() {
  const v = pcIpIn.value.trim();
  if (v) { localStorage.setItem('jarvisLocalPC', v); showToast('LOCAL PC SAVED: ' + v); }
  else  { localStorage.removeItem('jarvisLocalPC'); showToast('LOCAL PC CLEARED'); }
}

function getLocalPC() { return localStorage.getItem('jarvisLocalPC') || ''; }

// ── Battery ───────────────────────────────────────────────────────────────
async function showBattery() {
  let info = 'Battery API not supported on this browser.';
  try {
    if ('getBattery' in navigator) {
      const b = await navigator.getBattery();
      const lvl = Math.round(b.level * 100);
      const state = b.charging ? '⚡ Charging' : (b.discharging ? '🔋 Discharging' : '');
      const eta = b.chargingTime && b.chargingTime !== Infinity
        ? ` — full in ${Math.round(b.chargingTime/60)} min` : '';
      info = `Battery: ${lvl}% ${state}${eta}`;
    }
  } catch(e) { info = 'Battery status unavailable: ' + e.message; }
  showResponse(info);
  speak(info);
  closeDrawer();
}

// ── Wake lock ─────────────────────────────────────────────────────────────
let wakeLock = null;
const wakeTile = document.getElementById('wakeTile');
async function toggleWakeLock() {
  try {
    if (wakeLock) {
      await wakeLock.release(); wakeLock = null;
      wakeTile.classList.remove('active');
      showToast('WAKE LOCK RELEASED');
    } else {
      wakeLock = await navigator.wakeLock.request('screen');
      wakeTile.classList.add('active');
      showToast('SCREEN WAKE LOCK ACTIVE');
      wakeLock.addEventListener('release', () => {
        wakeLock = null; wakeTile.classList.remove('active');
      });
    }
  } catch(e) { showToast('WAKE LOCK: ' + e.message.toUpperCase()); }
}

// ── Torch ─────────────────────────────────────────────────────────────────
let torchStream = null;
let torchTrack  = null;
let torchOn     = false;
const torchTile = document.getElementById('torchTile');
async function toggleTorch() {
  if (torchOn) {
    try { await torchTrack?.applyConstraints({advanced: [{torch: false}]}); } catch(_){}
    torchTrack?.stop();
    torchStream?.getTracks().forEach(t => t.stop());
    torchStream = torchTrack = null; torchOn = false;
    torchTile.classList.remove('active');
    showToast('TORCH OFF');
    return;
  }
  try {
    torchStream = await navigator.mediaDevices.getUserMedia({video: {facingMode: 'environment'}});
    torchTrack  = torchStream.getVideoTracks()[0];
    const caps  = torchTrack.getCapabilities?.() || {};
    if ('torch' in caps) {
      await torchTrack.applyConstraints({advanced: [{torch: true}]});
      torchOn = true; torchTile.classList.add('active');
      showToast('TORCH ON');
    } else {
      torchTrack.stop(); torchStream.getTracks().forEach(t => t.stop());
      torchStream = torchTrack = null;
      showToast('TORCH NOT SUPPORTED ON THIS DEVICE');
    }
  } catch(e) { showToast('CAMERA ACCESS DENIED'); }
}

// ── Vibrate ───────────────────────────────────────────────────────────────
function doVibrate() {
  if ('vibrate' in navigator) {
    navigator.vibrate([150, 80, 150]);
    showToast('VIBRATING');
  } else { showToast('VIBRATION NOT SUPPORTED'); }
}

// ── Copy / Share last response ────────────────────────────────────────────
let lastReply = '';
function copyLast() {
  if (!lastReply) { showToast('NOTHING TO COPY'); return; }
  navigator.clipboard?.writeText(lastReply).then(() => showToast('COPIED TO CLIPBOARD'))
    .catch(() => showToast('CLIPBOARD UNAVAILABLE'));
}
function shareLast() {
  if (!lastReply) { showToast('NOTHING TO SHARE'); return; }
  navigator.share?.({title: 'J.A.R.V.I.S.', text: lastReply})
    .catch(() => showToast('SHARE CANCELLED'));
}

// ── Controls drawer ───────────────────────────────────────────────────────
const drawer        = document.getElementById('drawer');
const drawerOverlay = document.getElementById('drawerOverlay');
function openDrawer()  { drawer.classList.add('on'); drawerOverlay.classList.add('on'); }
function closeDrawer() { drawer.classList.remove('on'); drawerOverlay.classList.remove('on'); }

// ── Toast ─────────────────────────────────────────────────────────────────
let _toastTimer = null;
function showToast(msg) {
  toastEl.textContent = msg; toastEl.classList.add('on');
  if (_toastTimer) clearTimeout(_toastTimer);
  _toastTimer = setTimeout(() => toastEl.classList.remove('on'), 3200);
}

// ── Markdown / code rendering ─────────────────────────────────────────────
function renderMarkdown(text) {
  // Escape HTML first
  let t = text
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;');

  // Code blocks
  let cbIdx = 0;
  t = t.replace(/```(\w*)\n?([\s\S]*?)```/g, (_, lang, code) => {
    const id = 'cb' + (cbIdx++);
    const l  = lang || 'code';
    // Need to un-escape the code content for display
    return `<div class="cb"><div class="cb-hdr"><span class="cb-lang">${l.toUpperCase()}</span><button class="cb-copy" onclick="copyCb('${id}')">COPY</button></div><pre><code id="${id}">${code.trim()}</code></pre></div>`;
  });

  // Inline code
  t = t.replace(/`([^`\n]+)`/g, '<code class="ic">$1</code>');
  // Bold
  t = t.replace(/\*\*([^*\n]+)\*\*/g, '<strong>$1</strong>');
  // Newlines
  t = t.replace(/\n/g, '<br>');
  return t;
}

function copyCb(id) {
  const el = document.getElementById(id);
  if (el) navigator.clipboard?.writeText(el.textContent)
    .then(() => showToast('CODE COPIED'));
}

// ── State machine ─────────────────────────────────────────────────────────
let appState = 'idle';
function setState(s) {
  appState = s;
  arcWrap.className = 'arc-wrap' + (s==='listening' ? ' lst' : s==='thinking' ? ' thk' : '');
  waveEl.className  = 'wave' + (s==='listening' ? ' on' : '');
  tDots.className   = 'tdots' + (s==='thinking' ? ' on' : '');
  if (s === 'idle') {
    micLbl.textContent = 'TAP TO SPEAK'; micLbl.className = 'mic-lbl';
    led.className = 'led'; sysLbl.textContent = 'ONLINE'; sysLbl.className = 'sys-lbl';
  } else if (s === 'listening') {
    micLbl.textContent = 'LISTENING...'; micLbl.className = 'mic-lbl red';
    led.className = 'led red'; sysLbl.textContent = 'LISTENING'; sysLbl.className = 'sys-lbl red';
  } else if (s === 'thinking') {
    micLbl.textContent = 'PROCESSING...'; micLbl.className = 'mic-lbl blue';
    led.className = 'led blue'; sysLbl.textContent = 'THINKING'; sysLbl.className = 'sys-lbl blue';
  } else if (s === 'speaking') {
    micLbl.textContent = 'RESPONDING...'; micLbl.className = 'mic-lbl green';
    led.className = 'led'; sysLbl.textContent = 'ONLINE'; sysLbl.className = 'sys-lbl';
  }
}

// ── Typewriter / response renderer ────────────────────────────────────────
let twTimer = null;
function typewrite(text) {
  if (twTimer) clearInterval(twTimer);
  rTxt.className = 'resp-txt blink'; rTxt.textContent = '';
  let i = 0;
  twTimer = setInterval(() => {
    if (i < text.length) { rTxt.textContent += text[i++]; }
    else {
      rTxt.className = 'resp-txt';
      clearInterval(twTimer); twTimer = null;
      setTimeout(() => setState('idle'), 2500);
    }
  }, 12);
}

function showResponse(text) {
  if (twTimer) { clearInterval(twTimer); twTimer = null; }
  lastReply = text;
  const hasCode = text.includes('```');
  if (hasCode) {
    rTxt.innerHTML = renderMarkdown(text);
    rTxt.className = 'resp-txt';
    setTimeout(() => setState('idle'), 4000);
  } else {
    typewrite(text);
  }
}

// ── Offline canned responses ──────────────────────────────────────────────
const JOKES = [
  "Why do programmers prefer dark mode? Because light attracts bugs, sir.",
  "I would tell you a joke about the cloud, but I'm afraid it's currently offline.",
  "Why did the developer go broke? Because he used up all his cache.",
  "What do you call a fish without eyes? An FSH. I'll see myself out.",
];
function offlineReply(text) {
  const t = text.toLowerCase();
  if (/\btime\b/.test(t)) return `It's ${new Date().toLocaleTimeString('en-IN',{hour:'2-digit',minute:'2-digit',hour12:true})} local time.`;
  if (/\bdate\b|\btoday\b/.test(t)) return `Today is ${new Date().toLocaleDateString('en-IN',{weekday:'long',year:'numeric',month:'long',day:'numeric'})}.`;
  if (/joke/.test(t)) return JOKES[Math.floor(Math.random()*JOKES.length)];
  if (/motivat/.test(t)) return "Keep going, sir. Every master was once a beginner.";
  if (/hello|hi\b|hey/.test(t)) return "Hello, sir. I'm running in offline mode at the moment.";
  return "I'm currently offline, sir. Cloud systems are unreachable. Try your local PC address in settings, or check your connection.";
}

// ── Command runner ────────────────────────────────────────────────────────
// Local voice shortcuts handled before server call
const LOCAL_ACTIONS = {
  'torch on':        () => toggleTorch(),
  'turn on torch':   () => toggleTorch(),
  'flashlight on':   () => toggleTorch(),
  'torch off':       () => toggleTorch(),
  'turn off torch':  () => toggleTorch(),
  'flashlight off':  () => toggleTorch(),
  'vibrate':         () => doVibrate(),
  'buzz':            () => doVibrate(),
  'copy that':       () => copyLast(),
  'copy':            () => copyLast(),
  'share that':      () => shareLast(),
  'keep screen on':  () => toggleWakeLock(),
  'controls':        () => openDrawer(),
  'open controls':   () => openDrawer(),
};

async function runCmd(text) {
  if (!text.trim() || appState === 'thinking') return;
  synth.cancel();

  // Local commands
  const lower = text.toLowerCase().trim();
  for (const [kw, fn] of Object.entries(LOCAL_ACTIONS)) {
    if (lower.includes(kw)) {
      fn();
      return;
    }
  }
  // Battery local
  if (/battery/.test(lower)) { await showBattery(); return; }

  setState('thinking');
  qLine.textContent  = text; qLine.className = 'query-line on';
  rTxt.textContent   = ''; rTxt.className = 'resp-txt';

  let data = null;

  // 1. Try cloud server
  if (navigator.onLine) {
    try {
      const res = await fetch('/command', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({command: text}),
        signal: AbortSignal.timeout(45000),
      });
      data = await res.json();
    } catch(e) { console.log('[CLOUD] failed:', e.message); }
  }

  // 2. Try local PC (jarvis_web.py with Ollama)
  if (!data) {
    const pc = getLocalPC();
    if (pc) {
      try {
        const url = pc.startsWith('http') ? pc : `http://${pc}`;
        const res = await fetch(`${url}/command`, {
          method: 'POST',
          headers: {'Content-Type': 'application/json'},
          body: JSON.stringify({command: text}),
          signal: AbortSignal.timeout(60000),
        });
        data = await res.json();
        data._local = true;
      } catch(e) { console.log('[LOCAL PC] failed:', e.message); }
    }
  }

  // 3. Offline fallback
  if (!data) {
    setState('speaking');
    const rep = offlineReply(text);
    showResponse(rep); speak(rep);
    return;
  }

  setState('speaking');
  const reply = data.reply || 'No response received.';
  showResponse(reply);
  speak(reply);

  if (data.action === 'open_url' && data.url) {
    setTimeout(() => {
      showToast('>> OPENING ' + (data.url_label || 'LINK'));
      window.open(data.url, '_blank');
    }, 1000);
  }
}

// ── Voice recognition ─────────────────────────────────────────────────────
const SR = window.SpeechRecognition || window.webkitSpeechRecognition;
let rec = null, micActive = false;
if (!SR) {
  document.getElementById('noVoice').classList.add('on');
} else {
  rec = new SR();
  rec.lang = 'en-IN';
  rec.interimResults = false;
  rec.maxAlternatives = 1;
  rec.continuous = false;
  rec.onresult = e => { micActive = false; runCmd(e.results[0][0].transcript); };
  rec.onerror  = e => {
    micActive = false; setState('idle');
    const errs = {
      'no-speech':   'NO SPEECH DETECTED',
      'not-allowed': 'MIC ACCESS DENIED',
      'network':     'NETWORK ERROR',
    };
    showToast(errs[e.error] || 'MIC ERROR: ' + e.error.toUpperCase());
  };
  rec.onend = () => { if (micActive) { micActive = false; setState('idle'); } };
}

function toggleMic() {
  if (!rec) return;
  if (micActive) { rec.stop(); micActive = false; setState('idle'); }
  else {
    if (appState === 'thinking') return;
    synth.cancel();
    try { rec.start(); micActive = true; setState('listening'); }
    catch(e) { micActive = false; setState('idle'); showToast('MIC UNAVAILABLE'); }
  }
}

// ── Text input ────────────────────────────────────────────────────────────
const txtIn = document.getElementById('txtIn');
function sendTyped() {
  const t = txtIn.value.trim();
  if (!t) return;
  txtIn.value = '';
  runCmd(t);
}
txtIn.addEventListener('keydown', e => { if (e.key === 'Enter') sendTyped(); });

// ── Voice picker ──────────────────────────────────────────────────────────
function openVoicePicker() {
  if (voices.length === 0) loadVoices();
  const list = document.getElementById('vList');
  list.innerHTML = '';
  const eng = voices.filter(v => /^en/i.test(v.lang));
  if (eng.length === 0) {
    list.innerHTML = '<p style="color:var(--muted);text-align:center;padding:20px;font-size:.85rem;">No voices loaded yet. Wait a moment.</p>';
  } else {
    eng.forEach(v => {
      const isSel = selVoice && selVoice.voiceURI === v.voiceURI;
      const item  = document.createElement('div');
      item.className = 'vitem' + (isSel ? ' sel' : '');
      item.innerHTML = `<span class="vn">${v.name}<span class="sel-mark"> ✓</span></span><div class="vl">${v.lang}${v.localService ? ' · LOCAL' : ' · NETWORK'}</div>`;
      item.onclick = () => {
        selVoice = v; selVoice._auto = false;
        localStorage.setItem('jarvisVoiceURI', v.voiceURI);
        closeVoicePicker();
        synth.cancel();
        const u = new SpeechSynthesisUtterance('Voice confirmed. I am J.A.R.V.I.S., at your service.');
        u.voice = v; u.rate = 0.92; u.pitch = 0.82; u.volume = 1;
        synth.speak(u);
        showToast('VOICE SET: ' + v.name.toUpperCase());
      };
      list.appendChild(item);
    });
  }
  document.getElementById('vModal').classList.add('on');
}
function closeVoicePicker() { document.getElementById('vModal').classList.remove('on'); }
document.getElementById('vModal').addEventListener('click', function(e) { if (e.target === this) closeVoicePicker(); });

// ── PWA Install ───────────────────────────────────────────────────────────
let deferredPrompt = null;
const installBar = document.getElementById('installBar');

window.addEventListener('beforeinstallprompt', e => {
  e.preventDefault(); deferredPrompt = e;
  const dismissed = localStorage.getItem('installDismissed');
  if (!dismissed) installBar.classList.add('on');
});
async function installApp() {
  if (!deferredPrompt) return;
  deferredPrompt.prompt();
  const { outcome } = await deferredPrompt.userChoice;
  deferredPrompt = null;
  installBar.classList.remove('on');
  if (outcome === 'accepted') showToast('JARVIS INSTALLED ✓');
}
function dismissInstall() {
  installBar.classList.remove('on');
  localStorage.setItem('installDismissed', '1');
}

// ── Init ──────────────────────────────────────────────────────────────────
(async () => {
  // Load voices (async on some browsers)
  if (voices.length === 0) {
    await new Promise(res => {
      if (synth.getVoices().length > 0) { loadVoices(); res(); }
      else { const h = () => { loadVoices(); synth.onvoiceschanged = null; res(); }; synth.onvoiceschanged = h; setTimeout(res, 2000); }
    });
  }

  try {
    const res  = await fetch('/greet');
    const data = await res.json();
    setState('speaking');
    showResponse(data.greeting);
    speak(data.greeting);
  } catch(e) {
    showResponse('J.A.R.V.I.S. online. Tap the reactor or type a command.');
    setState('idle');
  }
})();
</script>
</body>
</html>"""

# ══════════════════════════════════════════════════════════════════════
# FLASK APP
# ══════════════════════════════════════════════════════════════════════

app = Flask(__name__)

@app.route("/")
def index():
    return render_template_string(HTML_PAGE)

@app.route("/greet")
def greet():
    h = datetime.datetime.now(IST).hour
    if   5  <= h < 12: msg = "Good morning, sir. Systems are fully operational. How may I assist you today?"
    elif 12 <= h < 17: msg = "Good afternoon. All systems nominal. What can I do for you?"
    elif 17 <= h < 21: msg = "Good evening. Ready when you are, sir."
    elif 21 <= h < 24: msg = "Good night, sir. Working late again, I see. I'm at your service."
    else:               msg = "It is past midnight. Still at it, sir? I'm here whenever you need me."
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
        "status":      "online",
        "ai_primary":  GEMINI_MODEL if GEMINI_API_KEY else "not configured",
        "ai_fallback": GROQ_MODEL   if (_groq_client) else "not configured",
        "memory":      MEM_OK,
        "docs":        _mem_col.count() if MEM_OK else 0,
        "version":     "v2.0",
    })

@app.route("/sw.js")
def service_worker():
    return Response(SERVICE_WORKER, mimetype="application/javascript",
                    headers={"Service-Worker-Allowed": "/"})

@app.route("/manifest.json")
def manifest():
    return jsonify({
        "name":             "J.A.R.V.I.S.",
        "short_name":       "JARVIS",
        "description":      "Your personal AI — Gemini-powered, always ready",
        "start_url":        "/",
        "display":          "standalone",
        "orientation":      "portrait",
        "background_color": "#000814",
        "theme_color":      "#00d4ff",
        "icons": [
            {"src": "/icon.svg", "sizes": "any", "type": "image/svg+xml", "purpose": "any maskable"}
        ],
    })

@app.route("/icon.svg")
def icon():
    svg = (
        '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 100 100">'
        '<rect width="100" height="100" fill="#000814"/>'
        '<circle cx="50" cy="50" r="42" fill="none" stroke="#00d4ff" stroke-width="2"/>'
        '<circle cx="50" cy="50" r="30" fill="none" stroke="#00d4ff" stroke-width="1" opacity=".5"/>'
        '<circle cx="50" cy="50" r="11" fill="#00d4ff" filter="url(#g)"/>'
        '<defs><filter id="g"><feGaussianBlur stdDeviation="3"/></filter></defs>'
        '<circle cx="50" cy="50" r="11" fill="#00d4ff"/>'
        '</svg>'
    )
    return Response(svg, mimetype="image/svg+xml")

# ══════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("+--------------------------------------------------+")
    print("|  J.A.R.V.I.S. v2 - Cloud Edition                |")
    print("+--------------------------------------------------+")
    print(f"|  Gemini API : {'OK' if GEMINI_API_KEY else 'NOT SET -- add GEMINI_API_KEY'}")
    print(f"|  Groq API   : {'OK (fallback)' if GROQ_API_KEY else 'NOT SET -- add GROQ_API_KEY'}")
    print(f"|  News API   : {'OK' if NEWSAPI_KEY else 'Optional (NEWSAPI_KEY)'}")
    print(f"|  Running on : http://0.0.0.0:{PORT}")
    print("+--------------------------------------------------+")

    if not GEMINI_API_KEY and not GROQ_API_KEY:
        print()
        print("WARNING: No AI keys set. Set GEMINI_API_KEY or GROQ_API_KEY.")
        print("  Gemini key (free): https://aistudio.google.com/apikey")
        print("  Groq key   (free): https://console.groq.com")
        print()

    app.run(host="0.0.0.0", port=PORT, debug=False, use_reloader=False)
