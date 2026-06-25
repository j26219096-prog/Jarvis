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
from zoneinfo import ZoneInfo

IST = ZoneInfo("Asia/Kolkata")
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
        now   = datetime.datetime.now(IST).strftime("%I:%M %p")
        reply = f"The current time is {now} IST."
        return {"reply": reply, "action": None}

    # Date
    if "date" in command or "today" in command:
        today = datetime.datetime.now(IST).strftime("%A, %B %d %Y")
        return {"reply": f"Today is {today}.", "action": None}

    # Weather (free - no API key needed)
    if any(kw in command for kw in ["weather", "temperature", "forecast", "rain", "sunny", "how hot", "how cold"]):
        return get_weather()

    # Google Maps
    if any(kw in command for kw in ["open maps", "maps", "navigate", "directions", "google maps"]):
        query = command
        for f in ["open maps", "navigate to", "navigate", "directions to", "google maps", "maps"]:
            query = query.replace(f, "")
        query = query.strip()
        url = f"https://maps.google.com/maps?q={query.replace(' ', '+')}" if query else "https://maps.google.com"
        return {"reply": f"Opening maps{f' for {query}' if query else ''}.", "action": "open_url", "url": url, "url_label": "GOOGLE MAPS"}

    # WhatsApp
    if any(kw in command for kw in ["whatsapp", "open whatsapp"]):
        return {"reply": "Opening WhatsApp.", "action": "open_url", "url": "https://wa.me", "url_label": "WHATSAPP"}

    # Spotify / Music
    if any(kw in command for kw in ["spotify", "play music", "open spotify", "music"]):
        return {"reply": "Opening Spotify.", "action": "open_url", "url": "https://open.spotify.com", "url_label": "SPOTIFY"}

    # Instagram
    if any(kw in command for kw in ["instagram", "open instagram"]):
        return {"reply": "Opening Instagram.", "action": "open_url", "url": "https://www.instagram.com", "url_label": "INSTAGRAM"}

    # Netflix
    if any(kw in command for kw in ["netflix", "open netflix"]):
        return {"reply": "Opening Netflix.", "action": "open_url", "url": "https://www.netflix.com", "url_label": "NETFLIX"}

    # Twitter / X
    if any(kw in command for kw in ["twitter", "open twitter", "open x"]):
        return {"reply": "Opening X.", "action": "open_url", "url": "https://x.com", "url_label": "X"}

    # LinkedIn
    if any(kw in command for kw in ["linkedin", "open linkedin"]):
        return {"reply": "Opening LinkedIn.", "action": "open_url", "url": "https://www.linkedin.com", "url_label": "LINKEDIN"}

    # Calculator
    if any(kw in command for kw in ["calculator", "calculate", "compute"]):
        return {"reply": "Opening calculator.", "action": "open_url", "url": "https://www.google.com/search?q=calculator", "url_label": "CALCULATOR"}

    # GitHub
    if any(kw in command for kw in ["github", "open github"]):
        return {"reply": "Opening GitHub.", "action": "open_url", "url": "https://github.com", "url_label": "GITHUB"}

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


def get_weather() -> dict:
    """Fetch live weather using wttr.in — no API key required."""
    try:
        resp = http_requests.get("https://wttr.in/?format=%C+%t,+humidity+%h", timeout=8)
        if resp.status_code == 200:
            info = resp.text.strip()
            return {"reply": f"Current weather: {info}. Shall I open the full forecast?", "action": None}
        return {"reply": "Weather service is unavailable right now, sir.", "action": None}
    except Exception:
        return {"reply": "Cannot reach weather service at the moment.", "action": None}


# ══════════════════════════════════════════════════════════════════════════════
# HTML — Iron Man HUD • Voice-only • Mobile-first
# ══════════════════════════════════════════════════════════════════════════════

HTML_PAGE = """
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8"/>
  <meta name="viewport" content="width=device-width, initial-scale=1.0, maximum-scale=1.0, user-scalable=no, viewport-fit=cover"/>
  <meta name="apple-mobile-web-app-capable" content="yes"/>
  <meta name="apple-mobile-web-app-status-bar-style" content="black-translucent"/>
  <title>J.A.R.V.I.S.</title>
  <meta name="description" content="J.A.R.V.I.S. Cloud AI — Voice Assistant"/>
  <meta name="theme-color" content="#000814"/>
  <link rel="manifest" href="/manifest.json"/>
  <link href="https://fonts.googleapis.com/css2?family=Orbitron:wght@400;600;700;900&family=Rajdhani:wght@300;400;600;700&family=Share+Tech+Mono&display=swap" rel="stylesheet"/>
  <style>
    :root {
      --bg:    #000814;
      --blue:  #00d4ff;
      --blue2: #0055ff;
      --green: #00ff88;
      --red:   #ff2255;
      --amber: #ffaa00;
      --text:  #90c8e8;
      --muted: #1e3a5a;
      --glow:  rgba(0,212,255,0.35);
    }
    *,*::before,*::after{box-sizing:border-box;margin:0;padding:0;}
    html{height:100%;background:var(--bg);}
    body{
      height:100dvh; background:var(--bg); color:var(--text);
      font-family:'Rajdhani',sans-serif;
      display:flex; flex-direction:column; overflow:hidden;
      -webkit-tap-highlight-color:transparent; user-select:none;
    }

    /* Background layers */
    .bg-glow{
      position:fixed;inset:0;pointer-events:none;z-index:0;
      background:
        radial-gradient(ellipse at 50% 0%,rgba(0,80,200,0.13) 0%,transparent 60%),
        radial-gradient(ellipse at 50% 100%,rgba(0,30,100,0.1) 0%,transparent 55%);
    }
    .hex-grid{
      position:fixed;inset:0;pointer-events:none;z-index:0;opacity:0.55;
      background-image:url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='56' height='48'%3E%3Cpath d='M28 2 L54 16 L54 32 L28 46 L2 32 L2 16 Z' fill='none' stroke='%23002255' stroke-width='0.8'/%3E%3C/svg%3E");
      background-size:56px 48px;
    }
    .scanlines{
      position:fixed;inset:0;pointer-events:none;z-index:0;
      background:repeating-linear-gradient(0deg,transparent,transparent 3px,rgba(0,0,0,0.04) 3px,rgba(0,0,0,0.04) 4px);
    }

    /* HUD Corners */
    .hc{position:fixed;width:54px;height:54px;z-index:5;pointer-events:none;}
    .hc::before,.hc::after{content:'';position:absolute;background:rgba(0,212,255,0.55);}
    .hc.tl{top:0;left:0;}
    .hc.tl::before{top:0;left:0;width:54px;height:2px;}
    .hc.tl::after{top:0;left:0;width:2px;height:54px;}
    .hc.tr{top:0;right:0;}
    .hc.tr::before{top:0;right:0;width:54px;height:2px;}
    .hc.tr::after{top:0;right:0;width:2px;height:54px;}
    .hc.bl{bottom:0;left:0;}
    .hc.bl::before{bottom:0;left:0;width:54px;height:2px;}
    .hc.bl::after{bottom:0;left:0;width:2px;height:54px;}
    .hc.br{bottom:0;right:0;}
    .hc.br::before{bottom:0;right:0;width:54px;height:2px;}
    .hc.br::after{bottom:0;right:0;width:2px;height:54px;}

    /* HEADER */
    .hud-header{
      position:relative;z-index:10;flex-shrink:0;
      padding:14px 20px 10px;
      background:linear-gradient(180deg,rgba(0,8,30,0.96) 0%,rgba(0,8,20,0.5) 100%);
      border-bottom:1px solid rgba(0,212,255,0.12);
    }
    .hud-row1{display:flex;align-items:center;justify-content:space-between;}
    .j-name{
      font-family:'Orbitron',monospace;font-size:1.05rem;font-weight:900;
      letter-spacing:5px;color:var(--blue);
      text-shadow:0 0 18px var(--blue),0 0 36px rgba(0,212,255,0.3);
    }
    .sys-badge{display:flex;align-items:center;gap:6px;
      font-family:'Share Tech Mono',monospace;font-size:0.62rem;letter-spacing:2px;}
    .led{
      width:8px;height:8px;border-radius:50%;
      background:var(--green);box-shadow:0 0 8px var(--green);
      animation:led-blink 2s infinite;
    }
    .led.red  {background:var(--red);  box-shadow:0 0 8px var(--red);  animation:led-blink 0.5s infinite;}
    .led.blue {background:var(--blue); box-shadow:0 0 8px var(--blue); animation:led-blink 0.35s infinite;}
    @keyframes led-blink{0%,100%{opacity:1;}50%{opacity:0.15;}}
    .sys-lbl{color:var(--green);}
    .sys-lbl.red{color:var(--red);}
    .sys-lbl.blue{color:var(--blue);}
    .hud-meta{
      margin-top:6px;display:flex;gap:14px;
      font-family:'Share Tech Mono',monospace;font-size:0.58rem;
      color:var(--muted);letter-spacing:0.5px;
    }
    .hud-meta .v{color:rgba(0,212,255,0.45);}

    /* RESPONSE PANEL */
    .resp-area{
      flex:1;position:relative;z-index:10;
      display:flex;align-items:center;justify-content:center;
      padding:14px 16px;overflow:hidden;
    }
    .holo-panel{
      width:100%;max-width:460px;
      background:rgba(0,15,50,0.6);
      border:1px solid rgba(0,212,255,0.18);
      border-radius:14px;padding:18px 20px 16px;
      backdrop-filter:blur(14px);-webkit-backdrop-filter:blur(14px);
      position:relative;
    }
    .holo-panel::before{
      content:'';position:absolute;top:-1px;left:22px;
      width:48px;height:2px;
      background:linear-gradient(90deg,var(--blue),transparent);
      box-shadow:0 0 8px var(--blue);
    }
    .holo-panel::after{
      content:'';position:absolute;bottom:-1px;right:22px;
      width:48px;height:2px;
      background:linear-gradient(270deg,var(--blue),transparent);
      box-shadow:0 0 8px var(--blue);
    }
    .panel-lbl{
      font-family:'Orbitron',monospace;font-size:0.52rem;
      letter-spacing:3px;color:rgba(0,212,255,0.45);margin-bottom:10px;
    }
    .query-line{
      font-family:'Share Tech Mono',monospace;font-size:0.72rem;
      color:var(--muted);margin-bottom:9px;padding-bottom:8px;
      border-bottom:1px solid rgba(0,212,255,0.08);display:none;
    }
    .query-line.on{display:block;}
    .query-line::before{content:'> ';color:rgba(0,212,255,0.35);}
    .resp-txt{font-size:1.1rem;line-height:1.65;color:var(--text);min-height:56px;}
    .resp-txt.blink::after{content:'|';animation:cur-blink 0.6s infinite;color:var(--blue);margin-left:1px;}
    @keyframes cur-blink{0%,100%{opacity:1;}50%{opacity:0;}}
    .tdots{display:none;gap:5px;margin-top:8px;}
    .tdots.on{display:flex;}
    .tdots b{
      width:6px;height:6px;border-radius:50%;
      background:var(--blue);animation:td 1.2s infinite;
    }
    .tdots b:nth-child(2){animation-delay:.18s;}
    .tdots b:nth-child(3){animation-delay:.36s;}
    @keyframes td{0%,60%,100%{transform:translateY(0);opacity:.3;}30%{transform:translateY(-7px);opacity:1;}}

    /* QUICK COMMANDS */
    .quick-wrap{position:relative;z-index:10;padding:6px 14px;flex-shrink:0;}
    .quick-inner{display:flex;gap:7px;overflow-x:auto;scrollbar-width:none;}
    .quick-inner::-webkit-scrollbar{display:none;}
    .qc{
      flex-shrink:0;
      font-family:'Rajdhani',sans-serif;font-size:0.72rem;font-weight:600;letter-spacing:0.5px;
      padding:7px 12px;border-radius:6px;
      border:1px solid var(--muted);
      background:rgba(0,15,45,0.55);
      color:rgba(144,200,232,0.7);
      cursor:pointer;white-space:nowrap;
      transition:border-color 0.15s,color 0.15s,box-shadow 0.15s;
    }
    .qc:active{border-color:var(--blue);color:var(--blue);box-shadow:0 0 10px rgba(0,212,255,0.25);}

    /* MIC / ARC REACTOR */
    .mic-section{
      position:relative;z-index:10;flex-shrink:0;
      display:flex;flex-direction:column;align-items:center;
      padding:10px 0;
      padding-bottom:max(env(safe-area-inset-bottom,16px),16px);
      gap:10px;
    }
    .wave{display:flex;align-items:center;gap:3px;height:28px;opacity:0;transition:opacity 0.3s;}
    .wave.on{opacity:1;}
    .wave i{display:block;width:3px;border-radius:2px;background:var(--red);box-shadow:0 0 5px var(--red);animation:wbar 0.9s ease-in-out infinite;}
    .wave i:nth-child(1){height:8px;animation-delay:0s;}
    .wave i:nth-child(2){height:18px;animation-delay:.08s;}
    .wave i:nth-child(3){height:26px;animation-delay:.16s;}
    .wave i:nth-child(4){height:14px;animation-delay:.24s;}
    .wave i:nth-child(5){height:22px;animation-delay:.12s;}
    .wave i:nth-child(6){height:16px;animation-delay:.08s;}
    .wave i:nth-child(7){height:10px;animation-delay:0s;}
    @keyframes wbar{0%,100%{transform:scaleY(.3);}50%{transform:scaleY(1);}}

    .arc-wrap{
      position:relative;width:148px;height:148px;
      display:flex;align-items:center;justify-content:center;
    }
    .ring{position:absolute;border-radius:50%;border:1px solid rgba(0,212,255,0.22);}
    .r1{width:148px;height:148px;animation:rbreath 3s ease-in-out infinite;}
    .r2{width:126px;height:126px;animation:rbreath 3s ease-in-out infinite 0.5s;border-color:rgba(0,212,255,0.13);}
    @keyframes rbreath{0%,100%{transform:scale(1);opacity:.5;}50%{transform:scale(1.05);opacity:1;}}

    /* Listening state */
    .arc-wrap.lst .r1{animation:rlisten 0.65s infinite;border-color:rgba(255,34,85,.75);}
    .arc-wrap.lst .r2{animation:rlisten 0.65s infinite .15s;border-color:rgba(255,34,85,.4);}
    @keyframes rlisten{0%,100%{transform:scale(1);}50%{transform:scale(1.11);}}

    /* Thinking state */
    .arc-wrap.thk .r1{animation:rspin 1.8s linear infinite;border-color:transparent;border-top-color:var(--blue);box-shadow:0 0 12px rgba(0,212,255,0.3);}
    .arc-wrap.thk .r2{animation:rspin 3s linear infinite reverse;border-color:transparent;border-top-color:var(--blue2);}
    @keyframes rspin{from{transform:rotate(0);}to{transform:rotate(360deg);}}

    /* Arc reactor button */
    .arc-btn{
      width:100px;height:100px;border-radius:50%;
      background:radial-gradient(circle at 38% 38%,rgba(0,55,130,0.9),rgba(0,4,18,0.97));
      border:2px solid rgba(0,212,255,0.45);
      box-shadow:0 0 24px rgba(0,212,255,0.2),0 0 50px rgba(0,212,255,0.07),inset 0 0 24px rgba(0,212,255,0.07);
      cursor:pointer;position:relative;
      display:flex;align-items:center;justify-content:center;
      transition:transform 0.15s,box-shadow 0.2s;
      -webkit-appearance:none;outline:none;
    }
    .arc-btn:active{transform:scale(0.93);}
    .arc-btn::before{
      content:'';position:absolute;width:52px;height:52px;
      border-radius:50%;border:1.5px solid rgba(0,212,255,0.3);
    }
    .arc-core{
      width:22px;height:22px;border-radius:50%;
      background:radial-gradient(circle,#9aeeff,var(--blue));
      box-shadow:0 0 18px var(--blue),0 0 36px rgba(0,212,255,0.35);
      animation:core-glow 3s ease-in-out infinite;position:relative;z-index:1;
    }
    @keyframes core-glow{0%,100%{box-shadow:0 0 18px var(--blue),0 0 36px rgba(0,212,255,0.35);}50%{box-shadow:0 0 28px var(--blue),0 0 56px rgba(0,212,255,0.55);}}

    .arc-wrap.lst .arc-btn{
      border-color:rgba(255,34,85,.7);
      box-shadow:0 0 28px rgba(255,34,85,.35),inset 0 0 24px rgba(255,34,85,.08);
    }
    .arc-wrap.lst .arc-btn::before{border-color:rgba(255,34,85,.45);}
    .arc-wrap.lst .arc-core{
      background:radial-gradient(circle,#ffaacc,var(--red));
      box-shadow:0 0 18px var(--red),0 0 36px rgba(255,34,85,.35);animation:none;
    }
    .arc-wrap.thk .arc-btn{
      box-shadow:0 0 36px rgba(0,212,255,.5),0 0 70px rgba(0,212,255,.2),inset 0 0 24px rgba(0,212,255,.12);
    }
    .arc-wrap.thk .arc-core{animation:core-spin 1s linear infinite;}
    @keyframes core-spin{to{filter:hue-rotate(360deg);}}

    .mic-lbl{
      font-family:'Orbitron',monospace;font-size:0.58rem;
      letter-spacing:3px;color:var(--muted);text-align:center;transition:color 0.3s;
    }
    .mic-lbl.red{color:var(--red);}
    .mic-lbl.blue{color:var(--blue);}
    .mic-lbl.green{color:var(--green);}

    /* TOAST */
    .toast{
      position:fixed;top:78px;left:50%;
      transform:translateX(-50%) translateY(-10px);
      background:rgba(0,15,50,0.96);border:1px solid rgba(0,212,255,0.3);
      border-radius:8px;padding:8px 18px;
      font-family:'Share Tech Mono',monospace;font-size:0.68rem;
      color:var(--blue);letter-spacing:1px;
      z-index:200;opacity:0;transition:all 0.25s;
      pointer-events:none;white-space:nowrap;
    }
    .toast.on{opacity:1;transform:translateX(-50%) translateY(0);}

    /* NO VOICE */
    #noVoice{
      display:none;position:fixed;inset:0;z-index:300;
      background:rgba(0,0,0,0.93);
      flex-direction:column;align-items:center;justify-content:center;
      gap:16px;padding:32px;text-align:center;
    }
    #noVoice.on{display:flex;}
    #noVoice h2{font-family:'Orbitron',monospace;font-size:1rem;letter-spacing:2px;color:var(--red);}
    #noVoice p{color:var(--muted);font-size:0.9rem;line-height:1.6;}

    @media(min-height:700px){
      .arc-wrap{width:164px;height:164px;}
      .r1{width:164px;height:164px;}
      .r2{width:140px;height:140px;}
      .arc-btn{width:112px;height:112px;}
    }
  </style>
</head>
<body>

<div class="bg-glow"></div>
<div class="hex-grid"></div>
<div class="scanlines"></div>
<div class="hc tl"></div><div class="hc tr"></div>
<div class="hc bl"></div><div class="hc br"></div>

<header class="hud-header">
  <div class="hud-row1">
    <div class="j-name">J.A.R.V.I.S.</div>
    <div class="sys-badge">
      <div class="led" id="led"></div>
      <span class="sys-lbl" id="sysLbl">ONLINE</span>
    </div>
  </div>
  <div class="hud-meta">
    <span>SYS&nbsp;<span class="v">NOMINAL</span></span>
    <span>AI&nbsp;<span class="v">GROQ&nbsp;LLM</span></span>
    <span>MEM&nbsp;<span class="v">ACTIVE</span></span>
    <span>NET&nbsp;<span class="v">CLOUD</span></span>
  </div>
</header>

<div class="resp-area">
  <div class="holo-panel">
    <div class="panel-lbl">&#9672; JARVIS OUTPUT</div>
    <div class="query-line" id="qLine"></div>
    <div class="resp-txt" id="rTxt">Initialising systems&hellip;</div>
    <div class="tdots" id="tDots"><b></b><b></b><b></b></div>
  </div>
</div>

<div class="quick-wrap">
  <div class="quick-inner">
    <button class="qc" id="qc1" onclick="runCmd('world news')">&#127758; NEWS</button>
    <button class="qc" id="qc2" onclick="runCmd('weather today')">&#127748; WEATHER</button>
    <button class="qc" id="qc3" onclick="runCmd('open youtube')">&#9654; YOUTUBE</button>
    <button class="qc" id="qc4" onclick="runCmd('open maps')">&#128506; MAPS</button>
    <button class="qc" id="qc5" onclick="runCmd('open whatsapp')">&#128172; WHATSAPP</button>
    <button class="qc" id="qc6" onclick="runCmd('open spotify')">&#127925; SPOTIFY</button>
    <button class="qc" id="qc7" onclick="runCmd('motivate me')">&#9889; MOTIVATE</button>
    <button class="qc" id="qc8" onclick="runCmd('tell me a joke')">&#128516; JOKE</button>
    <button class="qc" id="qc9" onclick="runCmd('what time is it')">&#128336; TIME</button>
    <button class="qc" id="qc10" onclick="runCmd('open instagram')">&#128247; INSTAGRAM</button>
    <button class="qc" id="qc11" onclick="runCmd('open netflix')">&#127909; NETFLIX</button>
    <button class="qc" id="qc12" onclick="runCmd('open calculator')">&#129518; CALC</button>
  </div>
</div>

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

<div class="toast" id="toast"></div>

<div id="noVoice">
  <h2>&#9888; VOICE NOT SUPPORTED</h2>
  <p>Please use <strong>Chrome on Android</strong><br>or <strong>Safari on iOS</strong> for voice input.</p>
</div>

<script>
  'use strict';
  const arcWrap = document.getElementById('arcWrap');
  const micLbl  = document.getElementById('micLbl');
  const led     = document.getElementById('led');
  const sysLbl  = document.getElementById('sysLbl');
  const rTxt    = document.getElementById('rTxt');
  const qLine   = document.getElementById('qLine');
  const tDots   = document.getElementById('tDots');
  const waveEl  = document.getElementById('wave');
  const toastEl = document.getElementById('toast');

  const synth = window.speechSynthesis;
  let voices  = [];
  function loadVoices(){ voices = synth.getVoices(); }
  if (synth.onvoiceschanged !== undefined) synth.onvoiceschanged = loadVoices;
  loadVoices();

  function speak(text){
    if(!synth) return;
    synth.cancel();
    const u = new SpeechSynthesisUtterance(text);
    u.rate=0.92; u.pitch=0.85; u.volume=1;
    const pick = voices.find(v=>v.lang.startsWith('en')&&/daniel|alex|mark|google uk english male/i.test(v.name))
              || voices.find(v=>v.lang.startsWith('en-')&&/male/i.test(v.name))
              || voices.find(v=>v.lang.startsWith('en'))
              || null;
    if(pick) u.voice=pick;
    synth.speak(u);
  }

  let appState='idle';
  function setState(s){
    appState=s;
    arcWrap.className='arc-wrap'+(s==='listening'?' lst':s==='thinking'?' thk':'');
    waveEl.className='wave'+(s==='listening'?' on':'');
    tDots.className='tdots'+(s==='thinking'?' on':'');
    if(s==='idle'){
      micLbl.textContent='TAP TO SPEAK'; micLbl.className='mic-lbl';
      led.className='led'; sysLbl.textContent='ONLINE'; sysLbl.className='sys-lbl';
    } else if(s==='listening'){
      micLbl.textContent='LISTENING...'; micLbl.className='mic-lbl red';
      led.className='led red'; sysLbl.textContent='LISTENING'; sysLbl.className='sys-lbl red';
    } else if(s==='thinking'){
      micLbl.textContent='PROCESSING...'; micLbl.className='mic-lbl blue';
      led.className='led blue'; sysLbl.textContent='THINKING'; sysLbl.className='sys-lbl blue';
    } else if(s==='speaking'){
      micLbl.textContent='RESPONDING...'; micLbl.className='mic-lbl green';
      led.className='led'; sysLbl.textContent='ONLINE'; sysLbl.className='sys-lbl';
    }
  }

  let twTimer=null;
  function typewrite(text){
    if(twTimer) clearInterval(twTimer);
    rTxt.className='resp-txt blink'; rTxt.textContent='';
    let i=0;
    twTimer=setInterval(()=>{
      if(i<text.length){
        rTxt.textContent+=text[i++];
      } else {
        rTxt.className='resp-txt';
        clearInterval(twTimer); twTimer=null;
        setTimeout(()=>setState('idle'),2200);
      }
    },14);
  }

  let toastTimer=null;
  function showToast(msg){
    toastEl.textContent=msg;
    toastEl.classList.add('on');
    if(toastTimer) clearTimeout(toastTimer);
    toastTimer=setTimeout(()=>toastEl.classList.remove('on'),3000);
  }

  async function runCmd(text){
    if(!text||appState==='thinking') return;
    synth.cancel();
    setState('thinking');
    qLine.textContent=text; qLine.className='query-line on';
    rTxt.textContent=''; rTxt.className='resp-txt';
    try{
      const res  = await fetch('/command',{
        method:'POST',
        headers:{'Content-Type':'application/json'},
        body:JSON.stringify({command:text})
      });
      const data = await res.json();
      const reply= data.reply||'No response received.';
      setState('speaking');
      typewrite(reply);
      speak(reply);
      if(data.action==='open_url'&&data.url){
        const lbl=data.url_label||'LINK';
        setTimeout(()=>{ showToast('>> OPENING '+lbl); window.open(data.url,'_blank'); },950);
      }
    } catch(e){
      setState('idle');
      typewrite('Neural link disrupted. Check connection, sir.');
    }
  }

  const SR=window.SpeechRecognition||window.webkitSpeechRecognition;
  let rec=null, micActive=false;
  if(!SR){
    document.getElementById('noVoice').classList.add('on');
  } else {
    rec=new SR();
    rec.lang='en-IN';
    rec.interimResults=false;
    rec.maxAlternatives=1;
    rec.continuous=false;
    rec.onresult=(e)=>{ micActive=false; runCmd(e.results[0][0].transcript); };
    rec.onerror=(e)=>{
      micActive=false; setState('idle');
      if(e.error==='no-speech') showToast('NO SPEECH DETECTED');
      else if(e.error==='not-allowed') showToast('MIC ACCESS DENIED');
      else showToast('MIC ERROR: '+e.error.toUpperCase());
    };
    rec.onend=()=>{ if(micActive){ micActive=false; setState('idle'); } };
  }

  function toggleMic(){
    if(!rec) return;
    if(micActive){ rec.stop(); micActive=false; setState('idle'); }
    else {
      if(appState==='thinking') return;
      synth.cancel();
      try{ rec.start(); micActive=true; setState('listening'); }
      catch(e){ micActive=false; setState('idle'); showToast('MIC UNAVAILABLE'); }
    }
  }

  (async()=>{
    try{
      const res  = await fetch('/greet');
      const data = await res.json();
      setState('speaking');
      typewrite(data.greeting);
      speak(data.greeting);
    } catch(e){
      typewrite('J.A.R.V.I.S. online. Tap the reactor to speak.');
      setState('idle');
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
    hour = datetime.datetime.now(IST).hour
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
