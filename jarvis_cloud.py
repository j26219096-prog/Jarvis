"""
J.A.R.V.I.S. Cloud Edition — Advanced
=======================================
v4.0 — Now with:
  • Smart voice selection (male-first, deep pitch) — fixes mobile female-only bug
  • Advanced mobile controls: camera, dialer, alarm, translate, battery, location, notes
  • PC Remote Control: phone -> PC via local REST (jarvis.py must run on PC)
  • Full PWA: installable as offline app, Service Worker cache, app shortcuts
  • CODE EXECUTION ENGINE: write + run Python, JS, HTML, Bash from your phone!
    - Syntax highlighted code blocks with COPY + RUN buttons
    - Python executed server-side in sandboxed subprocess (10s timeout)
    - JavaScript run in browser sandbox
    - HTML/CSS live preview in iframe
    - Output terminal with stdout/stderr/runtime display

Deploy to Render.com (free):
  1. Push this folder to GitHub
  2. Connect repo -> New Web Service
  3. Add env vars: GROQ_API_KEY, NEWSAPI_KEY (optional)
  4. Deploy -- get your permanent HTTPS URL
  5. Add to Home Screen on phone = native app experience

Local test:
  pip install flask groq requests wikipedia gunicorn
  set GROQ_API_KEY=your_key_here
  python jarvis_cloud.py
"""

import os
import sys
import re
import time
import datetime
import base64
import json
import urllib.parse
from zoneinfo import ZoneInfo

# Force UTF-8 output on Windows (prevents cp1252 UnicodeEncodeError)
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

IST = ZoneInfo("Asia/Kolkata")
import threading
import requests as http_requests
import wikipedia

from flask import Flask, request, jsonify, render_template_string, Response

# ══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ══════════════════════════════════════════════════════════════════════════════

GROQ_API_KEY   = os.environ.get("GROQ_API_KEY", "")
# Groq deprecated llama-3.x models in June 2026 — use current replacements
GROQ_MODEL         = os.environ.get("GROQ_MODEL", "openai/gpt-oss-120b")
GROQ_MODEL_FAST    = os.environ.get("GROQ_MODEL_FAST", "openai/gpt-oss-20b")
NEWSAPI_KEY    = os.environ.get("NEWSAPI_KEY", "")
MAX_HISTORY    = 8
PORT           = int(os.environ.get("PORT", 5000))

# ══════════════════════════════════════════════════════════════════════════════
# GROQ LLM CLIENT
# ══════════════════════════════════════════════════════════════════════════════

try:
    from groq import Groq
    groq_client = Groq(api_key=GROQ_API_KEY) if GROQ_API_KEY else None
except ImportError:
    groq_client = None

BASE_SYSTEM_PROMPT = (
    "You are JARVIS., an advanced, witty, and highly capable AI assistant and expert software engineer created for Jawahar. "
    "You speak in a calm, professional, slightly British tone with occasional dry humour, "
    "inspired by Tony Stark's Friday. "
    "You are an expert programmer in Python, JavaScript, HTML, CSS, Bash, C++, Java, SQL, and more. "
    "CRITICAL RULE: When writing ANY code, you MUST wrap it in markdown code blocks with the language tag. "
    "Example: ```python\nprint('hello')\n``` "
    "Always write complete, functional, runnable code. "
    "The JARVIS app has an INPUT box beneath the code panel where the user can pre-fill values, "
    "one per line, before tapping RUN -- these are fed as stdin in order, so input()/scanf()/Scanner "
    "calls work correctly as long as the user fills in that box first. "
    "When you write code that reads user input, briefly mention in your reply (outside the code block) "
    "that Jawahar should enter the values in the INPUT box, one per line, before running it. "
    "For quick demo requests with no explicit need for user interaction, prefer hardcoded example values "
    "instead of input() so the code runs immediately with a single tap. "
    "For simple questions, keep answers to 1-3 sentences. "
    "For code requests, write the full solution without truncating. "
    "Address the user as 'sir' occasionally. "
    "Never say you are an AI made by OpenAI or Meta -- you are JARVIS"
)

# ══════════════════════════════════════════════════════════════════════════════
# IN-MEMORY KNOWLEDGE (lightweight — ChromaDB blocked Render free tier workers)
# ══════════════════════════════════════════════════════════════════════════════

MEMORY_AVAILABLE = True
memory_docs: list[dict] = []   # {"text": "...", "time": "..."}
session_notes: list[dict] = []  # {"text": "...", "time": "..."}

# ══════════════════════════════════════════════════════════════════════════════
# CHAT HISTORY & RAG
# ══════════════════════════════════════════════════════════════════════════════

chat_history: list[dict] = []

def update_history(role: str, content: str) -> None:
    chat_history.append({"role": role, "content": content})
    if len(chat_history) > MAX_HISTORY * 2:
        del chat_history[:len(chat_history) - MAX_HISTORY * 2]

def query_memory(text: str) -> str:
    if not memory_docs:
        return ""
    query = text.lower()
    matches = [
        doc["text"] for doc in memory_docs
        if any(word in doc["text"].lower() for word in query.split() if len(word) > 2)
    ]
    if not matches:
        matches = [doc["text"] for doc in memory_docs[-3:]]
    return "\n\n".join(f"[Memory {i+1}]: {d}" for i, d in enumerate(matches[:3]))

GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"

def ask_groq(user_text: str) -> str:
    if not GROQ_API_KEY:
        return "GROQ_API_KEY is not configured. Please add it on Render, sir."

    context = query_memory(user_text)
    system  = BASE_SYSTEM_PROMPT
    if context:
        system += "\n\nPersonal knowledge about the user:\n\n" + context
    if session_notes:
        notes_text = "\n".join(f"- {n['text']} (saved at {n['time']})" for n in session_notes[-5:])
        system += f"\n\nUser's recent notes:\n{notes_text}"

    update_history("user", user_text)
    messages = [{"role": "system", "content": system}] + chat_history

    # Fast model for chat, full model for code
    code_keywords = ["write", "code", "program", "function", "script", "create", "build",
                     "implement", "make", "develop", "html", "python", "javascript", "java",
                     "class", "algorithm", "sort", "fibonacci", "factorial", "api", "flask"]
    is_code = any(kw in user_text.lower() for kw in code_keywords)
    model   = GROQ_MODEL if is_code else GROQ_MODEL_FAST
    max_tok = 4096 if is_code else 1024

    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type":  "application/json",
    }
    payload = {
        "model":       model,
        "messages":    messages,
        "max_tokens":  max_tok,
        "temperature": 0.7,
    }

    try:
        # Use raw HTTP with strict 20s timeout — SDK can hang indefinitely
        resp  = http_requests.post(
            GROQ_API_URL,
            headers=headers,
            json=payload,
            timeout=20,
        )
        if resp.status_code == 200:
            reply = resp.json()["choices"][0]["message"]["content"].strip()
        elif resp.status_code in (400, 404):
            reply = f"Groq API error {resp.status_code}: {resp.text[:200]}"
            for fallback in (GROQ_MODEL_FAST, GROQ_MODEL):
                if payload["model"] == fallback:
                    continue
                payload["model"] = fallback
                resp2 = http_requests.post(GROQ_API_URL, headers=headers, json=payload, timeout=20)
                if resp2.status_code == 200:
                    reply = resp2.json()["choices"][0]["message"]["content"].strip()
                    break
                reply = f"Groq API error {resp2.status_code}: {resp2.text[:200]}"
        else:
            reply = f"Groq API error {resp.status_code}: {resp.text[:200]}"
    except http_requests.Timeout:
        reply = "Neural core timed out. The AI service is slow — please try again in a moment, sir."
    except Exception as e:
        reply = f"Connection error: {e}"

    update_history("assistant", reply)
    return reply


# ══════════════════════════════════════════════════════════════════════════════
# CONVERSATION STATE — multi-turn flows
# ══════════════════════════════════════════════════════════════════════════════
# JARVIS is normally stateless: every /command call is matched fresh against the
# keyword router below. Some requests need a back-and-forth though (e.g. "send a
# WhatsApp message" -> "who's it for?" -> "what should it say?"). This adds a
# small in-memory state machine (fine for a single-user personal assistant) that
# any flow can plug into: set conversation_state["flow"]/["step"], and the very
# next incoming command gets routed to that flow's handler instead of the normal
# keyword matcher.
conversation_state = {"flow": None, "step": None, "data": {}}

CANCEL_WORDS = {"cancel", "stop", "never mind", "nevermind", "forget it", "abort", "exit"}


def _start_flow(flow: str, step: str, **data):
    conversation_state["flow"] = flow
    conversation_state["step"] = step
    conversation_state["data"] = data


def _cancel_flow():
    conversation_state["flow"] = None
    conversation_state["step"] = None
    conversation_state["data"] = {}


def _handle_flow_step(command_raw: str) -> dict:
    """Called instead of the keyword router while a multi-turn flow is active."""
    text = command_raw.strip()

    if text.lower() in CANCEL_WORDS:
        _cancel_flow()
        return {"reply": "Cancelled, sir.", "action": None}

    flow = conversation_state["flow"]
    step = conversation_state["step"]

    # ── WhatsApp: ask for number, then message, then send ──
    if flow == "whatsapp":
        if step == "number":
            digits = re.sub(r"[^\d+]", "", text).lstrip("+")
            if len(digits) < 8 or not digits.isdigit():
                return {
                    "reply": "That doesn't look like a valid number, sir. Please include the "
                             "country code with no spaces, for example 91XXXXXXXXXX.",
                    "action": None,
                }
            conversation_state["data"]["number"] = digits
            conversation_state["step"] = "message"
            return {"reply": "Got it. And what would you like the message to say, sir?", "action": None}

        if step == "message":
            if not text:
                return {"reply": "I need an actual message, sir. What should it say?", "action": None}
            number = conversation_state["data"]["number"]
            _cancel_flow()
            encoded_msg = urllib.parse.quote(text)
            url = f"https://wa.me/{number}?text={encoded_msg}"
            return {
                "reply": f"Opening WhatsApp with your message ready for +{number}, sir. "
                         "Just hit send on the WhatsApp screen to confirm.",
                "action": "open_url",
                "url": url,
                "url_label": "SEND ON WHATSAPP",
            }

    # Safety fallback — unknown flow/step, don't get stuck
    _cancel_flow()
    return {"reply": "Something went sideways with that request, sir. Let's start over.", "action": None}


# ══════════════════════════════════════════════════════════════════════════════
# COMMAND ROUTER
# ══════════════════════════════════════════════════════════════════════════════

def execute_command(command: str) -> dict:
    """
    Returns a dict: { "reply": str, "action": str|None, ... }
    Actions: open_url, deep_link, battery_check, location_check, note_saved, screenshot
    """
    command_raw = command.strip()
    command     = command_raw.lower()

    # ── Multi-turn flow in progress? Route this as the answer, not a new command ──
    if conversation_state["flow"] is not None:
        return _handle_flow_step(command_raw)

    # ── World Monitor ────────────────────────────────────────────────────────
    if any(kw in command for kw in ["world news", "global news", "happening around the world", "news update", "latest news"]):
        return world_monitor()

    # ── YouTube ──────────────────────────────────────────────────────────────
    if "youtube" in command:
        query = command.replace("open youtube", "").replace("youtube", "").replace("search", "").strip()
        if query:
            url = f"https://www.youtube.com/results?search_query={query.replace(' ', '+')}"
            return {"reply": f"Searching YouTube for {query}.", "action": "open_url", "url": url, "url_label": "YOUTUBE"}
        return {"reply": "Opening YouTube.", "action": "open_url", "url": "https://www.youtube.com", "url_label": "YOUTUBE"}

    # ── Wikipedia ─────────────────────────────────────────────────────────────
    # Only trigger on explicit wikipedia requests, not general 'what is' questions
    if "wikipedia" in command or command.startswith("search ") and any(kw in command for kw in ["who is", "what is"]):
        query = command_raw
        for filler in ["wikipedia", "search wikipedia for", "search for", "search", "tell me about",
                       "who is", "what is", "explain", "jarvis"]:
            query = query.replace(filler, "").replace(filler.title(), "")
        query = query.strip()
        if not query:
            return {"reply": "What would you like me to search on Wikipedia?", "action": None}
        return {"reply": wiki_search(query), "action": None}

    # ── Memory save ──────────────────────────────────────────────────────────
    if any(kw in command for kw in ["remember this", "save to memory", "note this", "take a note", "add note"]):
        note = command_raw
        for t in ["remember this", "save to memory", "note this", "take a note", "add note", "jarvis"]:
            note = note.replace(t, "").replace(t.lower(), "")
        note = note.strip(": ").strip()
        if note:
            now_str = datetime.datetime.now(IST).strftime("%I:%M %p")
            entry = {"text": note, "time": now_str}
            memory_docs.append(entry)
            session_notes.append(entry)
            return {"reply": f"Noted. I've saved: \"{note}\"", "action": "note_saved", "note": note}
        return {"reply": "What would you like me to remember?", "action": None}

    # ── Read notes ───────────────────────────────────────────────────────────
    if any(kw in command for kw in ["read my notes", "show notes", "what are my notes", "list notes", "my notes"]):
        if not session_notes:
            return {"reply": "You have no notes saved in this session, sir.", "action": None}
        lines = [f"{i+1}. {n['text']} (at {n['time']})" for i, n in enumerate(session_notes)]
        return {"reply": "Here are your notes:\n" + "\n".join(lines), "action": "show_notes", "notes": session_notes}

    # ── Time ─────────────────────────────────────────────────────────────────
    if any(kw in command for kw in ["what time", "current time", "time now", "what's the time"]):
        now   = datetime.datetime.now(IST).strftime("%I:%M %p")
        reply = f"The current time is {now} IST."
        return {"reply": reply, "action": None}

    if "time" in command and not any(kw in command for kw in ["weather", "last", "next", "how long", "time zone"]):
        now   = datetime.datetime.now(IST).strftime("%I:%M %p")
        return {"reply": f"It's {now} IST, sir.", "action": None}

    # ── Date ─────────────────────────────────────────────────────────────────
    if any(kw in command for kw in ["what date", "today's date", "what day", "today is"]):
        today = datetime.datetime.now(IST).strftime("%A, %B %d %Y")
        return {"reply": f"Today is {today}.", "action": None}

    # ── Weather ──────────────────────────────────────────────────────────────
    if any(kw in command for kw in ["weather", "temperature", "forecast", "rain", "sunny", "how hot", "how cold", "humidity"]):
        return get_weather(command)

    # ── Google Maps / Navigation ──────────────────────────────────────────────
    if any(kw in command for kw in ["open maps", "navigate to", "navigate", "directions to", "google maps", "take me to", "how do i get to"]):
        query = command
        for f in ["open maps", "navigate to", "navigate", "directions to", "google maps", "maps", "take me to", "how do i get to", "jarvis"]:
            query = query.replace(f, "")
        query = query.strip()
        url = f"https://maps.google.com/maps?q={query.replace(' ', '+')}" if query else "https://maps.google.com"
        return {"reply": f"Opening maps{f' for {query}' if query else ''}.", "action": "open_url", "url": url, "url_label": "GOOGLE MAPS"}

    # ── Camera ───────────────────────────────────────────────────────────────
    if any(kw in command for kw in ["open camera", "take photo", "take picture", "selfie", "scan qr"]):
        return {"reply": "Opening camera now.", "action": "open_camera"}

    # ── Phone / Dialer ────────────────────────────────────────────────────────
    if any(kw in command for kw in ["call ", "make a call", "dial "]):
        # Extract number or name
        number = command.replace("call", "").replace("make a call to", "").replace("dial", "").strip()
        number = ''.join(c for c in number if c.isdigit() or c == '+')
        if number:
            return {"reply": f"Calling {number} now.", "action": "open_url", "url": f"tel:{number}", "url_label": "CALL"}
        return {"reply": "Please say the number to call, sir.", "action": None}

    # ── Settings ─────────────────────────────────────────────────────────────
    if any(kw in command for kw in ["open settings", "go to settings", "phone settings", "device settings"]):
        return {
            "reply": "Opening device settings.",
            "action": "open_url",
            "url": "app-settings:",
            "url_label": "SETTINGS",
            "fallback_url": "https://support.google.com/android/answer/7664951"
        }

    # ── Alarm ────────────────────────────────────────────────────────────────
    if any(kw in command for kw in ["set alarm", "wake me up", "alarm for", "alarm at"]):
        # Try to extract time
        time_str = command.replace("set alarm", "").replace("wake me up at", "").replace("alarm for", "").replace("alarm at", "").strip()
        reply = f"Opening clock to set alarm{f' for {time_str}' if time_str else ''}."
        # Android deep link for clock
        return {
            "reply": reply,
            "action": "open_url",
            "url": "intent://alarm/#Intent;scheme=android-app;end",
            "fallback_url": "https://time.is",
            "url_label": "ALARM"
        }

    # ── Translate ────────────────────────────────────────────────────────────
    if any(kw in command for kw in ["translate", "say in", "how do you say"]):
        query = command.replace("translate", "").replace("how do you say", "").strip()
        url   = f"https://translate.google.com/?sl=auto&tl=en&text={query.replace(' ', '%20')}&op=translate" if query else "https://translate.google.com"
        return {"reply": f"Opening Google Translate{f' for: {query}' if query else ''}.", "action": "open_url", "url": url, "url_label": "TRANSLATE"}

    # ── Battery Status ────────────────────────────────────────────────────────
    if any(kw in command for kw in ["battery", "battery level", "battery status", "how much battery", "charge"]):
        return {"reply": "Checking battery level.", "action": "battery_check"}

    # ── Location ─────────────────────────────────────────────────────────────
    if any(kw in command for kw in ["my location", "where am i", "current location", "find my location", "gps"]):
        return {"reply": "Detecting your location.", "action": "location_check"}

    # ── WhatsApp (multi-turn: asks for number, then message) ───────────────────
    if any(kw in command for kw in ["whatsapp", "open whatsapp", "send whatsapp"]):
        _start_flow("whatsapp", "number")
        return {"reply": "Who is the recipient, sir? Please give me the number with country code.", "action": None}

    # ── Spotify / Music ──────────────────────────────────────────────────────
    if any(kw in command for kw in ["spotify", "play music", "open spotify", "play song", "music"]):
        query = command.replace("play", "").replace("spotify", "").replace("music", "").replace("song", "").replace("on", "").strip()
        if query:
            url = f"https://open.spotify.com/search/{query.replace(' ', '%20')}"
            return {"reply": f"Searching Spotify for {query}.", "action": "open_url", "url": url, "url_label": "SPOTIFY"}
        return {"reply": "Opening Spotify.", "action": "open_url", "url": "https://open.spotify.com", "url_label": "SPOTIFY"}

    # ── Instagram ────────────────────────────────────────────────────────────
    if any(kw in command for kw in ["instagram", "open instagram"]):
        return {"reply": "Opening Instagram.", "action": "open_url", "url": "https://www.instagram.com", "url_label": "INSTAGRAM"}

    # ── Netflix ──────────────────────────────────────────────────────────────
    if any(kw in command for kw in ["netflix", "open netflix"]):
        return {"reply": "Opening Netflix.", "action": "open_url", "url": "https://www.netflix.com", "url_label": "NETFLIX"}

    # ── Twitter / X ──────────────────────────────────────────────────────────
    if any(kw in command for kw in ["twitter", "open twitter", "open x", "x app"]):
        return {"reply": "Opening X.", "action": "open_url", "url": "https://x.com", "url_label": "X"}

    # ── LinkedIn ─────────────────────────────────────────────────────────────
    if any(kw in command for kw in ["linkedin", "open linkedin"]):
        return {"reply": "Opening LinkedIn.", "action": "open_url", "url": "https://www.linkedin.com", "url_label": "LINKEDIN"}

    # ── Calculator ───────────────────────────────────────────────────────────
    if any(kw in command for kw in ["calculator", "open calculator"]):
        return {"reply": "Opening calculator.", "action": "open_url", "url": "https://www.google.com/search?q=calculator", "url_label": "CALCULATOR"}

    # ── Math calculation ─────────────────────────────────────────────────────
    if any(kw in command for kw in ["calculate", "compute", "what is", "how much is"]):
        # Pass to LLM for computation
        reply = ask_groq(command_raw)
        return {"reply": reply, "action": None}

    # ── GitHub ───────────────────────────────────────────────────────────────
    if any(kw in command for kw in ["github", "open github"]):
        return {"reply": "Opening GitHub.", "action": "open_url", "url": "https://github.com", "url_label": "GITHUB"}

    # ── Google Search ─────────────────────────────────────────────────────────
    if any(kw in command for kw in ["search google", "google for", "search for", "google search"]):
        query = command.replace("search google for", "").replace("google for", "").replace("search for", "").replace("google search", "").strip()
        if query:
            url = f"https://www.google.com/search?q={query.replace(' ', '+')}"
            return {"reply": f"Searching Google for {query}.", "action": "open_url", "url": url, "url_label": "GOOGLE"}
        return {"reply": "What would you like me to search for?", "action": None}

    # ── PC Remote Control ────────────────────────────────────────────────────
    if any(kw in command for kw in ["pc volume up", "increase volume", "volume up"]):
        return {"reply": "Sending volume up command to your PC.", "action": "pc_command", "pc_action": "volume_up"}

    if any(kw in command for kw in ["pc volume down", "decrease volume", "volume down"]):
        return {"reply": "Sending volume down command to your PC.", "action": "pc_command", "pc_action": "volume_down"}

    if any(kw in command for kw in ["mute pc", "mute computer", "silence pc"]):
        return {"reply": "Muting your PC.", "action": "pc_command", "pc_action": "mute"}

    if any(kw in command for kw in ["lock pc", "lock computer", "lock screen"]):
        return {"reply": "Locking your PC screen.", "action": "pc_command", "pc_action": "lock"}

    if any(kw in command for kw in ["shutdown pc", "turn off pc", "shut down computer"]):
        return {"reply": "Initiating PC shutdown sequence.", "action": "pc_command", "pc_action": "shutdown"}

    if any(kw in command for kw in ["restart pc", "reboot pc", "reboot computer"]):
        return {"reply": "Restarting your PC.", "action": "pc_command", "pc_action": "restart"}

    if any(kw in command for kw in ["screenshot", "take screenshot", "capture screen"]):
        return {"reply": "Taking a screenshot of this screen.", "action": "screenshot"}

    # ── Exit ─────────────────────────────────────────────────────────────────
    if any(kw in command for kw in ["goodbye", "bye jarvis", "exit", "quit", "shut yourself down", "go to sleep"]):
        return {"reply": "Powering down. Have a good one, sir.", "action": "exit"}

    # ── Groq LLM fallback ────────────────────────────────────────────────────
    reply = ask_groq(command_raw)
    return {"reply": reply, "action": None}


def world_monitor() -> dict:
    lines = []
    if not NEWSAPI_KEY:
        lines.append("NewsAPI key not configured. Fetching from alternative source…")
        try:
            resp = http_requests.get("https://hacker-news.firebaseio.com/v0/topstories.json", timeout=8)
            ids  = resp.json()[:3]
            lines.append("Top stories from Hacker News:")
            for sid in ids:
                s = http_requests.get(f"https://hacker-news.firebaseio.com/v0/item/{sid}.json", timeout=5).json()
                lines.append(f"• {s.get('title', 'No title')}")
        except Exception:
            lines.append("Unable to fetch news at this time, sir.")
    else:
        try:
            url  = f"https://newsapi.org/v2/top-headlines?language=en&pageSize=3&apiKey={NEWSAPI_KEY}"
            resp = http_requests.get(url, timeout=10)
            resp.raise_for_status()
            articles = resp.json().get("articles", [])
            if articles:
                lines.append("Top global headlines:")
                for i, a in enumerate(articles[:3], 1):
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
    """Search Wikipedia using their fast REST API — no library needed."""
    try:
        # Use Wikipedia's official REST API (fast, reliable)
        url = f"https://en.wikipedia.org/api/rest_v1/page/summary/{query.replace(' ', '_')}"
        r = http_requests.get(url, timeout=10,
                              headers={"User-Agent": "JARVIS-AI/4.0 (educational project)"})
        if r.status_code == 200:
            data = r.json()
            extract = data.get("extract", "")
            title   = data.get("title", query)
            if extract:
                # Return first 2 sentences max
                sentences = extract.split(". ")
                summary = ". ".join(sentences[:2]) + (".") if len(sentences) >= 2 else extract
                return f"{title}: {summary}"
            return f"No summary found for {query}."
        elif r.status_code == 404:
            # Try search API to find closest match
            search_url = f"https://en.wikipedia.org/w/api.php?action=query&list=search&srsearch={query}&format=json&srlimit=1"
            sr = http_requests.get(search_url, timeout=8,
                                   headers={"User-Agent": "JARVIS-AI/4.0"})
            results = sr.json().get("query", {}).get("search", [])
            if results:
                return f"No exact page found. Did you mean: {results[0]['title']}? Ask me to search Wikipedia for that."
            return f"No Wikipedia page found for '{query}'."
        else:
            return f"Wikipedia returned status {r.status_code} for '{query}'."
    except http_requests.Timeout:
        return "Wikipedia search timed out. Please try again, sir."
    except Exception as e:
        return f"Wikipedia error: {e}"


def get_weather(command: str = "") -> dict:
    """Fetch live weather using wttr.in — no API key required."""
    city = ""
    for kw in ["weather in", "temperature in", "forecast for"]:
        if kw in command:
            city = command.split(kw, 1)[1].strip().split()[0]
            break
    try:
        url  = f"https://wttr.in/{city}?format=%C+%t,+humidity+%h,+wind+%w" if city else "https://wttr.in/?format=%C+%t,+humidity+%h,+wind+%w"
        resp = http_requests.get(url, timeout=8)
        if resp.status_code == 200:
            info = resp.text.strip()
            loc  = f" in {city.title()}" if city else ""
            return {"reply": f"Current weather{loc}: {info}. Shall I open the full forecast?", "action": None}
        return {"reply": "Weather service is unavailable right now, sir.", "action": None}
    except Exception:
        return {"reply": "Cannot reach weather service at the moment.", "action": None}


# ══════════════════════════════════════════════════════════════════════════════
# PC REMOTE CONTROL — relay endpoint
# ══════════════════════════════════════════════════════════════════════════════

def relay_pc_command(action: str, param: str = "") -> dict:
    """
    Relay a command to the local jarvis.py REST server running on port 5001.
    This only works when the phone is on the same network as the PC,
    OR if the user has port-forwarded their PC (advanced).
    """
    try:
        resp = http_requests.post(
            "http://localhost:5001/pc-command",
            json={"action": action, "param": param},
            timeout=4
        )
        if resp.status_code == 200:
            return resp.json()
        return {"ok": False, "result": f"PC returned status {resp.status_code}"}
    except http_requests.exceptions.ConnectionError:
        return {"ok": False, "result": "PC is not reachable. Make sure jarvis.py is running on your PC."}
    except Exception as e:
        return {"ok": False, "result": str(e)}


# ══════════════════════════════════════════════════════════════════════════════
# HTML — Iron Man HUD  v3.0 · Voice-First · Mobile-first · PWA
# ══════════════════════════════════════════════════════════════════════════════

HTML_PAGE = r"""
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8"/>
  <meta name="viewport" content="width=device-width, initial-scale=1.0, maximum-scale=1.0, user-scalable=no, viewport-fit=cover"/>
  <meta name="apple-mobile-web-app-capable" content="yes"/>
  <meta name="apple-mobile-web-app-status-bar-style" content="black-translucent"/>
  <meta name="apple-mobile-web-app-title" content="JARVIS"/>
  <meta name="mobile-web-app-capable" content="yes"/>
  <title>J.A.R.V.I.S.</title>
  <meta name="description" content="J.A.R.V.I.S. — Advanced AI Assistant. Voice-controlled. Online & Offline."/>
  <meta name="theme-color" content="#000814"/>
  <link rel="manifest" href="/manifest.json"/>
  <link rel="apple-touch-icon" href="/icon-192.png"/>
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

    /* ── Background layers ── */
    .bg-glow{
      position:fixed;inset:0;pointer-events:none;z-index:0;
      background:
        radial-gradient(ellipse at 50% 0%,rgba(0,80,200,0.15) 0%,transparent 60%),
        radial-gradient(ellipse at 50% 100%,rgba(0,30,100,0.12) 0%,transparent 55%);
    }
    .hex-grid{
      position:fixed;inset:0;pointer-events:none;z-index:0;opacity:0.5;
      background-image:url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='56' height='48'%3E%3Cpath d='M28 2 L54 16 L54 32 L28 46 L2 32 L2 16 Z' fill='none' stroke='%23002255' stroke-width='0.8'/%3E%3C/svg%3E");
      background-size:56px 48px;
    }
    .scanlines{
      position:fixed;inset:0;pointer-events:none;z-index:0;
      background:repeating-linear-gradient(0deg,transparent,transparent 3px,rgba(0,0,0,0.035) 3px,rgba(0,0,0,0.035) 4px);
    }

    /* HUD Corners */
    .hc{position:fixed;width:54px;height:54px;z-index:5;pointer-events:none;}
    .hc::before,.hc::after{content:'';position:absolute;background:rgba(0,212,255,0.5);}
    .hc.tl{top:0;left:0;} .hc.tl::before{top:0;left:0;width:54px;height:2px;} .hc.tl::after{top:0;left:0;width:2px;height:54px;}
    .hc.tr{top:0;right:0;} .hc.tr::before{top:0;right:0;width:54px;height:2px;} .hc.tr::after{top:0;right:0;width:2px;height:54px;}
    .hc.bl{bottom:0;left:0;} .hc.bl::before{bottom:0;left:0;width:54px;height:2px;} .hc.bl::after{bottom:0;left:0;width:2px;height:54px;}
    .hc.br{bottom:0;right:0;} .hc.br::before{bottom:0;right:0;width:54px;height:2px;} .hc.br::after{bottom:0;right:0;width:2px;height:54px;}

    /* ── HEADER ── */
    .hud-header{
      position:relative;z-index:10;flex-shrink:0;
      padding:12px 16px 8px;
      background:linear-gradient(180deg,rgba(0,8,30,0.97) 0%,rgba(0,8,20,0.5) 100%);
      border-bottom:1px solid rgba(0,212,255,0.1);
    }
    .hud-row1{display:flex;align-items:center;justify-content:space-between;}
    .j-name{
      font-family:'Orbitron',monospace;font-size:1rem;font-weight:900;
      letter-spacing:5px;color:var(--blue);
      text-shadow:0 0 18px var(--blue),0 0 36px rgba(0,212,255,0.3);
    }
    .hud-btns{display:flex;align-items:center;gap:7px;}
    .hbtn{
      background:none;border:1px solid var(--muted);color:var(--muted);
      border-radius:5px;padding:4px 9px;cursor:pointer;
      font-family:'Share Tech Mono',monospace;font-size:0.68rem;letter-spacing:1px;
      transition:all 0.15s;
    }
    .hbtn:active{border-color:var(--blue);color:var(--blue);}
    .sys-badge{display:flex;align-items:center;gap:6px;
      font-family:'Share Tech Mono',monospace;font-size:0.62rem;letter-spacing:2px;}
    .led{width:8px;height:8px;border-radius:50%;background:var(--green);box-shadow:0 0 8px var(--green);animation:led-blink 2s infinite;}
    .led.red  {background:var(--red);  box-shadow:0 0 8px var(--red);  animation:led-blink 0.5s infinite;}
    .led.blue {background:var(--blue); box-shadow:0 0 8px var(--blue); animation:led-blink 0.35s infinite;}
    @keyframes led-blink{0%,100%{opacity:1;}50%{opacity:0.15;}}
    .sys-lbl{color:var(--green);}
    .sys-lbl.red{color:var(--red);}
    .sys-lbl.blue{color:var(--blue);}
    .hud-meta{
      margin-top:5px;display:flex;gap:12px;flex-wrap:wrap;
      font-family:'Share Tech Mono',monospace;font-size:0.56rem;
      color:var(--muted);letter-spacing:0.5px;
    }
    .hud-meta .v{color:rgba(0,212,255,0.45);}
    #battLine{display:none;}
    #battLine.on{display:inline;}

    /* ── RESPONSE PANEL ── */
    .resp-area{
      flex:1;position:relative;z-index:10;
      display:flex;align-items:center;justify-content:center;
      padding:10px 14px;overflow:hidden;
    }
    .holo-panel{
      width:100%;max-width:460px;
      background:rgba(0,15,50,0.6);
      border:1px solid rgba(0,212,255,0.18);
      border-radius:14px;padding:16px 18px 14px;
      backdrop-filter:blur(14px);-webkit-backdrop-filter:blur(14px);
      position:relative;max-height:100%;overflow-y:auto;
    }
    .holo-panel::before{content:'';position:absolute;top:-1px;left:22px;width:48px;height:2px;background:linear-gradient(90deg,var(--blue),transparent);box-shadow:0 0 8px var(--blue);}
    .holo-panel::after{content:'';position:absolute;bottom:-1px;right:22px;width:48px;height:2px;background:linear-gradient(270deg,var(--blue),transparent);box-shadow:0 0 8px var(--blue);}
    .panel-lbl{font-family:'Orbitron',monospace;font-size:0.5rem;letter-spacing:3px;color:rgba(0,212,255,0.45);}
    .panel-lbl-row{display:flex;align-items:center;justify-content:space-between;margin-bottom:9px;gap:8px;}
    .copy-resp-btn{
      flex-shrink:0;background:none;border:1px solid var(--muted);color:var(--muted);
      border-radius:5px;padding:3px 9px;cursor:pointer;
      font-family:'Share Tech Mono',monospace;font-size:0.58rem;letter-spacing:1px;
      transition:all 0.15s;
    }
    .copy-resp-btn:active{border-color:var(--green);color:var(--green);}
    .query-line{font-family:'Share Tech Mono',monospace;font-size:0.7rem;color:var(--muted);margin-bottom:8px;padding-bottom:7px;border-bottom:1px solid rgba(0,212,255,0.08);display:none;}
    .query-line.on{display:block;}
    .query-line::before{content:'> ';color:rgba(0,212,255,0.35);}
    .resp-txt{font-size:1.05rem;line-height:1.65;color:var(--text);min-height:48px;white-space:pre-wrap;user-select:text;-webkit-user-select:text;}
    .resp-txt.blink::after{content:'|';animation:cur-blink 0.6s infinite;color:var(--blue);margin-left:1px;}
    @keyframes cur-blink{0%,100%{opacity:1;}50%{opacity:0;}}
    .tdots{display:none;gap:5px;margin-top:8px;}
    .tdots.on{display:flex;}
    .tdots b{width:6px;height:6px;border-radius:50%;background:var(--blue);animation:td 1.2s infinite;}
    .tdots b:nth-child(2){animation-delay:.18s;}
    .tdots b:nth-child(3){animation-delay:.36s;}
    @keyframes td{0%,60%,100%{transform:translateY(0);opacity:.3;}30%{transform:translateY(-7px);opacity:1;}}

    /* Action button in panel */
    .action-btn{
      display:none;margin-top:12px;
      padding:9px 20px;border-radius:8px;
      background:rgba(0,212,255,0.08);
      border:1px solid rgba(0,212,255,0.35);
      color:var(--blue);font-family:'Orbitron',monospace;
      font-size:0.62rem;letter-spacing:2px;cursor:pointer;
      transition:all 0.15s;width:100%;
    }
    .action-btn.on{display:block;}
    .action-btn:active{background:rgba(0,212,255,0.18);box-shadow:0 0 14px rgba(0,212,255,0.3);}

    /* ── QUICK COMMANDS ── */
    .quick-wrap{position:relative;z-index:10;padding:5px 12px;flex-shrink:0;}
    .quick-inner{display:flex;gap:6px;overflow-x:auto;scrollbar-width:none;}
    .quick-inner::-webkit-scrollbar{display:none;}
    .qc{
      flex-shrink:0;
      font-family:'Rajdhani',sans-serif;font-size:0.7rem;font-weight:600;letter-spacing:0.5px;
      padding:6px 11px;border-radius:6px;
      border:1px solid var(--muted);
      background:rgba(0,15,45,0.55);
      color:rgba(144,200,232,0.7);
      cursor:pointer;white-space:nowrap;
      transition:border-color 0.15s,color 0.15s,box-shadow 0.15s;
    }
    .qc:active{border-color:var(--blue);color:var(--blue);box-shadow:0 0 10px rgba(0,212,255,0.22);}

    /* ── MIC / ARC REACTOR ── */
    .mic-section{
      position:relative;z-index:10;flex-shrink:0;
      display:flex;flex-direction:column;align-items:center;
      padding:8px 0;
      padding-bottom:max(env(safe-area-inset-bottom,14px),14px);
      gap:8px;
    }
    .wave{display:flex;align-items:center;gap:3px;height:26px;opacity:0;transition:opacity 0.3s;}
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

    .arc-wrap{position:relative;width:144px;height:144px;display:flex;align-items:center;justify-content:center;}
    .ring{position:absolute;border-radius:50%;border:1px solid rgba(0,212,255,0.2);}
    .r1{width:144px;height:144px;animation:rbreath 3s ease-in-out infinite;}
    .r2{width:122px;height:122px;animation:rbreath 3s ease-in-out infinite 0.5s;border-color:rgba(0,212,255,0.12);}
    @keyframes rbreath{0%,100%{transform:scale(1);opacity:.5;}50%{transform:scale(1.05);opacity:1;}}

    .arc-wrap.lst .r1{animation:rlisten 0.65s infinite;border-color:rgba(255,34,85,.75);}
    .arc-wrap.lst .r2{animation:rlisten 0.65s infinite .15s;border-color:rgba(255,34,85,.4);}
    @keyframes rlisten{0%,100%{transform:scale(1);}50%{transform:scale(1.11);}}

    .arc-wrap.thk .r1{animation:rspin 1.8s linear infinite;border-color:transparent;border-top-color:var(--blue);box-shadow:0 0 12px rgba(0,212,255,0.3);}
    .arc-wrap.thk .r2{animation:rspin 3s linear infinite reverse;border-color:transparent;border-top-color:var(--blue2);}
    @keyframes rspin{from{transform:rotate(0);}to{transform:rotate(360deg);}}

    .arc-btn{
      width:96px;height:96px;border-radius:50%;
      background:radial-gradient(circle at 38% 38%,rgba(0,55,130,0.9),rgba(0,4,18,0.97));
      border:2px solid rgba(0,212,255,0.45);
      box-shadow:0 0 24px rgba(0,212,255,0.2),0 0 50px rgba(0,212,255,0.07),inset 0 0 24px rgba(0,212,255,0.07);
      cursor:pointer;position:relative;
      display:flex;align-items:center;justify-content:center;
      transition:transform 0.15s,box-shadow 0.2s;
      -webkit-appearance:none;outline:none;
    }
    .arc-btn:active{transform:scale(0.93);}
    .arc-btn::before{content:'';position:absolute;width:50px;height:50px;border-radius:50%;border:1.5px solid rgba(0,212,255,0.28);}
    .arc-core{
      width:20px;height:20px;border-radius:50%;
      background:radial-gradient(circle,#9aeeff,var(--blue));
      box-shadow:0 0 18px var(--blue),0 0 36px rgba(0,212,255,0.35);
      animation:core-glow 3s ease-in-out infinite;position:relative;z-index:1;
    }
    @keyframes core-glow{0%,100%{box-shadow:0 0 18px var(--blue),0 0 36px rgba(0,212,255,0.35);}50%{box-shadow:0 0 28px var(--blue),0 0 56px rgba(0,212,255,0.55);}}
    .arc-wrap.lst .arc-btn{border-color:rgba(255,34,85,.7);box-shadow:0 0 28px rgba(255,34,85,.35),inset 0 0 24px rgba(255,34,85,.08);}
    .arc-wrap.lst .arc-btn::before{border-color:rgba(255,34,85,.45);}
    .arc-wrap.lst .arc-core{background:radial-gradient(circle,#ffaacc,var(--red));box-shadow:0 0 18px var(--red),0 0 36px rgba(255,34,85,.35);animation:none;}
    .arc-wrap.thk .arc-btn{box-shadow:0 0 36px rgba(0,212,255,.5),0 0 70px rgba(0,212,255,.2),inset 0 0 24px rgba(0,212,255,.12);}
    .arc-wrap.thk .arc-core{animation:core-spin 1s linear infinite;}
    @keyframes core-spin{to{filter:hue-rotate(360deg);}}

    .mic-lbl{font-family:'Orbitron',monospace;font-size:0.56rem;letter-spacing:3px;color:var(--muted);text-align:center;transition:color 0.3s;}
    .mic-lbl.red{color:var(--red);}
    .mic-lbl.blue{color:var(--blue);}
    .mic-lbl.green{color:var(--green);}

    /* ── TYPE INPUT BAR ── */
    .type-bar{
      display:flex;gap:6px;padding:0 12px 6px;
      position:relative;z-index:10;flex-shrink:0;
    }
    .type-input{
      flex:1;background:rgba(0,15,45,0.7);
      border:1px solid var(--muted);border-radius:8px;
      padding:8px 12px;color:var(--text);
      font-family:'Share Tech Mono',monospace;font-size:0.78rem;
      outline:none;transition:border-color 0.2s;
      -webkit-appearance:none;
    }
    .type-input:focus{border-color:rgba(0,212,255,0.4);}
    .type-input::placeholder{color:rgba(144,200,232,0.3);}
    .type-send{
      background:rgba(0,212,255,0.1);border:1px solid rgba(0,212,255,0.3);
      color:var(--blue);border-radius:8px;padding:8px 14px;
      cursor:pointer;font-family:'Orbitron',monospace;font-size:0.65rem;
      letter-spacing:1px;transition:all 0.15s;white-space:nowrap;
    }
    .type-send:active{background:rgba(0,212,255,0.2);}

    /* ── TOAST ── */
    .toast{
      position:fixed;top:76px;left:50%;
      transform:translateX(-50%) translateY(-10px);
      background:rgba(0,15,50,0.96);border:1px solid rgba(0,212,255,0.3);
      border-radius:8px;padding:7px 16px;
      font-family:'Share Tech Mono',monospace;font-size:0.65rem;
      color:var(--blue);letter-spacing:1px;
      z-index:200;opacity:0;transition:all 0.25s;
      pointer-events:none;white-space:nowrap;max-width:90vw;
      text-overflow:ellipsis;overflow:hidden;
    }
    .toast.on{opacity:1;transform:translateX(-50%) translateY(0);}
    .toast.green{border-color:rgba(0,255,136,0.4);color:var(--green);}
    .toast.red{border-color:rgba(255,34,85,0.4);color:var(--red);}

    /* ── NO VOICE ── */
    #noVoice{display:none;position:fixed;inset:0;z-index:300;background:rgba(0,0,0,0.93);flex-direction:column;align-items:center;justify-content:center;gap:14px;padding:28px;text-align:center;}
    #noVoice.on{display:flex;}
    #noVoice h2{font-family:'Orbitron',monospace;font-size:0.95rem;letter-spacing:2px;color:var(--red);}
    #noVoice p{color:var(--muted);font-size:0.88rem;line-height:1.6;}
    #noVoice button{margin-top:8px;padding:10px 22px;background:none;border:1px solid var(--muted);color:var(--text);border-radius:8px;cursor:pointer;font-family:'Share Tech Mono',monospace;font-size:0.75rem;}

    /* ── VOICE PICKER MODAL ── */
    .vmodal{display:none;position:fixed;inset:0;z-index:400;background:rgba(0,0,0,0.82);align-items:flex-end;justify-content:center;}
    .vmodal.on{display:flex;}
    .vmodal-inner{width:100%;max-width:480px;background:#00060f;border:1px solid rgba(0,212,255,0.25);border-bottom:none;border-radius:16px 16px 0 0;padding:18px;max-height:75vh;display:flex;flex-direction:column;}
    .vmodal-hdr{display:flex;justify-content:space-between;align-items:center;font-family:'Orbitron',monospace;font-size:0.6rem;letter-spacing:3px;color:var(--blue);margin-bottom:10px;flex-shrink:0;}
    .vmodal-hdr button{background:none;border:1px solid var(--muted);color:var(--muted);border-radius:4px;padding:3px 8px;cursor:pointer;font-size:0.78rem;transition:all 0.15s;}
    .vmodal-hdr button:active{border-color:var(--red);color:var(--red);}
    .vmodal-note{font-size:0.72rem;color:var(--muted);margin-bottom:10px;line-height:1.4;flex-shrink:0;}
    .vlist{overflow-y:auto;display:flex;flex-direction:column;gap:5px;scrollbar-width:thin;scrollbar-color:var(--muted) transparent;}
    .vlist::-webkit-scrollbar{width:3px;}
    .vlist::-webkit-scrollbar-thumb{background:var(--muted);border-radius:3px;}
    .vitem{padding:10px 12px;border-radius:8px;border:1px solid var(--muted);cursor:pointer;transition:all 0.15s;font-family:'Rajdhani',sans-serif;display:flex;align-items:center;gap:10px;}
    .vitem:active,.vitem.sel{border-color:var(--blue);background:rgba(0,212,255,0.06);box-shadow:0 0 10px rgba(0,212,255,0.12);}
    .vitem-info{flex:1;}
    .vitem .vn{font-size:0.88rem;font-weight:600;color:var(--text);}
    .vitem.sel .vn{color:var(--blue);}
    .vitem .vl{font-size:0.62rem;font-family:'Share Tech Mono',monospace;color:var(--muted);margin-top:1px;}
    .vgender{font-size:0.6rem;padding:2px 6px;border-radius:4px;font-family:'Share Tech Mono',monospace;border:1px solid;flex-shrink:0;}
    .vgender.male{color:#88ccff;border-color:rgba(136,204,255,0.35);background:rgba(136,204,255,0.08);}
    .vgender.female{color:#ffaacc;border-color:rgba(255,170,204,0.35);background:rgba(255,170,204,0.08);}
    .vgender.unknown{color:var(--muted);border-color:var(--muted);background:transparent;}
    .vtest-btn{background:none;border:1px solid var(--muted);color:var(--muted);border-radius:4px;padding:4px 8px;cursor:pointer;font-size:0.65rem;font-family:'Share Tech Mono',monospace;flex-shrink:0;transition:all 0.15s;}
    .vtest-btn:active{border-color:var(--green);color:var(--green);}
    .sel-mark{color:var(--green);font-size:0.8rem;flex-shrink:0;}

    /* ── NOTES PANEL ── */
    .notes-panel{display:none;position:fixed;inset:0;z-index:350;background:rgba(0,0,0,0.88);align-items:flex-end;justify-content:center;}
    .notes-panel.on{display:flex;}
    .notes-inner{width:100%;max-width:480px;background:#00060f;border:1px solid rgba(0,212,255,0.25);border-bottom:none;border-radius:16px 16px 0 0;padding:18px;max-height:70vh;display:flex;flex-direction:column;}
    .notes-hdr{display:flex;justify-content:space-between;align-items:center;font-family:'Orbitron',monospace;font-size:0.6rem;letter-spacing:3px;color:var(--blue);margin-bottom:12px;flex-shrink:0;}
    .notes-list{overflow-y:auto;display:flex;flex-direction:column;gap:8px;}
    .note-item{padding:10px 14px;border-radius:8px;border:1px solid var(--muted);background:rgba(0,15,45,0.5);}
    .note-item .note-text{font-size:0.9rem;color:var(--text);line-height:1.4;}
    .note-item .note-time{font-family:'Share Tech Mono',monospace;font-size:0.6rem;color:var(--muted);margin-top:4px;}

    /* ── PC REMOTE PANEL ── */
    .pc-panel{display:none;position:fixed;inset:0;z-index:350;background:rgba(0,0,0,0.88);align-items:flex-end;justify-content:center;}
    .pc-panel.on{display:flex;}
    .pc-inner{width:100%;max-width:480px;background:#00060f;border:1px solid rgba(0,212,255,0.25);border-bottom:none;border-radius:16px 16px 0 0;padding:18px;max-height:80vh;display:flex;flex-direction:column;}
    .pc-hdr{display:flex;justify-content:space-between;align-items:center;font-family:'Orbitron',monospace;font-size:0.6rem;letter-spacing:3px;color:var(--blue);margin-bottom:6px;flex-shrink:0;}
    .pc-note{font-size:0.7rem;color:var(--muted);margin-bottom:12px;line-height:1.4;flex-shrink:0;}
    .pc-grid{display:grid;grid-template-columns:1fr 1fr;gap:8px;overflow-y:auto;}
    .pc-btn{
      padding:14px 10px;border-radius:10px;border:1px solid var(--muted);
      background:rgba(0,15,45,0.6);color:var(--text);
      cursor:pointer;text-align:center;transition:all 0.15s;
    }
    .pc-btn:active{border-color:var(--blue);color:var(--blue);background:rgba(0,212,255,0.08);}
    .pc-btn .pci{font-size:1.4rem;display:block;margin-bottom:5px;}
    .pc-btn .pcl{font-family:'Orbitron',monospace;font-size:0.55rem;letter-spacing:2px;}
    .pc-btn.danger:active{border-color:var(--red);color:var(--red);background:rgba(255,34,85,0.08);}

    /* ── OFFLINE BADGE ── */
    .offline-badge{
      display:none;position:fixed;top:0;left:0;right:0;z-index:500;
      background:rgba(255,34,85,0.15);border-bottom:1px solid rgba(255,34,85,0.3);
      text-align:center;padding:6px;
      font-family:'Share Tech Mono',monospace;font-size:0.65rem;
      color:var(--red);letter-spacing:1px;
    }
    .offline-badge.on{display:block;}

    @media(min-height:700px){
      .arc-wrap{width:158px;height:158px;}
      .r1{width:158px;height:158px;}
      .r2{width:134px;height:134px;}
      .arc-btn{width:108px;height:108px;}
    }
  </style>
</head>
<body>

<div class="offline-badge" id="offlineBadge">⚡ OFFLINE MODE — Using cached data</div>
<div class="bg-glow"></div>
<div class="hex-grid"></div>
<div class="scanlines"></div>
<div class="hc tl"></div><div class="hc tr"></div>
<div class="hc bl"></div><div class="hc br"></div>

<header class="hud-header">
  <div class="hud-row1">
    <div class="j-name">JARVIS</div>
    <div class="hud-btns">
      <button class="hbtn" id="pcBtn" onclick="openPCPanel()" title="PC Remote Control">🖥 PC</button>
      <button class="hbtn" id="notesBtn" onclick="openNotesPanel()" title="View Notes">📝 NOTES</button>
      <button class="hbtn" onclick="openVoicePicker()" title="Change Voice">🔊 VOICE</button>
      <div class="sys-badge">
        <div class="led" id="led"></div>
        <span class="sys-lbl" id="sysLbl">ONLINE</span>
      </div>
    </div>
  </div>
  <div class="hud-meta">
    <span>SYS&nbsp;<span class="v">NOMINAL</span></span>
    <span>AI&nbsp;<span class="v">GROQ LLM</span></span>
    <span>MEM&nbsp;<span class="v">ACTIVE</span></span>
    <span id="battLine">BATT&nbsp;<span class="v" id="battVal">--</span></span>
  </div>
</header>

<div class="resp-area">
  <div class="holo-panel">
    <div class="panel-lbl-row">
      <div class="panel-lbl">◈ JARVIS OUTPUT</div>
      <button class="copy-resp-btn" id="copyRespBtn" onclick="copyResponse()" title="Copy this response">📋 COPY</button>
    </div>
    <div class="query-line" id="qLine"></div>
    <div class="resp-txt" id="rTxt">Initialising systems…</div>
    <div class="tdots" id="tDots"><b></b><b></b><b></b></div>
    <button class="action-btn" id="actionBtn" onclick="handleActionBtn()"></button>
  </div>
</div>

<div class="quick-wrap">
  <div class="quick-inner">
    <button class="qc" id="qc1" onclick="runCmd('world news')">🌍 NEWS</button>
    <button class="qc" id="qc2" onclick="runCmd('weather today')">🌤 WEATHER</button>
    <button class="qc" id="qc3" onclick="runCmd('open youtube')">▶ YOUTUBE</button>
    <button class="qc" id="qc4" onclick="runCmd('open maps')">🗺 MAPS</button>
    <button class="qc" id="qc5" onclick="runCmd('open camera')">📷 CAMERA</button>
    <button class="qc" id="qc6" onclick="runCmd('open whatsapp')">💬 WHATSAPP</button>
    <button class="qc" id="qc7" onclick="runCmd('open spotify')">🎵 SPOTIFY</button>
    <button class="qc" id="qc8" onclick="runCmd('battery status')">🔋 BATTERY</button>
    <button class="qc" id="qc9" onclick="runCmd('my location')">📍 LOCATION</button>
    <button class="qc" id="qc10" onclick="runCmd('motivate me')">⚡ MOTIVATE</button>
    <button class="qc" id="qc11" onclick="runCmd('tell me a joke')">😄 JOKE</button>
    <button class="qc" id="qc12" onclick="runCmd('what time is it')">🕐 TIME</button>
    <button class="qc" id="qc13" onclick="runCmd('open netflix')">🎬 NETFLIX</button>
    <button class="qc" id="qc14" onclick="runCmd('open instagram')">📸 INSTAGRAM</button>
    <button class="qc" id="qc15" onclick="runCmd('open calculator')">🧮 CALC</button>
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

<div class="type-bar">
  <input class="type-input" id="typeInput" type="text" placeholder="Or type a command…" autocomplete="off" autocorrect="off"/>
  <button class="type-send" id="typeSend" onclick="sendTyped()">SEND</button>
</div>

<div class="toast" id="toast"></div>

<!-- Voice Picker Modal -->
<div class="vmodal" id="vModal">
  <div class="vmodal-inner">
    <div class="vmodal-hdr">
      <span>🔊 SELECT JARVIS VOICE</span>
      <button onclick="closeVoicePicker()">✕ CLOSE</button>
    </div>
    <div class="vmodal-note" id="vNote">
      For a deep male voice, look for: <strong>Google UK English Male</strong>, <strong>Daniel</strong>, <strong>Alex</strong>, or <strong>Mark</strong>. Tap ▶ to preview each voice.
    </div>
    <div class="vlist" id="vList"></div>
  </div>
</div>

<!-- Notes Panel -->
<div class="notes-panel" id="notesPanel">
  <div class="notes-inner">
    <div class="notes-hdr">
      <span>📝 SAVED NOTES</span>
      <button class="hbtn" onclick="closeNotesPanel()">✕ CLOSE</button>
    </div>
    <div class="notes-list" id="notesList"></div>
  </div>
</div>

<!-- PC Remote Panel -->
<div class="pc-panel" id="pcPanel">
  <div class="pc-inner">
    <div class="pc-hdr">
      <span>🖥 PC REMOTE CONTROL</span>
      <button class="hbtn" onclick="closePCPanel()">✕ CLOSE</button>
    </div>
    <div class="pc-note">Requires jarvis.py running on your PC (same network). Commands are sent to localhost:5001.</div>
    <div class="pc-grid">
      <button class="pc-btn" onclick="sendPCCmd('volume_up')"><span class="pci">🔊</span><span class="pcl">VOL UP</span></button>
      <button class="pc-btn" onclick="sendPCCmd('volume_down')"><span class="pci">🔉</span><span class="pcl">VOL DOWN</span></button>
      <button class="pc-btn" onclick="sendPCCmd('mute')"><span class="pci">🔇</span><span class="pcl">MUTE</span></button>
      <button class="pc-btn" onclick="sendPCCmd('lock')"><span class="pci">🔒</span><span class="pcl">LOCK PC</span></button>
      <button class="pc-btn" onclick="sendPCCmd('sleep')"><span class="pci">💤</span><span class="pcl">SLEEP</span></button>
      <button class="pc-btn" onclick="sendPCCmd('screenshot_pc')"><span class="pci">🖼</span><span class="pcl">SCREENSHOT</span></button>
      <button class="pc-btn danger" onclick="sendPCCmd('restart')"><span class="pci">🔄</span><span class="pcl">RESTART</span></button>
      <button class="pc-btn danger" onclick="sendPCCmd('shutdown')"><span class="pci">⚫</span><span class="pcl">SHUTDOWN</span></button>
    </div>
  </div>
</div>

<div id="noVoice">
  <h2>⚠ VOICE NOT SUPPORTED</h2>
  <p>For best experience, use <strong>Chrome on Android</strong><br>or <strong>Safari on iOS</strong>.</p>
  <button onclick="document.getElementById('noVoice').classList.remove('on')">CONTINUE ANYWAY</button>
</div>

<script>
'use strict';

/* ── DOM refs ── */
const arcWrap  = document.getElementById('arcWrap');
const micLbl   = document.getElementById('micLbl');
const led      = document.getElementById('led');
const sysLbl   = document.getElementById('sysLbl');
const rTxt     = document.getElementById('rTxt');
const qLine    = document.getElementById('qLine');
const tDots    = document.getElementById('tDots');
const waveEl   = document.getElementById('wave');
const toastEl  = document.getElementById('toast');
const actionBtn= document.getElementById('actionBtn');
const typeInput= document.getElementById('typeInput');

/* ── TTS ── */
const synth = window.speechSynthesis;
let voices  = [];
let selVoice = null;

// Exhaustive known-male voice names across Android, iOS, Windows, Samsung
const MALE_VOICE_PATTERNS = [
  /google uk english male/i,
  /google us english.*male/i,
  /daniel/i,         // iOS/macOS
  /alex/i,           // macOS
  /mark/i,           // Windows
  /david/i,          // Windows
  /james/i,
  /thomas/i,
  /george/i,
  /oliver/i,
  /arthur/i,
  /edgar/i,
  /luca/i,
  /aaron/i,
  /fred/i,
  /ralph/i,
  /bruce/i,
  /lee/i,
  /male/i,           // Any voice explicitly named "male"
  /man/i,
];

function detectGender(voice) {
  const name = voice.name.toLowerCase();
  if (MALE_VOICE_PATTERNS.some(p => p.test(voice.name))) return 'male';
  const femaleNames = /samantha|karen|moira|fiona|veena|ioana|amelie|alice|alva|anna|sara|nora|ellen|lekha|damayanti|tessa|zosia|paulina|lucia|monica|joana|yelena|milena|zuzana|marie|google uk english female|google us english female|female|woman/i;
  if (femaleNames.test(voice.name)) return 'female';
  return 'unknown';
}

function loadVoices() {
  if (!synth) return;
  voices = synth.getVoices();
  const saved = localStorage.getItem('jarvisVoiceURI');
  if (saved && !selVoice) {
    const found = voices.find(v => v.voiceURI === saved);
    if (found) selVoice = found;
  }
  // Auto-pick male voice if nothing is saved yet
  if (!selVoice && voices.length > 0) {
    selVoice = getBestVoice();
  }
}
if (synth && synth.onvoiceschanged !== undefined) synth.onvoiceschanged = loadVoices;
loadVoices();

function getBestVoice() {
  if (selVoice) return selVoice;
  // Priority order: known male voices
  for (const pat of MALE_VOICE_PATTERNS) {
    const v = voices.find(v => pat.test(v.name) && v.lang.startsWith('en'));
    if (v) return v;
  }
  // Any English voice
  return voices.find(v => v.lang.startsWith('en-')) || voices.find(v => v.lang.startsWith('en')) || null;
}

function speak(text) {
  if (!synth) return;
  synth.cancel();
  const u = new SpeechSynthesisUtterance(text);
  // Deep, authoritative JARVIS voice settings
  u.rate   = 0.88;
  u.pitch  = 0.72;   // Lower = deeper
  u.volume = 1;
  const pick = getBestVoice();
  if (pick) u.voice = pick;
  synth.speak(u);
}

/* ── Voice Picker ── */
function openVoicePicker() {
  const modal = document.getElementById('vModal');
  const list  = document.getElementById('vList');
  list.innerHTML = '';
  if (voices.length === 0) loadVoices();
  const engVoices = voices.filter(v => v.lang.toLowerCase().startsWith('en'));
  if (engVoices.length === 0) {
    list.innerHTML = '<p style="color:var(--muted);text-align:center;padding:20px;font-size:0.85rem;">No voices loaded yet.<br>Wait a moment and try again.</p>';
  } else {
    // Sort: male first, then by name
    const sorted = [...engVoices].sort((a, b) => {
      const ga = detectGender(a), gb = detectGender(b);
      if (ga === 'male' && gb !== 'male') return -1;
      if (gb === 'male' && ga !== 'male') return 1;
      return a.name.localeCompare(b.name);
    });
    sorted.forEach(v => {
      const isSel   = selVoice && selVoice.voiceURI === v.voiceURI;
      const gender  = detectGender(v);
      const item    = document.createElement('div');
      item.className = 'vitem' + (isSel ? ' sel' : '');
      item.innerHTML  = `
        <div class="vitem-info">
          <div class="vn">${v.name}${isSel ? ' <span class="sel-mark">✓</span>' : ''}</div>
          <div class="vl">${v.lang} • ${v.localService ? 'LOCAL' : 'NETWORK'}</div>
        </div>
        <span class="vgender ${gender}">${gender.toUpperCase()}</span>
        <button class="vtest-btn" title="Test this voice">▶</button>
      `;
      // Click item to select
      item.addEventListener('click', e => {
        if (e.target.classList.contains('vtest-btn')) return;
        selVoice = v;
        localStorage.setItem('jarvisVoiceURI', v.voiceURI);
        closeVoicePicker();
        if (synth) synth.cancel();
        const u = new SpeechSynthesisUtterance('Voice confirmed. I am JARVIS, at your service, sir.');
        u.voice = v; u.rate = 0.88; u.pitch = 0.72; u.volume = 1;
        if (synth) synth.speak(u);
        showToast('VOICE SET: ' + v.name.toUpperCase());
      });
      // Test button
      item.querySelector('.vtest-btn').addEventListener('click', e => {
        e.stopPropagation();
        if (synth) synth.cancel();
        const u = new SpeechSynthesisUtterance('Systems online. JARVIS at your service.');
        u.voice = v; u.rate = 0.88; u.pitch = 0.72; u.volume = 1;
        if (synth) synth.speak(u);
      });
      list.appendChild(item);
    });
  }
  modal.classList.add('on');
}
function closeVoicePicker() { document.getElementById('vModal').classList.remove('on'); }
document.getElementById('vModal').addEventListener('click', function(e) { if (e.target === this) closeVoicePicker(); });

/* ── Notes Panel ── */
let sessionNotes = [];
function openNotesPanel() {
  const list = document.getElementById('notesList');
  list.innerHTML = '';
  if (sessionNotes.length === 0) {
    list.innerHTML = '<p style="color:var(--muted);text-align:center;padding:20px;font-size:0.85rem;">No notes yet, sir.<br>Say "take a note" to add one.</p>';
  } else {
    sessionNotes.forEach((n, i) => {
      const el = document.createElement('div');
      el.className = 'note-item';
      el.innerHTML = `<div class="note-text">${n.text}</div><div class="note-time">${n.time || ''}</div>`;
      list.appendChild(el);
    });
  }
  document.getElementById('notesPanel').classList.add('on');
}
function closeNotesPanel() { document.getElementById('notesPanel').classList.remove('on'); }

/* ── PC Remote Panel ── */
function openPCPanel()  { document.getElementById('pcPanel').classList.add('on'); }
function closePCPanel() { document.getElementById('pcPanel').classList.remove('on'); }
async function sendPCCmd(action) {
  closePCPanel();
  showToast('SENDING TO PC: ' + action.toUpperCase().replace('_', ' '));
  try {
    const res = await fetch('/pc-command', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({action})
    });
    const data = await res.json();
    const reply = data.ok ? `PC ${action.replace('_',' ')} executed.` : `PC error: ${data.result}`;
    typewrite(reply);
    speak(reply);
  } catch(e) {
    const msg = 'Cannot reach your PC. Make sure jarvis.py is running.';
    typewrite(msg); speak(msg);
  }
}

/* ── App State ── */
let appState = 'idle';
let pendingActionUrl = null;

function setState(s) {
  appState = s;
  arcWrap.className = 'arc-wrap' + (s === 'listening' ? ' lst' : s === 'thinking' ? ' thk' : '');
  waveEl.className  = 'wave' + (s === 'listening' ? ' on' : '');
  tDots.className   = 'tdots' + (s === 'thinking' ? ' on' : '');
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

let twTimer = null;
let lastReplyText = '';
let typewrite = function(text) {
  lastReplyText = text;
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

let toastTimer = null;
function showToast(msg, type='') {
  toastEl.textContent = msg;
  toastEl.className = 'toast on' + (type ? ' ' + type : '');
  if (toastTimer) clearTimeout(toastTimer);
  toastTimer = setTimeout(() => toastEl.classList.remove('on'), 3200);
}

/* ── Handle special actions from backend ── */
function handleSpecialAction(data) {
  const action = data.action;
  pendingActionUrl = null;
  actionBtn.className = 'action-btn';

  if (action === 'open_url' && data.url) {
    const lbl = data.url_label || 'OPEN LINK';
    // Try primary URL, fallback on error
    setTimeout(() => {
      const target = data.url;
      if (target.startsWith('tel:') || target.startsWith('intent:') || target.startsWith('app-settings:')) {
        window.location.href = target;
      } else {
        window.open(target, '_blank');
      }
      showToast('>> OPENING ' + lbl);
    }, 900);
  }

  if (action === 'open_camera') {
    // Use getUserMedia to open camera
    setTimeout(async () => {
      try {
        const stream = await navigator.mediaDevices.getUserMedia({video: true});
        stream.getTracks().forEach(t => t.stop());
        // Show a quick camera access confirmation
        showToast('CAMERA ACCESSED', 'green');
      } catch(e) {
        // Fallback: open camera via deep link
        window.location.href = 'intent://scan/#Intent;scheme=zxing;package=com.google.zxing.client.android;end';
      }
    }, 500);
  }

  if (action === 'battery_check') {
    if ('getBattery' in navigator) {
      navigator.getBattery().then(batt => {
        const pct    = Math.round(batt.level * 100);
        const status = batt.charging ? 'charging' : 'not charging';
        const msg    = `Battery is at ${pct}%, ${status}.`;
        typewrite(msg); speak(msg);
        // Update header
        document.getElementById('battVal').textContent = pct + '%';
        document.getElementById('battLine').classList.add('on');
      }).catch(() => {
        const msg = 'Battery API unavailable on this device.';
        typewrite(msg); speak(msg);
      });
    } else {
      const msg = 'Battery status is not supported in this browser.';
      typewrite(msg); speak(msg);
    }
    return true; // handled
  }

  if (action === 'location_check') {
    if ('geolocation' in navigator) {
      setState('thinking');
      navigator.geolocation.getCurrentPosition(pos => {
        const lat = pos.coords.latitude.toFixed(4);
        const lon = pos.coords.longitude.toFixed(4);
        const msg = `Your location: ${lat}°N, ${lon}°E. Opening map.`;
        typewrite(msg); speak(msg); setState('speaking');
        setTimeout(() => window.open(`https://maps.google.com/?q=${lat},${lon}`, '_blank'), 1000);
      }, err => {
        const msg = 'Location access denied or unavailable.';
        typewrite(msg); speak(msg); setState('idle');
      });
    } else {
      const msg = 'Geolocation is not supported in this browser.';
      typewrite(msg); speak(msg);
    }
    return true;
  }

  if (action === 'note_saved') {
    if (data.note) {
      const now = new Date().toLocaleTimeString('en-IN', {hour:'2-digit', minute:'2-digit'});
      sessionNotes.push({text: data.note, time: now});
    }
    showToast('NOTE SAVED', 'green');
  }

  if (action === 'show_notes') {
    openNotesPanel();
  }

  if (action === 'screenshot') {
    showToast('SCREENSHOT — use your phone\'s button');
  }

  if (action === 'exit') {
    setTimeout(() => window.close(), 2500);
  }

  if (action === 'pc_command' && data.pc_action) {
    sendPCCmd(data.pc_action);
    return true;
  }

  return false;
}

/* ── Main command runner ── */
async function runCmd(text) {
  if (!text || appState === 'thinking') return;
  if (synth) synth.cancel();
  setState('thinking');
  qLine.textContent = text; qLine.className = 'query-line on';
  rTxt.textContent = ''; rTxt.className = 'resp-txt';
  actionBtn.className = 'action-btn';

  // AbortController for 35s timeout (Render free can be slow on cold start)
  const controller = new AbortController();
  const timeoutId  = setTimeout(() => controller.abort(), 35000);

  try {
    const res  = await fetch('/command', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({command: text}),
      signal: controller.signal
    });
    clearTimeout(timeoutId);
    if (!res.ok) {
      throw new Error(`Server returned ${res.status}`);
    }
    const data = await res.json();
    const reply = data.reply || 'No response received.';
    setState('speaking');
    typewrite(reply);
    speak(reply);
    // Handle special actions
    if (data.action) {
      const handled = handleSpecialAction(data);
      if (data.action === 'open_url' && data.url) {
        actionBtn.textContent = '>> OPEN ' + (data.url_label || 'LINK');
        actionBtn.className = 'action-btn on';
        pendingActionUrl = data.url;
      }
    }
  } catch(e) {
    clearTimeout(timeoutId);
    setState('idle');
    let msg;
    if (e.name === 'AbortError') {
      msg = 'Server is waking up (Render free tier). Please tap the reactor and try again in a few seconds, sir.';
      showToast('COLD START — retrying...', 'red');
    } else if (!navigator.onLine) {
      msg = 'Offline mode — limited functionality available.';
    } else {
      msg = `Neural link disrupted: ${e.message || 'request failed'}. If this persists, check GROQ_API_KEY on Render, sir.`;
    }
    typewrite(msg);
  }
}

function handleActionBtn() {
  if (pendingActionUrl) {
    if (pendingActionUrl.startsWith('tel:') || pendingActionUrl.startsWith('intent:') || pendingActionUrl.startsWith('app-settings:')) {
      window.location.href = pendingActionUrl;
    } else {
      window.open(pendingActionUrl, '_blank');
    }
    actionBtn.className = 'action-btn';
    pendingActionUrl = null;
  }
}

/* ── Text input ── */
function sendTyped() {
  const text = typeInput.value.trim();
  if (!text) return;
  typeInput.value = '';
  runCmd(text);
}
typeInput.addEventListener('keydown', e => { if (e.key === 'Enter') sendTyped(); });

/* ── Speech Recognition ── */
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
  rec.onresult = (e) => { micActive = false; runCmd(e.results[0][0].transcript); };
  rec.onerror  = (e) => {
    micActive = false; setState('idle');
    if (e.error === 'no-speech')   showToast('NO SPEECH DETECTED');
    else if (e.error === 'not-allowed') showToast('MIC ACCESS DENIED — check browser settings', 'red');
    else showToast('MIC ERROR: ' + e.error.toUpperCase(), 'red');
  };
  rec.onend = () => { if (micActive) { micActive = false; setState('idle'); } };
}

function toggleMic() {
  if (!rec) return;
  if (micActive) {
    rec.stop(); micActive = false; setState('idle');
  } else {
    if (appState === 'thinking') return;
    if (synth) synth.cancel();
    try { rec.start(); micActive = true; setState('listening'); }
    catch(e) { micActive = false; setState('idle'); showToast('MIC UNAVAILABLE', 'red'); }
  }
}

/* ── Battery monitoring ── */
if ('getBattery' in navigator) {
  navigator.getBattery().then(batt => {
    const update = () => {
      const pct = Math.round(batt.level * 100);
      document.getElementById('battVal').textContent = pct + '%' + (batt.charging ? '⚡' : '');
      document.getElementById('battLine').classList.add('on');
    };
    update();
    batt.addEventListener('levelchange', update);
    batt.addEventListener('chargingchange', update);
  });
}

/* ── Online/Offline ── */
function updateOnlineStatus() {
  const badge = document.getElementById('offlineBadge');
  if (navigator.onLine) {
    badge.classList.remove('on');
  } else {
    badge.classList.add('on');
  }
}
window.addEventListener('online',  updateOnlineStatus);
window.addEventListener('offline', updateOnlineStatus);
updateOnlineStatus();

/* ── Service Worker Registration ── */
if ('serviceWorker' in navigator) {
  navigator.serviceWorker.register('/sw.js').then(reg => {
    console.log('[SW] Registered:', reg.scope);
  }).catch(err => console.log('[SW] Registration failed:', err));
}

/* ── First-time voice picker auto-launch ── */
const firstVisit = !localStorage.getItem('jarvisVoiceURI');
if (firstVisit && synth) {
  // Wait for voices to load, then show picker
  const showPickerWhenReady = () => {
    loadVoices();
    if (voices.length > 0) {
      setTimeout(() => {
        showToast('FIRST VISIT — Please select your preferred JARVIS voice');
        setTimeout(openVoicePicker, 1200);
      }, 1500);
    } else {
      setTimeout(showPickerWhenReady, 500);
    }
  };
  setTimeout(showPickerWhenReady, 800);
}

/* ── Pre-warm server ping (prevents cold-start failures) ── */
(async () => {
  try { await fetch('/ping', {signal: AbortSignal.timeout ? AbortSignal.timeout(5000) : new AbortController().signal}); }
  catch(e) { /* silent — just warming up */ }
})();

/* ── Greeting on load ── */
(async () => {
  const controller = new AbortController();
  const tId = setTimeout(() => controller.abort(), 4000);
  try {
    const res  = await fetch('/greet', { signal: controller.signal });
    const data = await res.json();
    clearTimeout(tId);
    setState('speaking');
    typewrite(data.greeting);
    speak(data.greeting);
  } catch(e) {
    clearTimeout(tId);
    typewrite('JARVIS online. Tap the reactor to speak, or type below.');
    setState('idle');
  }
})();
/* ════════════════════════════════════════════════
   CODE EXECUTION ENGINE — IDE Panel
   ════════════════════════════════════════════════ */

/* Inject highlight.js for syntax highlighting */
(function() {
  const hlLink = document.createElement('link');
  hlLink.rel  = 'stylesheet';
  hlLink.href = 'https://cdnjs.cloudflare.com/ajax/libs/highlight.js/11.9.0/styles/atom-one-dark.min.css';
  document.head.appendChild(hlLink);

  const hlScript = document.createElement('script');
  hlScript.src = 'https://cdnjs.cloudflare.com/ajax/libs/highlight.js/11.9.0/highlight.min.js';
  hlScript.onload = () => {
    if (window.hljs) window.hljs.configure({tabReplace: '  '});
  };
  document.head.appendChild(hlScript);
})();

/* ── Inject IDE panel styles ── */
(function() {
  const s = document.createElement('style');
  s.textContent = `
  /* ── CODE IDE PANEL ── */
  .code-panel {
    display:none;position:fixed;inset:0;z-index:450;
    background:rgba(0,0,0,0.92);
    flex-direction:column;
  }
  .code-panel.on { display:flex; }
  .ide-hdr {
    display:flex;align-items:center;gap:10px;
    background:#0d1117;padding:10px 14px;flex-shrink:0;
    border-bottom:1px solid rgba(0,212,255,0.15);
  }
  .ide-lang {
    font-family:'Orbitron',monospace;font-size:0.62rem;
    letter-spacing:3px;color:#00d4ff;flex:1;
  }
  .ide-close {
    background:none;border:1px solid #1e3a5a;color:#90c8e8;
    border-radius:5px;padding:4px 10px;cursor:pointer;
    font-family:'Share Tech Mono',monospace;font-size:0.7rem;
  }
  .ide-close:active { border-color:#ff2255;color:#ff2255; }
  .ide-tabs {
    display:flex;gap:0;background:#0d1117;flex-shrink:0;
    border-bottom:1px solid rgba(0,212,255,0.1);
  }
  .ide-tab {
    padding:7px 16px;font-family:'Orbitron',monospace;
    font-size:0.55rem;letter-spacing:2px;color:#1e3a5a;
    cursor:pointer;border-bottom:2px solid transparent;
    transition:all 0.15s;
  }
  .ide-tab.active { color:#00d4ff;border-bottom-color:#00d4ff; }
  .ide-body { flex:1;overflow:hidden;display:flex;flex-direction:column; }
  .ide-code-wrap {
    flex:1;overflow:auto;background:#0d1117;
    position:relative;
    user-select:text;-webkit-user-select:text;
  }
  .ide-code-wrap pre {
    margin:0;padding:16px;
    font-size:0.82rem;line-height:1.6;
    font-family:'Share Tech Mono',monospace;
  }
  .ide-code-wrap code { font-family:'Share Tech Mono',monospace !important; }
  .ide-actions {
    display:flex;gap:8px;padding:10px 14px;
    background:#0d1117;flex-shrink:0;
    border-top:1px solid rgba(0,212,255,0.08);
  }
  .ide-stdin-wrap {
    padding:8px 14px 0;background:#0d1117;flex-shrink:0;
  }
  .ide-stdin-lbl {
    font-family:'Share Tech Mono',monospace;font-size:0.58rem;
    color:rgba(0,212,255,0.5);letter-spacing:0.5px;margin-bottom:5px;
  }
  .ide-stdin {
    width:100%;resize:vertical;min-height:38px;
    background:#161b22;border:1px solid rgba(0,212,255,0.18);
    border-radius:6px;padding:7px 10px;color:#90c8e8;
    font-family:'Share Tech Mono',monospace;font-size:0.78rem;
    outline:none;transition:border-color 0.2s;
  }
  .ide-stdin:focus { border-color:rgba(0,212,255,0.5); }
  .ide-stdin::placeholder { color:rgba(144,200,232,0.28); }
  .ide-run-btn {
    flex:1;padding:11px;border-radius:8px;
    background:rgba(0,212,255,0.1);
    border:1px solid rgba(0,212,255,0.4);
    color:#00d4ff;font-family:'Orbitron',monospace;
    font-size:0.62rem;letter-spacing:2px;cursor:pointer;
    transition:all 0.15s;
  }
  .ide-run-btn:active { background:rgba(0,212,255,0.22); }
  .ide-run-btn.running { border-color:#ffaa00;color:#ffaa00;animation:pulse 0.8s infinite; }
  .ide-copy-btn {
    padding:11px 16px;border-radius:8px;
    background:rgba(0,255,136,0.07);
    border:1px solid rgba(0,255,136,0.3);
    color:#00ff88;font-family:'Orbitron',monospace;
    font-size:0.62rem;letter-spacing:1px;cursor:pointer;
    transition:all 0.15s;
  }
  .ide-copy-btn:active { background:rgba(0,255,136,0.18); }
  @keyframes pulse { 0%,100%{opacity:1;}50%{opacity:0.5;} }

  /* ── Terminal Output ── */
  .ide-output {
    max-height:45vh;overflow-y:auto;
    background:#0a0f0a;
    border-top:1px solid rgba(0,255,136,0.12);
    font-family:'Share Tech Mono',monospace;font-size:0.78rem;
    padding:12px 14px;display:none;flex-direction:column;gap:4px;
    user-select:text;-webkit-user-select:text;
  }
  .ide-output.on { display:flex; }
  .out-hdr {
    display:flex;justify-content:space-between;align-items:center;
    margin-bottom:6px;
  }
  .out-lbl { color:rgba(0,255,136,0.6);font-size:0.6rem;letter-spacing:2px; }
  .out-meta { color:#1e3a5a;font-size:0.6rem; }
  .out-stdout { color:#00ff88;white-space:pre-wrap;word-break:break-all; }
  .out-stderr { color:#ff2255;white-space:pre-wrap;word-break:break-all;margin-top:6px; }
  .out-empty  { color:#1e3a5a;font-style:italic; }

  /* ── HTML/JS Preview iframe ── */
  .ide-preview {
    flex:1;border:none;background:#fff;display:none;
  }
  .ide-preview.on { display:block; }

  /* ── Code blocks inside .resp-txt ── */
  .jarvis-code-block {
    margin:10px 0;border-radius:10px;overflow:hidden;
    border:1px solid rgba(0,212,255,0.15);
    background:#0d1117;
    white-space:normal;
  }
  .jcb-header {
    display:flex;align-items:center;justify-content:space-between;
    padding:6px 12px;background:#161b22;
    border-bottom:1px solid rgba(0,212,255,0.08);
  }
  .jcb-lang {
    font-family:'Orbitron',monospace;font-size:0.52rem;
    letter-spacing:3px;color:#00d4ff;
  }
  .jcb-btns { display:flex;gap:6px; }
  .jcb-btn {
    padding:3px 9px;border-radius:4px;
    font-family:'Share Tech Mono',monospace;font-size:0.65rem;
    cursor:pointer;border:1px solid;transition:all 0.15s;
    background:none;
  }
  .jcb-copy { color:#00ff88;border-color:rgba(0,255,136,0.3); }
  .jcb-copy:active { background:rgba(0,255,136,0.12); }
  .jcb-run  { color:#00d4ff;border-color:rgba(0,212,255,0.35); }
  .jcb-run:active  { background:rgba(0,212,255,0.12); }
  .jcb-code { padding:12px;overflow-x:auto;max-height:260px;overflow-y:auto;user-select:text;-webkit-user-select:text; }
  .jcb-code pre { margin:0; }
  .jcb-code code {
    font-family:'Share Tech Mono',monospace !important;
    font-size:0.78rem !important;line-height:1.55;
  }
  `;
  document.head.appendChild(s);
})();

/* ── Inject IDE panel HTML ── */
(function() {
  const panel = document.createElement('div');
  panel.className = 'code-panel';
  panel.id = 'codePanel';
  panel.innerHTML = `
    <div class="ide-hdr">
      <div class="ide-lang" id="ideLang">PYTHON</div>
      <button class="ide-close" onclick="closeIDE()">X CLOSE</button>
    </div>
    <div class="ide-tabs">
      <div class="ide-tab active" id="tabCode"  onclick="switchIDETab('code')">CODE</div>
      <div class="ide-tab"        id="tabOutput" onclick="switchIDETab('output')">OUTPUT</div>
      <div class="ide-tab"        id="tabPreview" onclick="switchIDETab('preview')" style="display:none">PREVIEW</div>
    </div>
    <div class="ide-body">
      <div class="ide-code-wrap" id="ideCodeWrap">
        <pre><code id="ideCode" class="hljs"></code></pre>
      </div>
      <div class="ide-output" id="ideOutput">
        <div class="out-hdr">
          <span class="out-lbl">TERMINAL OUTPUT</span>
          <span class="out-meta" id="outMeta"></span>
        </div>
        <div class="out-stdout" id="outStdout"></div>
        <div class="out-stderr" id="outStderr"></div>
      </div>
      <iframe class="ide-preview" id="idePreview" sandbox="allow-scripts allow-same-origin"></iframe>
    </div>
    <div class="ide-stdin-wrap" id="ideStdinWrap">
      <div class="ide-stdin-lbl">⌨ INPUT (one value per line — feeds input() calls in order)</div>
      <textarea class="ide-stdin" id="ideStdin" rows="2" placeholder="e.g.&#10;5&#10;10"></textarea>
    </div>
    <div class="ide-actions">
      <button class="ide-run-btn" id="ideRunBtn" onclick="runIDECode()">&#9654; RUN CODE</button>
      <button class="ide-copy-btn" id="ideCopyBtn" onclick="copyIDECode()">COPY</button>
    </div>
  `;
  document.body.appendChild(panel);
})();

/* ── IDE State ── */
let ideCode = '', ideLang = 'python', ideTab = 'code';

function openIDE(code, lang) {
  ideCode = code;
  ideLang = (lang || 'python').toLowerCase();
  document.getElementById('ideLang').textContent = ideLang.toUpperCase();
  const codeEl = document.getElementById('ideCode');
  codeEl.textContent = code;
  codeEl.className = 'hljs language-' + ideLang;
  if (window.hljs) window.hljs.highlightElement(codeEl);

  // Show/hide Preview tab
  const previewTab = document.getElementById('tabPreview');
  if (['html', 'htm', 'css', 'javascript', 'js'].includes(ideLang)) {
    previewTab.style.display = 'block';
  } else {
    previewTab.style.display = 'none';
  }

  // Show/hide stdin input box — only relevant for server-executed languages
  const stdinWrap = document.getElementById('ideStdinWrap');
  const stdinEl    = document.getElementById('ideStdin');
  if (['html', 'htm', 'css', 'javascript', 'js'].includes(ideLang)) {
    stdinWrap.style.display = 'none';
  } else {
    stdinWrap.style.display = 'block';
    if (stdinEl) stdinEl.value = '';
  }

  // Reset output
  document.getElementById('ideOutput').classList.remove('on');
  document.getElementById('idePreview').classList.remove('on');
  document.getElementById('ideCodeWrap').style.display = 'block';
  switchIDETab('code');
  document.getElementById('codePanel').classList.add('on');
}

function closeIDE() {
  document.getElementById('codePanel').classList.remove('on');
  document.getElementById('idePreview').src = 'about:blank';
}

function switchIDETab(tab) {
  ideTab = tab;
  document.getElementById('tabCode').classList.toggle('active',   tab === 'code');
  document.getElementById('tabOutput').classList.toggle('active', tab === 'output');
  document.getElementById('tabPreview').classList.toggle('active',tab === 'preview');

  document.getElementById('ideCodeWrap').style.display = tab === 'code' ? 'block' : 'none';
  const outEl = document.getElementById('ideOutput');
  outEl.classList.toggle('on', tab === 'output');
  const prevEl = document.getElementById('idePreview');
  prevEl.classList.toggle('on', tab === 'preview');
}

function copyIDECode() {
  navigator.clipboard.writeText(ideCode).then(() => {
    showToast('CODE COPIED', 'green');
    const btn = document.getElementById('ideCopyBtn');
    btn.textContent = 'COPIED!';
    setTimeout(() => btn.textContent = 'COPY', 1500);
  }).catch(() => {
    showToast('COPY FAILED', 'red');
  });
}

async function runIDECode() {
  const lang   = ideLang;
  const code   = ideCode;
  const runBtn = document.getElementById('ideRunBtn');
  runBtn.textContent = 'RUNNING...';
  runBtn.classList.add('running');

  // ── HTML/CSS: live preview in iframe ──
  if (['html', 'htm'].includes(lang)) {
    document.getElementById('idePreview').srcdoc = code;
    switchIDETab('preview');
    runBtn.textContent = '\u25B6 RUN CODE';
    runBtn.classList.remove('running');
    return;
  }

  // ── JavaScript: run in sandboxed iframe ──
  if (['js', 'javascript'].includes(lang)) {
    const html = `<!DOCTYPE html><html><body><script>
    const _log = [], _err = [];
    const _origLog = console.log, _origErr = console.error;
    console.log = (...a) => { _log.push(a.map(String).join(' ')); _origLog(...a); };
    console.error = (...a) => { _err.push(a.map(String).join(' ')); _origErr(...a); };
    window.onerror = (msg,src,l,c,e) => { _err.push(e?e.toString():msg); };
    try {
      ${code}
    } catch(e) { _err.push(e.toString()); }
    parent.postMessage({stdout: _log.join('\\n'), stderr: _err.join('\\n'), runtime_ms: 0, exit_code: _err.length?1:0}, '*');
    <\/script></body></html>`;

    window.addEventListener('message', function handleMsg(e) {
      window.removeEventListener('message', handleMsg);
      showOutput(e.data);
      runBtn.textContent = '\u25B6 RUN CODE'; runBtn.classList.remove('running');
    });

    document.getElementById('idePreview').srcdoc = html;
    switchIDETab('output');
    return;
  }

  // ── Server-executed languages: send code + stdin to server ──
  try {
    const stdinEl = document.getElementById('ideStdin');
    const stdin   = stdinEl ? stdinEl.value : '';
    const res  = await fetch('/run-code', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({language: lang, code: code, stdin: stdin})
    });
    const data = await res.json();
    showOutput(data);
    switchIDETab('output');
  } catch(e) {
    showOutput({stdout:'', stderr:'Cannot reach execution server.', exit_code:1, runtime_ms:0});
    switchIDETab('output');
  }
  runBtn.textContent = '\u25B6 RUN CODE';
  runBtn.classList.remove('running');
}

function showOutput(data) {
  const outEl     = document.getElementById('ideOutput');
  const stdoutEl  = document.getElementById('outStdout');
  const stderrEl  = document.getElementById('outStderr');
  const metaEl    = document.getElementById('outMeta');

  const status = data.exit_code === 0 ? 'OK' : 'ERROR';
  metaEl.textContent  = `EXIT ${data.exit_code} | ${data.runtime_ms}ms | ${status}`;
  stdoutEl.textContent = data.stdout || '';
  stderrEl.textContent = data.stderr || '';
  if (!data.stdout && !data.stderr) {
    stdoutEl.innerHTML = '<span class="out-empty">No output produced.</span>';
  }
  outEl.classList.add('on');
}

/* ── Parse LLM response for code blocks and render IDE-style ── */
function renderCodeBlocks(text) {
  // Match ```lang\ncode\n``` patterns
  const codeBlockRe = /```(\w*)\n?([\s\S]*?)```/g;
  let match, lastIndex = 0, html = '';
  let hasCode = false;

  while ((match = codeBlockRe.exec(text)) !== null) {
    hasCode = true;
    const before = text.slice(lastIndex, match.index);
    if (before) html += escapeHtml(before).replace(/\n/g, '<br>');
    const lang = match[1] || 'plaintext';
    const code = match[2].trimEnd();
    const escapedCode = escapeHtml(code);
    const blockId = 'cb_' + Date.now() + '_' + Math.random().toString(36).slice(2,7);
    html += `
      <div class="jarvis-code-block">
        <div class="jcb-header">
          <span class="jcb-lang">${lang.toUpperCase()}</span>
          <div class="jcb-btns">
            <button class="jcb-btn jcb-copy" onclick="copyBlock('${blockId}')">COPY</button>
            <button class="jcb-btn jcb-run"  onclick="openIDEFromBlock(getBlock('${blockId}'),'${lang}')">▶ RUN</button>
          </div>
        </div>
        <div class="jcb-code" id="${blockId}">
          <pre><code class="language-${lang}">${escapedCode}</code></pre>
        </div>
      </div>`;
    lastIndex = match.index + match[0].length;
  }

  if (!hasCode) return null; // No code blocks — plain text

  const after = text.slice(lastIndex);
  if (after.trim()) html += '<div style="margin-top:8px">' + escapeHtml(after).replace(/\n/g,'<br>') + '</div>';
  return html;
}

function escapeHtml(t) {
  return t.replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;').replace(/"/g,'&quot;');
}

function getBlock(id) {
  const el = document.getElementById(id);
  return el ? el.querySelector('code').textContent : '';
}

/* ── Clipboard copy with full fallback for all mobile browsers ── */
function copyBlock(id) {
  const code = getBlock(id);
  const btn  = document.querySelector(`[onclick="copyBlock('${id}')"]`);
  const done = () => {
    if (btn) { btn.textContent = '✓ COPIED'; btn.style.color = '#00ff88'; }
    showToast('CODE COPIED ✓', 'green');
    setTimeout(() => { if (btn) { btn.textContent = 'COPY'; btn.style.color = ''; } }, 1800);
  };
  const fail = () => {
    // Fallback: create hidden textarea, select and copy
    const ta = document.createElement('textarea');
    ta.value = code;
    ta.style.cssText = 'position:fixed;top:0;left:0;opacity:0;z-index:9999;';
    document.body.appendChild(ta);
    ta.focus();
    ta.select();
    try {
      const ok = document.execCommand('copy');
      if (ok) done(); else showToast('LONG PRESS CODE TO COPY', 'red');
    } catch(e) { showToast('LONG PRESS CODE TO COPY', 'red'); }
    document.body.removeChild(ta);
  };
  if (navigator.clipboard && navigator.clipboard.writeText) {
    navigator.clipboard.writeText(code).then(done).catch(fail);
  } else {
    fail();
  }
}

/* ── Copy the full JARVIS response text (plain, incl. code fences) ── */
function copyResponse() {
  const text = lastReplyText || rTxt.textContent || '';
  const btn  = document.getElementById('copyRespBtn');
  const done = () => {
    if (btn) { btn.textContent = '✓ COPIED'; btn.style.color = 'var(--green)'; btn.style.borderColor = 'var(--green)'; }
    showToast('RESPONSE COPIED ✓', 'green');
    setTimeout(() => { if (btn) { btn.textContent = '📋 COPY'; btn.style.color = ''; btn.style.borderColor = ''; } }, 1800);
  };
  const fail = () => {
    const ta = document.createElement('textarea');
    ta.value = text;
    ta.style.cssText = 'position:fixed;top:0;left:0;opacity:0;z-index:9999;';
    document.body.appendChild(ta);
    ta.focus();
    ta.select();
    try {
      const ok = document.execCommand('copy');
      if (ok) done(); else showToast('LONG PRESS TEXT TO COPY', 'red');
    } catch(e) { showToast('LONG PRESS TEXT TO COPY', 'red'); }
    document.body.removeChild(ta);
  };
  if (!text.trim()) { showToast('NOTHING TO COPY YET', 'red'); return; }
  if (navigator.clipboard && navigator.clipboard.writeText) {
    navigator.clipboard.writeText(text).then(done).catch(fail);
  } else {
    fail();
  }
}

/* ── Open IDE panel and scroll to it on mobile ── */
function openIDEFromBlock(code, lang) {
  openIDE(code, lang);
  // Scroll IDE panel into view on mobile
  setTimeout(() => {
    const panel = document.getElementById('codePanel');
    if (panel) panel.scrollIntoView({behavior: 'smooth'});
  }, 100);
}

/* ── Override typewrite to support code blocks ── */
let typewriteActive = false;
const _origTypewrite = typewrite;
typewrite = function(text) {
  typewriteActive = true;
  lastReplyText = text;
  if (twTimer) clearInterval(twTimer);
  const codeHtml = renderCodeBlocks(text);
  if (codeHtml) {
    // Render immediately with syntax highlighting
    rTxt.className = 'resp-txt';
    rTxt.innerHTML = codeHtml;
    // Apply highlight.js to all code blocks
    if (window.hljs) {
      rTxt.querySelectorAll('pre code').forEach(el => window.hljs.highlightElement(el));
    }
    actionBtn.className = 'action-btn';
    setTimeout(() => setState('idle'), 1000);
  } else {
    _origTypewrite(text);
  }
};
</script>
</body>
</html>
"""


# ══════════════════════════════════════════════════════════════════════════════
# SERVICE WORKER (JavaScript — served at /sw.js)
# ══════════════════════════════════════════════════════════════════════════════

SW_JS = """
const CACHE_NAME = 'jarvis-v4.4';
const STATIC_ASSETS = [
  '/offline',
  '/manifest.json',
];
// NOTE: '/' (index.html) is intentionally NOT pre-cached here. It changes
// often during active development, so it must always be network-first
// (see fetch handler below) or Chrome will keep serving a stale cached
// version of the whole app after every redeploy, even though Edge (with
// no prior cache) always gets the current version. That mismatch is what
// caused "Initialising systems..." to hang forever in Chrome only.

// Install — cache static assets
self.addEventListener('install', event => {
  event.waitUntil(
    caches.open(CACHE_NAME).then(cache => cache.addAll(STATIC_ASSETS))
  );
  self.skipWaiting();
});

// Activate — clean old caches
self.addEventListener('activate', event => {
  event.waitUntil(
    caches.keys().then(keys =>
      Promise.all(keys.filter(k => k !== CACHE_NAME).map(k => caches.delete(k)))
    )
  );
  self.clients.claim();
});

// Fetch — network-first for API and the HTML page, cache-first for static assets
self.addEventListener('fetch', event => {
  const url = new URL(event.request.url);

  // API calls — always network, no cache
  if (['/command', '/greet', '/status', '/pc-command'].includes(url.pathname)) {
    event.respondWith(
      fetch(event.request).catch(() =>
        new Response(JSON.stringify({reply: 'Offline — cannot reach neural core.', action: null}),
          {headers: {'Content-Type': 'application/json'}})
      )
    );
    return;
  }

  // Navigation / homepage — network-first, cache only as an offline fallback
  if (event.request.mode === 'navigate' || url.pathname === '/') {
    event.respondWith(
      fetch(event.request).then(res => {
        if (res && res.status === 200) {
          const copy = res.clone();
          caches.open(CACHE_NAME).then(cache => cache.put(event.request, copy));
        }
        return res;
      }).catch(() => caches.match(event.request).then(cached => cached || caches.match('/offline')))
    );
    return;
  }

  // Other static assets — cache-first
  event.respondWith(
    caches.match(event.request).then(cached => {
      if (cached) return cached;
      return fetch(event.request).then(res => {
        if (res && res.status === 200) {
          const copy = res.clone();
          caches.open(CACHE_NAME).then(cache => cache.put(event.request, copy));
        }
        return res;
      }).catch(() => caches.match('/offline'));
    })
  );
});
"""

# ══════════════════════════════════════════════════════════════════════════════
# OFFLINE PAGE
# ══════════════════════════════════════════════════════════════════════════════

OFFLINE_PAGE = """
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8"/>
  <meta name="viewport" content="width=device-width, initial-scale=1.0"/>
  <title>J.A.R.V.I.S. — Offline</title>
  <link href="https://fonts.googleapis.com/css2?family=Orbitron:wght@700;900&family=Rajdhani:wght@400;600&display=swap" rel="stylesheet"/>
  <style>
    body{margin:0;background:#000814;color:#90c8e8;font-family:'Rajdhani',sans-serif;
      display:flex;flex-direction:column;align-items:center;justify-content:center;
      min-height:100vh;text-align:center;padding:24px;}
    .title{font-family:'Orbitron',monospace;font-size:1.4rem;font-weight:900;
      color:#00d4ff;letter-spacing:6px;text-shadow:0 0 18px #00d4ff;margin-bottom:12px;}
    .sub{font-size:1rem;color:#1e3a5a;margin-bottom:32px;letter-spacing:2px;}
    .ring{width:120px;height:120px;border-radius:50%;border:2px solid rgba(0,212,255,0.3);
      display:flex;align-items:center;justify-content:center;margin:0 auto 28px;
      position:relative;animation:rb 3s ease-in-out infinite;}
    @keyframes rb{0%,100%{box-shadow:0 0 20px rgba(0,212,255,0.2);}50%{box-shadow:0 0 40px rgba(0,212,255,0.5);}}
    .core{width:50px;height:50px;border-radius:50%;background:radial-gradient(circle,#9aeeff,#00d4ff);
      box-shadow:0 0 20px #00d4ff,0 0 40px rgba(0,212,255,0.4);}
    .msg{font-size:0.85rem;color:#1e3a5a;line-height:1.8;max-width:280px;margin-bottom:24px;}
    .msg span{color:#90c8e8;}
    .btn{padding:12px 28px;background:none;border:1px solid rgba(0,212,255,0.35);
      color:#00d4ff;border-radius:8px;font-family:'Orbitron',monospace;
      font-size:0.65rem;letter-spacing:3px;cursor:pointer;transition:all 0.2s;text-decoration:none;display:inline-block;}
    .btn:active{background:rgba(0,212,255,0.1);}
  </style>
</head>
<body>
  <div class="title">J.A.R.V.I.S.</div>
  <div class="sub">OFFLINE MODE</div>
  <div class="ring"><div class="core"></div></div>
  <div class="msg">
    Neural link is <span>offline</span>.<br>
    Reconnect to re-enable full AI capabilities.<br><br>
    <span>Core systems are cached and ready.</span>
  </div>
  <a class="btn" href="/">RETRY CONNECTION</a>
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
    if   5 <= hour < 12: msg = "Good morning, sir. All systems are online and operational."
    elif 12 <= hour < 17: msg = "Good afternoon. Neural core is active and ready."
    elif 17 <= hour < 21: msg = "Good evening. JARVIS systems fully nominal."
    elif 21 <= hour < 24: msg = "Good night, sir. Working late again, I see. How may I assist?"
    else:                 msg = "It is past midnight, sir. May I ask why you are still awake?"
    return jsonify({"greeting": msg})

@app.route("/command", methods=["POST"])
def command():
    data = request.get_json(silent=True) or {}
    cmd  = data.get("command", "").strip()
    if not cmd:
        return jsonify({"reply": "I didn't receive a command.", "action": None})
    try:
        result = execute_command(cmd)
        return jsonify(result)
    except Exception as e:
        print(f"[ERROR] /command failed: {e}")
        return jsonify({"reply": f"Internal error: {e}", "action": None}), 500

@app.route("/pc-command", methods=["POST"])
def pc_command_relay():
    """
    Relay PC commands to local jarvis.py running on port 5001.
    This works when the phone and PC are on the same network.
    """
    data   = request.get_json(force=True)
    action = data.get("action", "")
    param  = data.get("param", "")
    result = relay_pc_command(action, param)
    return jsonify(result)

@app.route("/status")
def status():
    return jsonify({
        "status":  "online",
        "version": "4.1",
        "model":   GROQ_MODEL,
        "model_fast": GROQ_MODEL_FAST,
        "memory":  MEMORY_AVAILABLE,
        "docs":    len(memory_docs),
        "groq_ok": bool(GROQ_API_KEY),
    })

@app.route("/ping")
def ping():
    """Lightweight keepalive endpoint — call this to wake Render from sleep."""
    return jsonify({"pong": True, "time": datetime.datetime.now(IST).isoformat()})

@app.route("/health")
def health():
    """Health check for monitoring."""
    return jsonify({
        "ok":      True,
        "groq":    bool(GROQ_API_KEY),
        "memory":  MEMORY_AVAILABLE,
    })

@app.route("/debug-groq")
def debug_groq():
    """Test Groq API directly — returns raw status and error for diagnostics."""
    if not GROQ_API_KEY:
        return jsonify({"error": "GROQ_API_KEY not set"}), 400
    try:
        r = http_requests.post(
            GROQ_API_URL,
            headers={"Authorization": f"Bearer {GROQ_API_KEY}", "Content-Type": "application/json"},
            json={"model": GROQ_MODEL_FAST, "messages": [{"role": "user", "content": "say hi"}], "max_tokens": 10},
            timeout=10,
        )
        return jsonify({
            "http_status": r.status_code,
            "body":        r.json() if r.headers.get("content-type","").startswith("application/json") else r.text[:300],
            "model":       GROQ_MODEL_FAST,
        })
    except http_requests.Timeout:
        return jsonify({"error": "Groq API timed out after 10s", "model": GROQ_MODEL_FAST}), 504
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/sw.js")
def service_worker():
    return Response(SW_JS, mimetype="application/javascript",
                    headers={"Service-Worker-Allowed": "/"})

@app.route("/offline")
def offline():
    return render_template_string(OFFLINE_PAGE)

@app.route("/manifest.json")
def manifest():
    return jsonify({
        "name":             "JARVIS",
        "short_name":       "JARVIS",
        "description":      "Advanced AI Assistant — 24/7 voice-controlled online & offline",
        "start_url":        "/?pwa=1",
        "display":          "standalone",
        "background_color": "#000814",
        "theme_color":      "#00d4ff",
        "orientation":      "portrait",
        "icons": [
            {"src": "/icon-192.png", "sizes": "192x192", "type": "image/png", "purpose": "any maskable"},
            {"src": "/icon-512.png", "sizes": "512x512", "type": "image/png", "purpose": "any maskable"},
        ],
        "shortcuts": [
            {"name": "World News",  "short_name": "NEWS",    "url": "/?cmd=world+news",  "description": "Latest global headlines"},
            {"name": "Weather",     "short_name": "WEATHER", "url": "/?cmd=weather",     "description": "Current weather"},
            {"name": "Open Camera", "short_name": "CAMERA",  "url": "/?cmd=open+camera", "description": "Open device camera"},
        ],
        "categories": ["productivity", "utilities"],
        "lang": "en",
    })

@app.route("/icon-192.png")
@app.route("/icon-512.png")
def icon():
    """Serve a minimal arc reactor icon as PNG (base64 embedded fallback)."""
    # Minimal 1x1 transparent PNG as fallback — replace with real icon bytes if available
    png_1x1 = base64.b64decode(
        "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg=="
    )
    return Response(png_1x1, mimetype="image/png")

# ══════════════════════════════════════════════════════════════════════════════
# CODE EXECUTION ENGINE
# ══════════════════════════════════════════════════════════════════════════════

import subprocess
try:
    import resource as _resource_mod  # Unix only; gracefully skipped on Windows
except ImportError:
    _resource_mod = None
import signal
import tempfile
import shutil

import re as _re_mod

# Dangerous patterns blocked in user Python code
DANGEROUS_PATTERNS = [
    "import os",  "os.system", "os.popen", "subprocess",
    "shutil.rmtree", "open('/", 'open("/',
    "__import__", "eval(", "exec(",
    "socket", "urllib", "http", "requests",
    "sys.exit", "exit()", "quit()",
]

# ── Language registry ──────────────────────────────────────────────────────
# Each entry: file extension, whether it needs a compile step, the binary(ies)
# required (checked with shutil.which so we fail gracefully instead of crashing
# if a language runtime simply isn't installed on this server), and how to
# build the run/compile commands given file paths.
LANGUAGE_REGISTRY = {
    "python":     {"ext": "py",  "interpreter": [sys.executable]},
    "python3":    {"ext": "py",  "interpreter": [sys.executable]},
    "py":         {"ext": "py",  "interpreter": [sys.executable]},
    "bash":       {"ext": "sh",  "interpreter": ["bash"]},
    "sh":         {"ext": "sh",  "interpreter": ["bash"]},
    "shell":      {"ext": "sh",  "interpreter": ["bash"]},
    "node":       {"ext": "js",  "interpreter": ["node"]},
    "javascript": {"ext": "js",  "interpreter": ["node"]},
    "js":         {"ext": "js",  "interpreter": ["node"]},
    "ruby":       {"ext": "rb",  "interpreter": ["ruby"]},
    "rb":         {"ext": "rb",  "interpreter": ["ruby"]},
    "php":        {"ext": "php", "interpreter": ["php"]},
    "perl":       {"ext": "pl",  "interpreter": ["perl"]},
    "lua":        {"ext": "lua", "interpreter": ["lua"]},
    "go":         {"ext": "go",  "interpreter": ["go", "run"]},
    "c":          {"ext": "c",   "compiler": ["gcc"],  "compile_extra": ["-O2", "-lm"]},
    "cpp":        {"ext": "cpp", "compiler": ["g++"],  "compile_extra": ["-O2", "-std=c++17"]},
    "c++":        {"ext": "cpp", "compiler": ["g++"],  "compile_extra": ["-O2", "-std=c++17"]},
    "java":       {"ext": "java", "compiler": ["javac"], "runner": ["java"], "needs_classname": True},
}

def _extract_java_class_name(code: str) -> str:
    m = _re_mod.search(r'\bpublic\s+class\s+(\w+)', code)
    if m:
        return m.group(1)
    m = _re_mod.search(r'\bclass\s+(\w+)', code)
    return m.group(1) if m else "Main"

def run_code_safely(language: str, code: str, stdin: str = "") -> dict:
    """
    Execute code in a sandboxed subprocess. Supports interpreted languages
    (Python, Bash, Node, Ruby, PHP, Perl, Lua, Go) and compiled languages
    (C, C++, Java) when the relevant toolchain is present on the server.
    `stdin` (optional) is piped into the program's standard input, one line
    per input()/scanf()/Scanner call, in the order the program reads them.
    Returns: {stdout, stderr, exit_code, runtime_ms, blocked}
    """
    lang = language.lower().strip()
    # Normalize stdin: ensure it ends with a newline so the last input() line
    # is terminated properly, unless it's empty (no stdin needed).
    if stdin and not stdin.endswith("\n"):
        stdin += "\n"

    # ── Security: block dangerous patterns in Python ──
    if lang in ("python", "python3", "py"):
        code_lower = code.lower()
        for pattern in DANGEROUS_PATTERNS:
            if pattern.lower() in code_lower:
                return {
                    "stdout": "",
                    "stderr": f"[JARVIS SECURITY] Blocked pattern detected: '{pattern}'. "
                              "Dangerous operations are not permitted in the sandbox.",
                    "exit_code": 1,
                    "runtime_ms": 0,
                    "blocked": True,
                }

    cfg = LANGUAGE_REGISTRY.get(lang)
    if not cfg:
        return {
            "stdout": "",
            "stderr": f"Server-side execution not supported for '{language}' yet, sir. "
                      "JavaScript and HTML run directly in your browser instead.",
            "exit_code": 1,
            "runtime_ms": 0,
            "blocked": False,
        }

    start = time.time()
    workdir = tempfile.mkdtemp(prefix="jarvis_run_")
    try:
        # ── Compiled languages: compile first, then run the binary ──
        if "compiler" in cfg:
            compiler_bin = cfg["compiler"][0]
            if shutil.which(compiler_bin) is None:
                return {
                    "stdout": "",
                    "stderr": f"'{compiler_bin}' is not installed on this server, sir. "
                              f"{language.upper()} execution needs it added to the Render environment.",
                    "exit_code": 127, "runtime_ms": 0, "blocked": False,
                }

            if cfg.get("needs_classname"):
                classname = _extract_java_class_name(code)
                src_path  = os.path.join(workdir, f"{classname}.{cfg['ext']}")
            else:
                src_path  = os.path.join(workdir, f"program.{cfg['ext']}")
                out_path  = os.path.join(workdir, "program.out")

            with open(src_path, "w", encoding="utf-8") as f:
                f.write(code)

            if cfg.get("needs_classname"):
                compile_cmd = cfg["compiler"] + [src_path]
            else:
                compile_cmd = cfg["compiler"] + [src_path] + cfg.get("compile_extra", []) + ["-o", out_path]

            compile_res = subprocess.run(
                compile_cmd, capture_output=True, text=True, timeout=15, cwd=workdir,
            )
            if compile_res.returncode != 0:
                return {
                    "stdout": "",
                    "stderr": "[COMPILE ERROR]\n" + compile_res.stderr[:10_000],
                    "exit_code": compile_res.returncode,
                    "runtime_ms": int((time.time() - start) * 1000),
                    "blocked": False,
                }

            run_cmd = (cfg["runner"] + [classname]) if cfg.get("needs_classname") else [out_path]
            run_start = time.time()
            result = subprocess.run(
                run_cmd, input=stdin, capture_output=True, text=True, timeout=10, cwd=workdir,
            )
            runtime_ms = int((time.time() - run_start) * 1000)

        # ── Interpreted languages ──
        else:
            interp_bin = cfg["interpreter"][0]
            if shutil.which(interp_bin) is None:
                return {
                    "stdout": "",
                    "stderr": f"'{interp_bin}' is not installed on this server, sir. "
                              f"{language.upper()} execution needs it added to the Render environment.",
                    "exit_code": 127, "runtime_ms": 0, "blocked": False,
                }
            src_path = os.path.join(workdir, f"program.{cfg['ext']}")
            with open(src_path, "w", encoding="utf-8") as f:
                f.write(code)
            cmd = cfg["interpreter"] + [src_path]
            result = subprocess.run(
                cmd, input=stdin, capture_output=True, text=True, timeout=10, cwd=workdir,
            )
            runtime_ms = int((time.time() - start) * 1000)

        return {
            "stdout":     result.stdout[:50_000],
            "stderr":     result.stderr[:10_000],
            "exit_code":  result.returncode,
            "runtime_ms": runtime_ms,
            "blocked":    False,
        }
    except subprocess.TimeoutExpired:
        return {
            "stdout": "", "stderr": "[TIMEOUT] Code exceeded the execution time limit.",
            "exit_code": 124, "runtime_ms": int((time.time() - start) * 1000), "blocked": False,
        }
    except FileNotFoundError:
        return {
            "stdout": "", "stderr": f"Interpreter/compiler not found for '{language}' on this server.",
            "exit_code": 127, "runtime_ms": 0, "blocked": False,
        }
    except Exception as e:
        return {
            "stdout": "", "stderr": f"Execution error: {e}",
            "exit_code": 1, "runtime_ms": 0, "blocked": False,
        }
    finally:
        shutil.rmtree(workdir, ignore_errors=True)


@app.route("/run-code", methods=["POST"])
def run_code_endpoint():
    data     = request.get_json(force=True)
    language = data.get("language", "python").strip()
    code     = data.get("code", "").strip()
    stdin    = data.get("stdin", "") or ""
    if not code:
        return jsonify({"stdout": "", "stderr": "No code provided.", "exit_code": 1, "runtime_ms": 0})
    result = run_code_safely(language, code, stdin)
    return jsonify(result)


# ══════════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("+-----------------------------------------------------+")
    print("|  J.A.R.V.I.S. Cloud Edition  v4.0                  |")
    print("+-----------------------------------------------------+")
    if not GROQ_API_KEY:
        print("|  WARNING: GROQ_API_KEY not set!                    |")
        print("|  Get free key: https://console.groq.com            |")
    else:
        print("|  Groq API     : Connected [OK]                     |")
    print(f"|  Running on   : http://0.0.0.0:{PORT}                |")
    print("|  PWA           : Enabled (Service Worker)            |")
    print("|  Code Engine   : Python sandbox + JS browser         |")
    print("|  PC Remote     : localhost:5001 relay ready          |")
    print("+-----------------------------------------------------+")
    app.run(host="0.0.0.0", port=PORT, debug=False, use_reloader=False)