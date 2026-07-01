"""
╔══════════════════════════════════════════════════════╗
║          JARVIS - SUPER UPGRADED AI ASSISTANT        ║
║  Features:                                           ║
║   • Gemini 2.0 Flash-Lite — full multi-turn memory   ║
║   • Python code writer + executor                    ║
║   • Live web search (DuckDuckGo)                     ║
║   • System info: battery, CPU, RAM, disk             ║
║   • App launcher & file manager                      ║
║   • Persistent JSON long-term memory                 ║
║   • Pure Python wake-word (no native DLLs)           ║
╚══════════════════════════════════════════════════════╝
"""

import pyttsx3
import speech_recognition as sr
import os
import sys
import webbrowser
import datetime
import time
import wikipedia
import requests
import json
import hashlib
from google import genai
from google.genai import types

# ── Add script's folder to path so tools.py is importable ──────────────────
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from tools import search_web, run_python_code, get_system_info, open_app, manage_files

# ── Import API keys ─────────────────────────────────────────────────────────
from apikey import GEMINI_API_KEY

# ── CONFIGURATION ───────────────────────────────────────────────────────────
WEATHER_API_KEY  = "YOUR_OPENWEATHER_KEY"
GEMINI_MODEL     = "gemini-2.0-flash-lite"
MEMORY_RESULTS   = 3      # How many long-term memories to inject per query
MAX_HISTORY      = 20     # Max conversation turns kept in RAM

SYSTEM_PROMPT = """You are JARVIS, an elite AI assistant modelled after the one from Iron Man.
Rules:
- Always address the user as "Sir"
- Be concise, witty, and sophisticated
- When you receive TOOL RESULT: blocks, summarise them naturally — do NOT repeat raw data verbatim
- When asked to write or fix Python code, output ONLY the code block, no explanation
- Never break character
"""

# ── AUDIO ENGINE ────────────────────────────────────────────────────────────
engine = pyttsx3.init('sapi5')
engine.setProperty('rate', 185)

# ── LONG-TERM MEMORY (JSON — no DLL dependencies) ──────────────────────────
_dir         = os.path.dirname(os.path.abspath(__file__))
_memory_file = os.path.join(_dir, "jarvis_memory.json")

def _load_memory() -> dict:
    """Load the memory store from disk."""
    if os.path.exists(_memory_file):
        try:
            with open(_memory_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception:
            pass
    return {}

def _save_memory(store: dict):
    """Persist the memory store to disk."""
    try:
        with open(_memory_file, 'w', encoding='utf-8') as f:
            json.dump(store, f, indent=2, ensure_ascii=False)
    except Exception:
        pass

# ── SHORT-TERM CONVERSATIONAL MEMORY ────────────────────────────────────────
# Gemini multi-turn format: list of Content objects
chat_history: list[dict] = []   # [{"role": "user"|"model", "parts": [{"text": "..."}]}]


# ════════════════════════════════════════════════════════════════════════════
#  CORE UTILITIES
# ════════════════════════════════════════════════════════════════════════════

def speak(text: str):
    """Print and speak text."""
    if text:
        print(f"\nJARVIS: {text}\n")
        engine.say(text)
        engine.runAndWait()


def take_command() -> str:
    """Listen via microphone and return recognised text (lowercase)."""
    r = sr.Recognizer()
    with sr.Microphone() as source:
        print("[Voice: Listening...]")
        r.adjust_for_ambient_noise(source, duration=0.5)
        r.pause_threshold = 2
        try:
            audio = r.listen(source, timeout=8, phrase_time_limit=10)
        except sr.WaitTimeoutError:
            return "none"
    try:
        print("[Voice: Recognizing...]")
        query = r.recognize_google(audio, language='en-IN')
        print(f"You (Voice): {query}")
        return query.lower()
    except Exception:
        return "none"


# ════════════════════════════════════════════════════════════════════════════
#  LONG-TERM MEMORY HELPERS
# ════════════════════════════════════════════════════════════════════════════

def _query_long_term_memory(query: str) -> str:
    """Retrieve relevant memories using keyword matching."""
    try:
        store = _load_memory()
        if not store:
            return ""
        query_words = set(query.lower().split())
        # Score each memory by how many query words it contains
        scored = []
        for doc in store.values():
            doc_words = set(doc.lower().split())
            score = len(query_words & doc_words)
            if score > 0:
                scored.append((score, doc))
        scored.sort(reverse=True)
        top = [doc for _, doc in scored[:MEMORY_RESULTS]]
        if top:
            return "Relevant memories:\n" + "\n".join(f"- {d}" for d in top)
    except Exception:
        pass
    return ""


def _save_to_long_term_memory(user_msg: str, jarvis_reply: str):
    """Store a condensed memory entry in the JSON store."""
    try:
        store = _load_memory()
        doc_id = hashlib.md5((user_msg + jarvis_reply).encode()).hexdigest()
        store[doc_id] = f"User said: {user_msg} | JARVIS replied: {jarvis_reply[:200]}"
        _save_memory(store)
    except Exception:
        pass


# ════════════════════════════════════════════════════════════════════════════
#  GEMINI BRAIN (with full multi-turn memory)
# ════════════════════════════════════════════════════════════════════════════

_gemini_client = genai.Client(api_key=GEMINI_API_KEY)

def ask_gemini(user_message: str) -> str:
    """
    Send a message to Gemini with full conversation history.
    Returns Gemini's reply as a string.
    """
    global chat_history

    # Inject relevant long-term memories into this turn's context
    memory_context = _query_long_term_memory(user_message)
    augmented_message = user_message
    if memory_context:
        augmented_message = f"{memory_context}\n\nUser's current message: {user_message}"

    # Build the contents list (history + new message)
    contents = []
    for turn in chat_history[-MAX_HISTORY:]:
        contents.append(types.Content(
            role=turn["role"],
            parts=[types.Part(text=turn["parts"][0]["text"])]
        ))
    contents.append(types.Content(
        role="user",
        parts=[types.Part(text=augmented_message)]
    ))

    # Retry up to 3 times on 429 rate-limit errors
    for attempt in range(3):
        try:
            response = _gemini_client.models.generate_content(
                model=GEMINI_MODEL,
                contents=contents,
                config=types.GenerateContentConfig(
                    system_instruction=SYSTEM_PROMPT,
                    temperature=0.7,
                )
            )
            reply = response.text.strip()

            # Update short-term history (store original message, not augmented)
            chat_history.append({"role": "user",  "parts": [{"text": user_message}]})
            chat_history.append({"role": "model", "parts": [{"text": reply}]})

            # Save to long-term memory
            _save_to_long_term_memory(user_message, reply)

            return reply

        except Exception as e:
            err_str = str(e)
            if "429" in err_str or "RESOURCE_EXHAUSTED" in err_str:
                wait = 60 * (attempt + 1)   # 60s, 120s, 180s
                print(f"[Rate limit hit. Waiting {wait}s before retrying... attempt {attempt+1}/3]")
                speak(f"I've hit my rate limit, Sir. Waiting {wait} seconds and retrying.")
                time.sleep(wait)
            else:
                print(f"[Gemini Error: {e}]")
                return f"I'm having trouble reaching my online brain, Sir. Error: {e}"

    return "I'm sorry Sir, I'm still rate-limited. Please try again in a few minutes."


# ════════════════════════════════════════════════════════════════════════════
#  TOOL DISPATCHER — Fast local keyword router (zero API calls)
# ════════════════════════════════════════════════════════════════════════════

# Keyword map: intent → list of trigger phrases
INTENT_KEYWORDS = {
    "EXIT":      ["exit", "quit", "bye", "goodbye", "shut yourself", "turn yourself off"],
    "SHUTDOWN":  ["shutdown the system", "power off", "shut down the computer", "turn off the laptop"],
    "RESTART":   ["restart the system", "reboot", "restart the computer"],
    "SLEEP":     ["sleep mode", "hibernate", "standby", "go to sleep"],
    "YOUTUBE":   ["open youtube", "play youtube", "youtube"],
    "WHATSAPP":  ["whatsapp", "send a message", "send message", "send whatsapp"],
    "WEATHER":   ["weather", "temperature outside", "how hot is", "will it rain"],
    "WIKIPEDIA": ["wikipedia", "wiki "],
    "SYSINFO":   ["battery", "cpu", "ram", "memory usage", "disk space", "system info",
                  "what time", "current time", "how much space", "processor"],
    "PYTHON":    ["python", "write a function", "write code", "code for", "fix this code",
                  "program to", "script to", "solve this code", "debug", "def ", "import "],
    "SEARCH":    ["search for", "search the web", "look up", "find online", "latest news",
                  "what is the latest", "current news", "who won", "price of"],
    "OPENAPP":   ["open ", "launch ", "start ", "run notepad", "run calculator"],
    "FILES":     ["my files", "list files", "read file", "create file", "open file",
                  "my documents", "list folder", "show files"],
}

def classify_command(command: str) -> str:
    """Route command using fast local keyword matching — zero API calls."""
    cmd = command.lower()
    for intent, keywords in INTENT_KEYWORDS.items():
        if any(kw in cmd for kw in keywords):
            print(f"[Router] Intent classified as: {intent}")
            return intent
    print("[Router] Intent classified as: CHAT")
    return "CHAT"


# ════════════════════════════════════════════════════════════════════════════
#  TASK EXECUTION
# ════════════════════════════════════════════════════════════════════════════

def execute_task(command: str):
    """Route the command to the correct handler."""
    global chat_history

    intent = classify_command(command)

    # ── EXIT / POWER ──────────────────────────────────────────────────────
    if intent == "EXIT":
        speak("Powering down internal processes. Goodbye, Sir.")
        os._exit(0)

    elif intent == "SHUTDOWN":
        speak("Initializing shutdown sequence. Goodbye, Sir.")
        os.system("shutdown /s /t 5")
        os._exit(0)

    elif intent == "RESTART":
        speak("Restarting all systems. I will be back shortly, Sir.")
        os.system("shutdown /r /t 5")

    elif intent == "SLEEP":
        speak("Entering standby mode, Sir.")
        os.system("shutdown.exe /h")

    # ── PYTHON CODE SOLVER ────────────────────────────────────────────────
    elif intent == "PYTHON":
        speak("Let me write and test that for you, Sir.")
        # Ask Gemini to write the code
        code_prompt = (
            f"Write a complete, runnable Python script for the following task. "
            f"Output ONLY the Python code, nothing else.\n\nTask: {command}"
        )
        code = ask_gemini(code_prompt)
        # Strip markdown fences if present
        if "```" in code:
            code = code.split("```python")[-1].split("```")[0].strip()
            if not code:
                code = code.split("```")[-2].strip()

        print(f"\n[Code generated]:\n{code}\n")
        speak("Executing the code now, Sir.")
        result = run_python_code(code)
        print(f"[Execution Result]: {result}")

        # Ask Gemini to summarise the result for voice
        summary = ask_gemini(
            f"I ran this Python code:\n{code}\n\nResult:\n{result}\n\n"
            f"Please summarise the result naturally for me as JARVIS would."
        )
        speak(summary)

    # ── WEB SEARCH ────────────────────────────────────────────────────────
    elif intent == "SEARCH":
        speak("Searching the web for you, Sir.")
        raw_results = search_web(command, max_results=5)
        print(f"[Web Results]:\n{raw_results}\n")
        # Ask Gemini to summarise
        summary = ask_gemini(
            f"I searched the web for: '{command}'\n\n"
            f"Here are the raw results:\n{raw_results}\n\n"
            f"Please summarise the key findings naturally as JARVIS would."
        )
        speak(summary)

    # ── SYSTEM INFO ───────────────────────────────────────────────────────
    elif intent == "SYSINFO":
        info = get_system_info()
        print(f"[System Info]:\n{info}")
        summary = ask_gemini(
            f"TOOL RESULT:\n{info}\n\n"
            f"User asked: '{command}'\n"
            f"Summarise this as JARVIS concisely."
        )
        speak(summary)

    # ── APP LAUNCHER ──────────────────────────────────────────────────────
    elif intent == "OPENAPP":
        result = open_app(command)
        print(f"[App Launcher]: {result}")
        speak(result)

    # ── FILE MANAGER ──────────────────────────────────────────────────────
    elif intent == "FILES":
        # Let Gemini parse what action/path is needed
        parse_prompt = (
            f"The user said: '{command}'\n"
            f"Respond with JSON only, format: {{\"action\": \"list|read|create\", \"path\": \"filename_or_folder\", \"content\": \"file content if creating else empty\"}}"
        )
        import json, re
        parsed_str = ask_gemini(parse_prompt)
        try:
            # Extract JSON from response
            json_match = re.search(r'\{.*\}', parsed_str, re.DOTALL)
            params = json.loads(json_match.group()) if json_match else {}
            result = manage_files(
                action=params.get("action", "list"),
                path=params.get("path", ""),
                content=params.get("content", "")
            )
        except Exception:
            result = manage_files("list")
        print(f"[File Manager]: {result}")
        speak(ask_gemini(f"TOOL RESULT:\n{result}\nSummarise briefly as JARVIS."))

    # ── WIKIPEDIA ─────────────────────────────────────────────────────────
    elif intent == "WIKIPEDIA":
        speak("Accessing Wikipedia databases, Sir.")
        try:
            topic = command.replace("wikipedia", "").replace("search", "").strip()
            summary = wikipedia.summary(topic, sentences=3)
            speak(ask_gemini(f"Summarise this Wikipedia content naturally:\n{summary}"))
        except Exception:
            speak("I'm sorry Sir, I couldn't find relevant data on Wikipedia.")

    # ── WEATHER ───────────────────────────────────────────────────────────
    elif intent == "WEATHER":
        try:
            city = command.split("in")[-1].strip() if "in" in command else "Chennai"
            url = (f"http://api.openweathermap.org/data/2.5/weather"
                   f"?q={city}&appid={WEATHER_API_KEY}&units=metric")
            data = requests.get(url, timeout=5).json()
            temp   = data['main']['temp']
            desc   = data['weather'][0]['description']
            humid  = data['main']['humidity']
            info   = f"Temperature: {temp}°C, Conditions: {desc}, Humidity: {humid}%"
            speak(ask_gemini(f"TOOL RESULT for {city} weather:\n{info}\nDeliver as JARVIS."))
        except Exception:
            speak("Weather servers are unreachable, Sir.")

    # ── WHATSAPP ──────────────────────────────────────────────────────────
    elif intent == "WHATSAPP":
        try:
            import pywhatkit as kit
            speak("Who is the recipient, Sir? Please provide the phone number.")
            number = take_command().replace(" ", "")
            speak("And the message, Sir?")
            msg = take_command()
            if msg not in ("none", ""):
                speak("Sending your message, Sir.")
                kit.sendwhatmsg_instantly(f"+{number}", msg, wait_time=15)
                speak("Message sent successfully, Sir.")
        except ImportError:
            speak("WhatsApp module is offline, Sir. Internet connection required.")

    # ── YOUTUBE ───────────────────────────────────────────────────────────
    elif intent == "YOUTUBE":
        speak("Opening YouTube, Sir.")
        webbrowser.open("https://www.youtube.com")

    # ── GENERAL CHAT (default) ────────────────────────────────────────────
    else:  # CHAT
        reply = ask_gemini(command)
        speak(reply)


# ════════════════════════════════════════════════════════════════════════════
#  MAIN WAKE-WORD LOOP
# ════════════════════════════════════════════════════════════════════════════

def _listen_for_wake_word() -> bool:
    """
    Listen continuously until 'jarvis' is detected in speech.
    Returns True when wake word heard, False on unrecoverable error.
    Uses only PyAudio + Google Speech API — zero native DLLs.
    """
    r = sr.Recognizer()
    r.energy_threshold = 3000       # Adjust if too sensitive / not sensitive enough
    r.dynamic_energy_threshold = True
    r.pause_threshold = 0.8

    with sr.Microphone() as source:
        print("[Listening for wake word: 'Jarvis'...]")
        r.adjust_for_ambient_noise(source, duration=1)
        while True:
            try:
                audio = r.listen(source, timeout=None, phrase_time_limit=4)
                text  = r.recognize_google(audio, language='en-IN').lower()
                print(f"[Heard]: {text}")
                if "jarvis" in text:
                    return True
            except sr.UnknownValueError:
                pass    # Nothing heard — keep listening
            except sr.RequestError as e:
                print(f"[Speech API error]: {e}")
                time.sleep(2)
            except Exception as e:
                print(f"[Wake-word listener error]: {e}")
                return False


def run_jarvis():
    speak("All systems operational. Jarvis online. Ready for your commands, Sir.")
    print("\n[Say 'Wake word' to activate]\n")

    while True:
        try:
            woken = _listen_for_wake_word()
            if not woken:
                print("[Wake-word listener stopped. Restarting...]")
                time.sleep(2)
                continue

            print("\n[Wake word detected!]")
            speak("At your service, Sir.")
            time.sleep(0.3)

            command = take_command()
            if command and command != "none":
                execute_task(command)

        except KeyboardInterrupt:
            speak("Powering down. Goodbye, Sir.")
            break
        except Exception as e:
            print(f"[Loop Error]: {e}")
            time.sleep(1)


# ════════════════════════════════════════════════════════════════════════════
#  STARTUP
# ════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    try:
        run_jarvis()
    except KeyboardInterrupt:
        print("\n[Force Quit]")
        os._exit(0)
