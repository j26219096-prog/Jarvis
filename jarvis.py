"""
╔══════════════════════════════════════════════════════════════════════════════╗
║         J.A.R.V.I.S. — Just A Rather Very Intelligent System                ║
║         Fully Local, Voice-Activated AI Assistant                            ║
║         Author  : Built for Jawahar                                          ║
║         Engine  : Ollama (llama3.2:1b) · ChromaDB · pyttsx3 · SpeechRecog   ║
╚══════════════════════════════════════════════════════════════════════════════╝

DEPENDENCIES (install before running):
    pip install pyttsx3 SpeechRecognition ollama chromadb requests wikipedia

OPTIONAL (gracefully disabled if missing / offline):
    pip install pywhatkit

EXTERNAL SERVICES:
    - Ollama must be running locally: https://ollama.com/
      Pull the model first:  ollama pull llama3.2:1b
    - NewsAPI key (free): https://newsapi.org/
      Set YOUR_NEWSAPI_KEY below.
"""

# ─────────────────────────────────────────────────────────────────────────────
# STANDARD LIBRARY IMPORTS
# ─────────────────────────────────────────────────────────────────────────────
import os                      # OS-level commands (shutdown, restart, etc.)
import sys                     # System utilities
import time                    # Sleep / timing

# Force UTF-8 output on Windows (prevents cp1252 UnicodeEncodeError)
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
import datetime                # Time-of-day checks for wish_me()
import webbrowser              # Opening URLs in the default browser
import threading               # Non-blocking TTS (optional future use)

# ─────────────────────────────────────────────────────────────────────────────
# THIRD-PARTY IMPORTS — core
# ─────────────────────────────────────────────────────────────────────────────
import pyttsx3                 # Text-to-Speech (fully offline)
import speech_recognition as sr  # Voice input
import requests                # HTTP requests (NewsAPI)
import wikipedia               # Wikipedia summaries

# ─────────────────────────────────────────────────────────────────────────────
# THIRD-PARTY IMPORTS — AI / Memory
# ─────────────────────────────────────────────────────────────────────────────
import ollama                  # Local LLM via Ollama
import chromadb                # Vector DB for RAG memory bank

# ─────────────────────────────────────────────────────────────────────────────
# OPTIONAL IMPORT: pywhatkit (WhatsApp automation)
# Wrapped in try/except so the script runs even if the package is absent
# or if the machine is offline.
# ─────────────────────────────────────────────────────────────────────────────
try:
    import pywhatkit as kit
    WHATSAPP_AVAILABLE = True
    print("[JARVIS BOOT] pywhatkit loaded — WhatsApp features ENABLED.")
except ImportError:
    WHATSAPP_AVAILABLE = False
    print("[JARVIS BOOT] pywhatkit not found — WhatsApp features DISABLED.")
except Exception as _e:
    WHATSAPP_AVAILABLE = False
    print(f"[JARVIS BOOT] pywhatkit failed to load ({_e}) — WhatsApp DISABLED.")


# ══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ══════════════════════════════════════════════════════════════════════════════

# -- NewsAPI --
# Sign up free at https://newsapi.org/register and paste your key here.
NEWSAPI_KEY = "YOUR_NEWSAPI_KEY"

# -- Ollama model --
OLLAMA_MODEL = "llama3.2:1b"

# -- Short-term conversational memory (rolling window of last N exchanges) --
MAX_HISTORY = 5   # keep last 5 user+assistant message pairs

# -- ChromaDB persistent storage path (same folder as this script) --
SCRIPT_DIR        = os.path.dirname(os.path.abspath(__file__))
CHROMA_DB_PATH    = os.path.join(SCRIPT_DIR, "jarvis_memory")
CHROMA_COLLECTION = "personal_knowledge"

# -- TTS speech rate --
TTS_RATE = 190


# ══════════════════════════════════════════════════════════════════════════════
# INITIALISE TEXT-TO-SPEECH ENGINE
# ══════════════════════════════════════════════════════════════════════════════

engine = pyttsx3.init()
engine.setProperty("rate", TTS_RATE)   # Words per minute

# ── Voice Selection — prefer a deep English male voice ────────────────────
# Priority list of known male voice name substrings (Windows / SAPI)
MALE_VOICE_KEYWORDS = [
    "david", "mark", "james", "george", "daniel", "thomas", "alex",
    "zira" "male", "en-us-guy", "en-gb-guy",
]

_voices = engine.getProperty("voices")
if _voices:
    chosen_voice = _voices[0]   # default fallback
    # Pass 1: look for a known male keyword in an English voice
    for v in _voices:
        vname = v.name.lower()
        vid   = v.id.lower() if v.id else ""
        if any(kw in vname or kw in vid for kw in MALE_VOICE_KEYWORDS):
            chosen_voice = v
            break
    # Pass 2: if still on index-0 (likely female), try any voice with 'male'
    if chosen_voice is _voices[0]:
        for v in _voices:
            if "male" in v.name.lower() or "male" in (v.id or "").lower():
                chosen_voice = v
                break
    engine.setProperty("voice", chosen_voice.id)
    print(f"[JARVIS BOOT] TTS Voice: {chosen_voice.name}")

# Lower pitch makes pyttsx3 sound deeper on Windows (0.5–2.0; default 1.0)
engine.setProperty("pitch", 0.8)

# ──────────────────────────────────────────────────────────────────────────────
def speak(text: str) -> None:
    """Convert text to speech and block until the audio finishes playing."""
    print(f"[JARVIS] {text}")
    engine.say(text)
    engine.runAndWait()


# ══════════════════════════════════════════════════════════════════════════════
# INITIALISE CHROMADB  (Persistent RAG Memory Bank)
# ══════════════════════════════════════════════════════════════════════════════

print(f"[JARVIS BOOT] Initialising ChromaDB at: {CHROMA_DB_PATH}")

try:
    chroma_client     = chromadb.PersistentClient(path=CHROMA_DB_PATH)
    memory_collection = chroma_client.get_or_create_collection(
        name=CHROMA_COLLECTION,
        metadata={"hnsw:space": "cosine"}
    )
    print(f"[JARVIS BOOT] Memory bank ready. "
          f"Documents stored: {memory_collection.count()}")
    MEMORY_AVAILABLE = True
except Exception as _e:
    print(f"[JARVIS BOOT] ChromaDB failed ({_e}) — running without long-term memory.")
    memory_collection = None
    MEMORY_AVAILABLE  = False


# ══════════════════════════════════════════════════════════════════════════════
# CONVERSATIONAL MEMORY  (short-term chat history for Ollama)
# ══════════════════════════════════════════════════════════════════════════════

# Each entry is a dict: {"role": "user"|"assistant", "content": "..."}
chat_history: list[dict] = []

def update_history(role: str, content: str) -> None:
    """
    Append a message to chat_history and trim to the last MAX_HISTORY pairs.
    A 'pair' consists of one user turn + one assistant turn.
    We keep at most (MAX_HISTORY * 2) individual messages in total.
    """
    chat_history.append({"role": role, "content": content})
    max_messages = MAX_HISTORY * 2          # 5 pairs = 10 individual messages
    if len(chat_history) > max_messages:
        # Drop the oldest messages (from the front), keep the most recent
        del chat_history[:len(chat_history) - max_messages]


# ══════════════════════════════════════════════════════════════════════════════
# RAG MEMORY HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def query_memory(user_text: str, n_results: int = 3) -> str:
    """
    Query the ChromaDB collection for context relevant to user_text.
    Returns a formatted string of retrieved documents, or an empty string
    if nothing useful is found or the memory bank is unavailable.
    """
    if not MEMORY_AVAILABLE or memory_collection is None:
        return ""
    if memory_collection.count() == 0:
        return ""

    try:
        results = memory_collection.query(
            query_texts=[user_text],
            n_results=min(n_results, memory_collection.count())
        )
        docs = results.get("documents", [[]])[0]
        if not docs:
            return ""
        # Format the retrieved snippets for injection into the system prompt
        context_block = "\n\n".join(
            f"[Memory {i+1}]: {doc}" for i, doc in enumerate(docs)
        )
        return context_block
    except Exception as e:
        print(f"[MEMORY] Query error: {e}")
        return ""


def add_to_memory(text: str, doc_id: str | None = None) -> None:
    """
    Add a piece of text to the persistent ChromaDB memory bank.
    If no doc_id is supplied, a timestamp-based ID is generated.
    """
    if not MEMORY_AVAILABLE or memory_collection is None:
        speak("Sorry, the memory bank is unavailable right now.")
        return
    try:
        _id = doc_id or f"mem_{int(time.time())}"
        memory_collection.add(documents=[text], ids=[_id])
        speak(f"Got it. I've saved that to memory.")
        print(f"[MEMORY] Stored document ID: {_id}")
    except Exception as e:
        print(f"[MEMORY] Failed to store: {e}")
        speak("I encountered an error while saving to memory.")


# ══════════════════════════════════════════════════════════════════════════════
# OFFLINE LOCAL LLM — OLLAMA
# ══════════════════════════════════════════════════════════════════════════════

# System prompt base — defines JARVIS personality
BASE_SYSTEM_PROMPT = (
    "You are J.A.R.V.I.S., an advanced, witty, and highly capable AI assistant. "
    "You speak in a calm, professional, slightly British tone with occasional dry humour, "
    "inspired by Tony Stark's Friday. "
    "Keep responses concise (1–3 sentences unless asked for more). "
    "You are running fully locally on the user's machine — no cloud, no internet required."
)


def ask_ollama(user_text: str) -> str:
    """
    Send the user's query to the local Ollama LLM (llama3.2:1b).

    Pipeline:
      1. Query ChromaDB for relevant personal knowledge (RAG).
      2. If context is found, inject it into the system prompt dynamically.
      3. Append the user message to chat_history.
      4. Call ollama.chat() with the full rolling history.
      5. Append the assistant reply to chat_history.
      6. Return the assistant reply as a string.
    """

    # ── Step 1 & 2: RAG context injection ────────────────────────────────────
    memory_context = query_memory(user_text)

    if memory_context:
        # Dynamically augment the system prompt with retrieved knowledge
        system_prompt = (
            BASE_SYSTEM_PROMPT
            + "\n\nYou have access to the following personal knowledge about the user. "
            "Use it when relevant:\n\n"
            + memory_context
        )
        print("[MEMORY] Context injected into system prompt.")
    else:
        system_prompt = BASE_SYSTEM_PROMPT

    # ── Step 3: Record user message ───────────────────────────────────────────
    update_history("user", user_text)

    # ── Step 4: Build message list for Ollama ─────────────────────────────────
    # Prepend the (possibly augmented) system message, then the rolling history
    messages_to_send = [
        {"role": "system", "content": system_prompt}
    ] + chat_history   # chat_history already includes the current user turn

    try:
        response = ollama.chat(
            model=OLLAMA_MODEL,
            messages=messages_to_send
        )
        reply = response["message"]["content"].strip()
    except Exception as e:
        reply = f"I'm having trouble reaching my neural core. Error: {e}"
        print(f"[OLLAMA] Error: {e}")

    # ── Step 5: Record assistant reply ────────────────────────────────────────
    update_history("assistant", reply)

    return reply


# ══════════════════════════════════════════════════════════════════════════════
# VOICE RECOGNITION HELPERS
# ══════════════════════════════════════════════════════════════════════════════

# Shared recogniser instance (reused across calls for efficiency)
recogniser = sr.Recognizer()
recogniser.pause_threshold = 1     # Seconds of silence before end-of-speech
recogniser.dynamic_energy_threshold = True   # Auto-adjust for ambient noise


def listen_for_audio(timeout: int = 5, phrase_limit: int = 10) -> str | None:
    """
    Capture audio from the microphone and return the recognised text in lowercase,
    or None if recognition fails.

    Args:
        timeout     : Maximum seconds to wait for speech to start.
        phrase_limit: Maximum seconds to record a single phrase.

    Returns:
        Recognised text (str) or None on failure / silence.
    """
    with sr.Microphone() as source:
        # Brief ambient noise calibration on each call
        recogniser.adjust_for_ambient_noise(source, duration=0.5)
        try:
            print("[MIC] Listening …")
            audio = recogniser.listen(
                source,
                timeout=timeout,
                phrase_time_limit=phrase_limit
            )
        except sr.WaitTimeoutError:
            # No speech detected within timeout window — not an error
            return None

    # Attempt speech-to-text using Google's online STT API.
    # Note: If completely offline, swap this for recogniser.recognize_sphinx(audio)
    # after installing: pip install pocketsphinx
    try:
        text = recogniser.recognize_google(audio).lower()
        print(f"[HEARD] {text}")
        return text
    except sr.UnknownValueError:
        # Audio captured but unintelligible
        return None
    except sr.RequestError as e:
        # Network / API issue
        print(f"[STT] Google STT error: {e}")
        return None


def listen_for_wake_word() -> bool:
    """
    Listen continuously for the wake word "jarvis".
    Uses a tight loop with a 100 ms pause between failed attempts to reduce
    CPU load.  Returns True as soon as the wake word is detected.
    """
    print("[WAKE] Waiting for wake word: 'jarvis' …")
    while True:
        text = listen_for_audio(timeout=5, phrase_limit=4)
        if text and "jarvis" in text:
            return True
        # Brief pause to prevent CPU spinning when no audio is detected
        time.sleep(0.1)


def listen_for_command() -> str | None:
    """
    After the wake word is confirmed, listen for the actual user command.
    Returns the recognised command string or None.
    """
    speak("At your service.")
    print("[CMD] Listening for command …")
    return listen_for_audio(timeout=7, phrase_limit=15)


# ══════════════════════════════════════════════════════════════════════════════
# WISH ME — Context-Aware Greeting
# ══════════════════════════════════════════════════════════════════════════════

def wish_me() -> None:
    """
    Greet the user based on the current time of day.
    Special behaviour: if it is late at night (midnight–4 AM), JARVIS will
    proactively inquire why the user is still awake — mimicking the
    'Friday' personality trait from the MCU.
    """
    hour = datetime.datetime.now().hour

    if 5 <= hour < 12:
        greeting = "Good morning, sir. Systems are fully operational."
    elif 12 <= hour < 17:
        greeting = "Good afternoon. All systems nominal."
    elif 17 <= hour < 21:
        greeting = "Good evening. Ready when you are."
    elif 21 <= hour < 24:
        greeting = "Good night, sir. Working late again, I see."
    else:
        # Midnight → 4:59 AM — Friday-style concerned nudge
        greeting = (
            "It is past midnight. "
            "May I ask why you are still awake at this hour? "
            "Shall I reduce the display brightness and activate a focus-mode playlist?"
        )

    speak(greeting)


# ══════════════════════════════════════════════════════════════════════════════
# WORLD MONITOR — Global News + Threat Dashboard
# ══════════════════════════════════════════════════════════════════════════════

def world_monitor() -> None:
    """
    Pull top global headlines from NewsAPI, read them aloud, then open
    liveuamap.com as a live visual situational-awareness dashboard.

    Triggered by: "happening around the world" | "world news"
    """

    # ── Preamble ─────────────────────────────────────────────────────────────
    speak("Give me a second, let me pull up the global feeds.")
    print("[WORLD MONITOR] Fetching headlines from NewsAPI …")

    # ── Fetch headlines ───────────────────────────────────────────────────────
    if NEWSAPI_KEY == "YOUR_NEWSAPI_KEY":
        # Key not configured — graceful fallback
        speak(
            "The NewsAPI key has not been configured. "
            "Please add your key to the script and try again. "
            "Opening the live map instead."
        )
    else:
        try:
            url = (
                "https://newsapi.org/v2/top-headlines"
                f"?language=en&pageSize=2&apiKey={NEWSAPI_KEY}"
            )
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            data = response.json()

            articles = data.get("articles", [])
            if articles:
                speak("Here are the top global headlines right now:")
                for i, article in enumerate(articles[:2], start=1):
                    headline = article.get("title", "No title available.")
                    # Strip source tag often appended after ' - '
                    headline = headline.split(" - ")[0].strip()
                    speak(f"Headline {i}: {headline}")
            else:
                speak("No headlines were returned from the feed at this time.")

        except requests.exceptions.ConnectionError:
            speak("I cannot reach the news feed. Check your internet connection.")
        except requests.exceptions.Timeout:
            speak("The news feed timed out. Moving on.")
        except Exception as e:
            speak("I encountered an error while retrieving the news feed.")
            print(f"[WORLD MONITOR] Error: {e}")

    # ── Open visual threat-assessment dashboard ───────────────────────────────
    speak("Opening the live situational-awareness map now.")
    webbrowser.open("https://liveuamap.com/")


# ══════════════════════════════════════════════════════════════════════════════
# OS CONTROLS
# ══════════════════════════════════════════════════════════════════════════════

def shutdown_system() -> None:
    """Schedule a system shutdown after a 5-second warning."""
    speak("Initiating system shutdown sequence. Goodbye, sir.")
    time.sleep(2)
    # Windows: shutdown /s /t 5   |  Linux/Mac: shutdown -h +1
    if sys.platform.startswith("win"):
        os.system("shutdown /s /t 5")
    else:
        os.system("shutdown -h +1")


def restart_system() -> None:
    """Restart the system after a short warning."""
    speak("Restarting the system. Back in a moment.")
    time.sleep(2)
    if sys.platform.startswith("win"):
        os.system("shutdown /r /t 5")
    else:
        os.system("shutdown -r +1")


def hibernate_system() -> None:
    """Put the system into hibernate / suspend mode."""
    speak("Entering hibernation mode. Dormant until you wake me.")
    time.sleep(2)
    if sys.platform.startswith("win"):
        os.system("shutdown /h")
    else:
        os.system("systemctl hibernate")


# ══════════════════════════════════════════════════════════════════════════════
# WIKIPEDIA LOOKUP
# ══════════════════════════════════════════════════════════════════════════════

def search_wikipedia(query: str) -> None:
    """
    Fetch a short Wikipedia summary for the given query and read it aloud.
    Handles disambiguation and network errors gracefully.
    """
    speak(f"Searching Wikipedia for {query}.")
    try:
        # sentences=2 keeps the response concise for voice output
        summary = wikipedia.summary(query, sentences=2, auto_suggest=True)
        speak(summary)
    except wikipedia.exceptions.DisambiguationError as e:
        speak(
            f"There are multiple results for that query. "
            f"Could you be more specific? For example: {e.options[0]}."
        )
    except wikipedia.exceptions.PageError:
        speak("I could not find a Wikipedia page for that topic.")
    except Exception as e:
        speak("I ran into an error while searching Wikipedia.")
        print(f"[WIKI] Error: {e}")


# ══════════════════════════════════════════════════════════════════════════════
# WHATSAPP
# ══════════════════════════════════════════════════════════════════════════════

def send_whatsapp(number: str, message: str, hour: int, minute: int) -> None:
    """
    Schedule a WhatsApp message using pywhatkit.
    All args must be provided; basic validation is performed.

    Args:
        number  : Recipient's number in international format, e.g. "+919876543210".
        message : Text of the message.
        hour    : Scheduled hour (24-hour clock).
        minute  : Scheduled minute.
    """
    if not WHATSAPP_AVAILABLE:
        speak(
            "WhatsApp integration is currently unavailable. "
            "Please install pywhatkit and ensure you are online."
        )
        return
    try:
        speak(f"Scheduling WhatsApp message for {hour}:{minute:02d}.")
        kit.sendwhatmsg(number, message, hour, minute)
    except Exception as e:
        speak("I encountered an error while trying to send the WhatsApp message.")
        print(f"[WHATSAPP] Error: {e}")


# ══════════════════════════════════════════════════════════════════════════════
# TASK EXECUTOR — Command Router
# ══════════════════════════════════════════════════════════════════════════════

def execute_command(command: str) -> None:
    """
    Parse the recognised command string and route to the appropriate handler.
    Falls through to the Ollama LLM for anything not matched by a keyword.

    Command keywords are evaluated in priority order:
        1. System controls  (shutdown / restart / hibernate)
        2. World Monitor    (world news / happening around the world)
        3. YouTube          (open youtube)
        4. Wikipedia        (wikipedia / search / tell me about)
        5. WhatsApp         (send whatsapp / message)
        6. Memory store     (remember this / save to memory)
        7. Ollama LLM       (fallback for all unrecognised queries)
    """

    command = command.strip().lower()
    print(f"[COMMAND] Processing: '{command}'")

    # ── 1. OS Controls ────────────────────────────────────────────────────────
    if any(kw in command for kw in ["shutdown", "power off", "turn off computer"]):
        shutdown_system()
        return

    if any(kw in command for kw in ["restart", "reboot"]):
        restart_system()
        return

    if "hibernate" in command or "sleep mode" in command:
        hibernate_system()
        return

    # ── 2. World Monitor ──────────────────────────────────────────────────────
    if any(kw in command for kw in [
        "happening around the world",
        "world news",
        "global news",
        "news update",
        "what's in the news"
    ]):
        world_monitor()
        return

    # ── 3. YouTube ────────────────────────────────────────────────────────────
    if "youtube" in command:
        speak("Opening YouTube.")
        webbrowser.open("https://www.youtube.com")
        return

    # ── 4. Wikipedia ──────────────────────────────────────────────────────────
    if any(kw in command for kw in ["wikipedia", "search", "tell me about", "who is", "what is"]):
        # Extract the search query by removing leading filler phrases
        query = command
        for filler in [
            "wikipedia", "search for", "search", "tell me about",
            "who is", "what is", "look up", "jarvis"
        ]:
            query = query.replace(filler, "")
        query = query.strip()
        if query:
            search_wikipedia(query)
        else:
            speak("What would you like me to search for?")
        return

    # ── 5. WhatsApp ───────────────────────────────────────────────────────────
    if "whatsapp" in command or ("send" in command and "message" in command):
        # Example usage: "send a whatsapp message to +919876543210 saying hello"
        # Full NLP parsing would require more infrastructure; we demonstrate
        # a structured call here for extensibility.
        if not WHATSAPP_AVAILABLE:
            speak("WhatsApp integration is disabled on this system.")
        else:
            speak(
                "WhatsApp messaging requires a phone number and message. "
                "Please call the send_whatsapp function directly with the required parameters."
            )
        return

    # ── 6. Memory — Save a note ───────────────────────────────────────────────
    if any(kw in command for kw in ["remember this", "save to memory", "note this"]):
        # Extract what to remember: everything after the trigger phrase
        note = command
        for trigger in ["remember this", "save to memory", "note this", "jarvis"]:
            note = note.replace(trigger, "")
        note = note.strip(": ").strip()
        if note:
            add_to_memory(note)
        else:
            speak("What would you like me to remember?")
        return

    # ── 7. Exit / Stop ────────────────────────────────────────────────────────
    if any(kw in command for kw in ["goodbye", "exit", "quit", "shut yourself down", "go to sleep"]):
        speak("Powering down. Have a good one, sir.")
        sys.exit(0)

    # ── 8. Ollama LLM Fallback ────────────────────────────────────────────────
    print("[OLLAMA] No keyword matched — routing to local LLM.")
    reply = ask_ollama(command)
    speak(reply)


# ══════════════════════════════════════════════════════════════════════════════
# MAIN ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    """
    Application entry point.

    Boot sequence:
      1. Print ASCII banner.
      2. Greet the user with a time-aware wish_me() call.
      3. Enter the infinite wake-word detection loop.
         ┌─ listen for "jarvis"
         │    └─ detected → say "At your service"
         │         └─ listen for command
         │              └─ execute_command()
         └─ repeat
    """

    # ── ASCII Banner ─────────────────────────────────────────────────────────
    print("""
    +----------------------------------------------------------+
    |   J.A.R.V.I.S.  --  Fully Local AI Assistant           |
    |   Engine  : Ollama llama3.2:1b                         |
    |   Memory  : ChromaDB (jarvis_memory/)                  |
    |   TTS     : pyttsx3 @ 190 wpm                          |
    |   STT     : SpeechRecognition (Google)                 |
    +----------------------------------------------------------+
    """)

    # ── Step 1: Time-aware greeting ───────────────────────────────────────────
    wish_me()

    # ── Step 2: Tell the user how to activate JARVIS ──────────────────────────
    speak("Say 'Jarvis' at any time to get my attention.")

    # ── Step 3: Infinite wake-word + command loop ─────────────────────────────
    print("\n[MAIN LOOP] Entering wake-word detection loop. Say 'Jarvis' to begin.\n")

    while True:
        try:
            # ── Phase A: Wait for the wake word ──────────────────────────────
            wake_detected = listen_for_wake_word()

            if not wake_detected:
                # Should not reach here (listen_for_wake_word loops internally),
                # but handled defensively.
                continue

            # ── Phase B: Listen for the actual command ────────────────────────
            command = listen_for_command()

            if not command:
                # Microphone captured nothing intelligible
                speak("I didn't quite catch that. Say 'Jarvis' again when you're ready.")
                continue

            # ── Phase C: Route the command to the task executor ───────────────
            execute_command(command)

        except KeyboardInterrupt:
            # Ctrl+C — clean exit
            print("\n[MAIN LOOP] Keyboard interrupt received.")
            speak("Shutting down gracefully. Goodbye.")
            break
        except Exception as e:
            # Catch-all: log the error but keep the loop alive
            print(f"[MAIN LOOP] Unhandled exception: {e}")
            speak("I encountered an unexpected error but I'm still online.")
            time.sleep(1)


# ══════════════════════════════════════════════════════════════════════════════
# PC REMOTE CONTROL — Local REST Server (port 5001)
# ══════════════════════════════════════════════════════════════════════════════
# Allows your phone (via jarvis_cloud.py) to control this PC over LAN.
# Requires: pip install flask pyautogui
# Optional: pip install pycaw  (for precise Windows volume control)
# ──────────────────────────────────────────────────────────────────────────────

PC_REMOTE_PORT = 5001

try:
    from flask import Flask as _Flask, request as _req, jsonify as _jsonify
    _pc_app = _Flask("jarvis_remote")

    def _vol(delta: int):
        """Adjust system volume using pyautogui or pycaw."""
        try:
            from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume
            from ctypes import cast, POINTER
            from comtypes import CLSCTX_ALL
            devices = AudioUtilities.GetSpeakers()
            interface = devices.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
            volume = cast(interface, POINTER(IAudioEndpointVolume))
            cur = volume.GetMasterVolumeLevelScalar()
            volume.SetMasterVolumeLevelScalar(max(0.0, min(1.0, cur + delta / 100.0)), None)
        except Exception:
            # Fallback to pyautogui hotkeys
            import pyautogui
            if delta > 0:
                for _ in range(abs(delta) // 5):
                    pyautogui.press('volumeup')
            else:
                for _ in range(abs(delta) // 5):
                    pyautogui.press('volumedown')

    @_pc_app.route("/pc-command", methods=["POST", "GET"])
    def _pc_command():
        data   = _req.get_json(force=True) if _req.method == "POST" else {}
        action = data.get("action", "")
        param  = data.get("param", "")
        print(f"[PC REMOTE] Received action: {action}")
        try:
            import pyautogui
            if action == "volume_up":
                _vol(+10)
                return _jsonify({"ok": True, "result": "Volume increased"})
            elif action == "volume_down":
                _vol(-10)
                return _jsonify({"ok": True, "result": "Volume decreased"})
            elif action == "mute":
                pyautogui.press("volumemute")
                return _jsonify({"ok": True, "result": "Muted"})
            elif action == "lock":
                if sys.platform.startswith("win"):
                    os.system("rundll32.exe user32.dll,LockWorkStation")
                else:
                    os.system("loginctl lock-session")
                return _jsonify({"ok": True, "result": "Screen locked"})
            elif action == "sleep":
                if sys.platform.startswith("win"):
                    os.system("rundll32.exe powrprof.dll,SetSuspendState 0,1,0")
                return _jsonify({"ok": True, "result": "Going to sleep"})
            elif action == "shutdown":
                speak("PC shutdown initiated by phone command.")
                time.sleep(2)
                if sys.platform.startswith("win"):
                    os.system("shutdown /s /t 5")
                return _jsonify({"ok": True, "result": "Shutting down"})
            elif action == "restart":
                speak("PC restart initiated by phone command.")
                time.sleep(2)
                if sys.platform.startswith("win"):
                    os.system("shutdown /r /t 5")
                return _jsonify({"ok": True, "result": "Restarting"})
            elif action == "screenshot_pc":
                screenshot = pyautogui.screenshot()
                path = os.path.join(SCRIPT_DIR, "jarvis_screenshot.png")
                screenshot.save(path)
                return _jsonify({"ok": True, "result": f"Screenshot saved to {path}"})
            elif action == "open_app" and param:
                if sys.platform.startswith("win"):
                    os.startfile(param)
                return _jsonify({"ok": True, "result": f"Opened {param}"})
            else:
                return _jsonify({"ok": False, "result": f"Unknown action: {action}"})
        except ImportError as ie:
            return _jsonify({"ok": False, "result": f"Missing dependency: {ie}. Run: pip install pyautogui"})
        except Exception as e:
            return _jsonify({"ok": False, "result": str(e)})

    @_pc_app.route("/status")
    def _pc_status():
        return _jsonify({"ok": True, "service": "JARVIS PC Remote", "port": PC_REMOTE_PORT})

    def _run_remote_server():
        print(f"[PC REMOTE] Server starting on http://0.0.0.0:{PC_REMOTE_PORT}")
        _pc_app.run(host="0.0.0.0", port=PC_REMOTE_PORT, debug=False, use_reloader=False)

    PC_REMOTE_AVAILABLE = True
except ImportError:
    PC_REMOTE_AVAILABLE = False
    print("[PC REMOTE] Flask not installed — PC remote control disabled.")


# ══════════════════════════════════════════════════════════════════════════════
# SCRIPT ENTRY
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    # Start PC remote control server in background thread
    if PC_REMOTE_AVAILABLE:
        _t = threading.Thread(target=_run_remote_server, daemon=True)
        _t.start()
        print(f"[PC REMOTE] Active on port {PC_REMOTE_PORT} — phone can now control this PC")
    main()
