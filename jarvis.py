import pvporcupine
from pvrecorder import PvRecorder
import pyttsx3
import speech_recognition as sr
import pywhatkit as kit
import os
import webbrowser
import datetime
import ollama
from google import genai
from google.genai import types
from apikey import GEMINI_API_KEY 
from apikey import PICOVOICE_API_KEY
# --- CONFIGURATION ---
PICOVOICE_API_KEY = PICOVOICE_API_KEY
GEMINI_API_KEY = GEMINI_API_KEY

# Initialize once at the top
engine = pyttsx3.init('sapi5') 
engine.setProperty('rate', 180) # JARVIS speaks at a sophisticated pace

def speak(text):
    if text: # Only speak if there is content
        print(f"JARVIS: {text}")
        engine.say(text)
        engine.runAndWait() # CRITICAL: This makes the sound play

def take_command():
    r = sr.Recognizer()
    with sr.Microphone() as source:
        print("Listening...")
        # Reduce background noise for better spelling
        r.adjust_for_ambient_noise(source, duration=0.5) 
        r.pause_threshold = 1
        audio = r.listen(source)
    try:
        print("Recognizing...")
        # Use a specific language code for better accuracy
        query = r.recognize_google(audio, language='en-IN') 
        print(f"User: {query}")
        return query.lower()
    except Exception as e:
        return "none"

# --- BRAIN FUNCTIONS ---
def get_offline_brain(query):
    """Uses Ollama when internet is down or for privacy."""
    response = ollama.chat(model='llama3.2:1b', messages=[
        {'role': 'system', 'content': 'You are JARVIS. Be witty and brief.'},
        {'role': 'user', 'content': query}
    ])
    return response['message']['content']

def get_online_brain(query):
    """Uses Gemini 2.0 for high-level reasoning."""
    client = genai.Client(api_key=GEMINI_API_KEY)
    response = client.models.generate_content(
        model="gemini-2.0-flash",
        contents=query,
        config=types.GenerateContentConfig(
            system_instruction="You are JARVIS. Call the user Sir. Be sophisticated."
        )
    )
    return response.text

# --- TASK EXECUTION ---
def execute_task(query):
    query = query.lower()
    
    if "send a message" in query or "whatsapp" in query:
        speak("Who is the recipient, Sir? Say the phone number with country code.")
        # Re-use simple speech recognition for input
        r = sr.Recognizer()
        with sr.Microphone() as source:
            audio = r.listen(source)
            number = r.recognize_google(audio).replace(" ", "")
        speak("What is the message?")
        with sr.Microphone() as source:
            audio = r.listen(source)
            msg = r.recognize_google(audio)
        kit.sendwhatmsg_instantly(f"+{number}", msg, wait_time=15)
        speak("Message sent to the cloud, Sir.")

    elif "open notepad" in query:
        os.system("notepad")
        speak("Opened, Sir.")
    elif "open youtube" in query:
        webbrowser.open("https://www.youtube.com")
        speak("Opened YouTube, Sir.")
    elif "the time" in query:
        strTime = datetime.datetime.now().strftime("%H:%M:%S")
        speak(f"Sir, the time is {strTime}")
    

    else:
        # Check internet and choose brain
        try:
            reply = get_online_brain(query)
        except:
            reply = get_offline_brain(query)
        speak(reply)

# --- MAIN WAKE-WORD LOOP ---
def run_jarvis():
    porcupine = pvporcupine.create(access_key=PICOVOICE_API_KEY, keywords=['jarvis'])
    recorder = PvRecorder(device_index=-1, frame_length=porcupine.frame_length)
    recorder.start()
    
    speak("Systems online. Standing by for the wake word, Sir.")

    try:
        while True:
            pcm = recorder.read()
            if porcupine.process(pcm) >= 0:
                print("Wake word detected!")
                speak("At your service.")
                
                # Use Google Speech for the actual command
                r = sr.Recognizer()
                with sr.Microphone() as source:
                    audio = r.listen(source, timeout=5)
                    command = r.recognize_google(audio)
                    execute_task(command)
    except Exception as e:
        print(f"no more task: {e}")
    finally:
        recorder.stop()
        porcupine.delete()

if __name__ == "__main__":
    run_jarvis()