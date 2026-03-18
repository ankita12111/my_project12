"""
voice_server.py  —  my_project12
=================================
Voice + Text command server for the multi-task NAO arena.

Pipelines:
  Mic  → VOSK → Ollama → WebSocket → unified_controller
  Text → (Ollama if needed) → WebSocket → unified_controller

HOW TO RUN:
  1. Start Ollama:         ollama run llama3.2
  2. Run this server:      python voice_server.py
  3. Open Webots:          worlds/unified.wbt  → Press Play ▶

TEXT COMMANDS  (type in the terminal while server is running):
  - Type a natural phrase: "go to chair" / "play football" / "dance"
  - Or type a command directly: SCORE / STOP / PATROL etc.
  - Type  help  to list all commands.
  - Type  quit  to exit the server.

Requirements:
  pip install vosk pyaudio websockets requests
  Download VOSK model: https://alphacephei.com/vosk/models
  Place the model folder next to this file and set VOSK_MODEL_PATH below.
"""

import asyncio
import json
import queue
import threading
import websockets
import pyaudio
import requests
from vosk import Model, KaldiRecognizer

# ── Config ────────────────────────────────────────────────────────────────────
VOSK_MODEL_PATH = r"vosk-model-small-en-us-0.15\vosk-model-small-en-us-0.15"
WS_HOST         = "localhost"
WS_PORT         = 8765
OLLAMA_URL      = "http://localhost:11434/api/generate"
OLLAMA_MODEL    = "llama3.2"

# ── Commands ──────────────────────────────────────────────────────────────────
VALID_COMMANDS = [
    "GO_TO_CHAIR1", "GO_TO_CHAIR2",
    "GO_TO_BALL1",
    "KICK_BALL",    "GRAB_BOX",    "PLACE_BOX",
    "SCORE",        "DELIVER",     "TOUR",        "DANCE",
    "HAPPY",        "SAD",         "ANGRY",
    "WAVE",         "TAICHI",      "CELEBRATE",
    "PATROL",       "PLAY_BALL",   "STOP",        "STAND_UP",
]

SYSTEM_PROMPT = f"""You are a robot command interpreter.
Output ONLY one command name from this list — nothing else, no punctuation:

{chr(10).join('  ' + c for c in VALID_COMMANDS)}

Examples:
"go to chair one" / "first chair" / "chair 1"           → GO_TO_CHAIR1
"go to chair two" / "second chair" / "chair 2"          → GO_TO_CHAIR2
"go to the ball" / "go to ball" / "ball"                → GO_TO_BALL1
"kick the ball" / "shoot" / "kick it"                   → KICK_BALL
"score a goal" / "score" / "go score"                   → SCORE
"grab the box" / "pick up the box" / "get it"           → GRAB_BOX
"put down" / "place the box" / "drop it"                → PLACE_BOX
"deliver the box" / "bring the box to the chair"        → DELIVER
"tour" / "visit all" / "go everywhere"                  → TOUR
"dance" / "show me your moves" / "do a dance"           → DANCE
"be happy" / "show happy" / "happy"                     → HAPPY
"be sad" / "look sad" / "sad"                           → SAD
"be angry" / "get angry" / "angry"                      → ANGRY
"wave" / "wave your hand" / "say hello"                 → WAVE
"tai chi" / "do tai chi" / "taichi"                     → TAICHI
"celebrate" / "nice one" / "wipe forehead"              → CELEBRATE
"patrol" / "start patrolling" / "walk around"           → PATROL
"play football" / "play with the ball" / "play ball" / "keep playing" / "football"  → PLAY_BALL
"stop" / "halt" / "freeze"                              → STOP
"stand up" / "get up" / "recover"                       → STAND_UP
"""

# ── Helpers ───────────────────────────────────────────────────────────────────
def normalize_direct_command(text: str) -> str:
    return text.strip().upper().replace(" ", "_")

def looks_like_direct_command(text: str) -> bool:
    return normalize_direct_command(text) in VALID_COMMANDS

# ── Audio setup ───────────────────────────────────────────────────────────────
print("[SERVER] Loading VOSK model...")
vosk_model = Model(VOSK_MODEL_PATH)
recognizer = KaldiRecognizer(vosk_model, 16000)

pa     = pyaudio.PyAudio()
stream = pa.open(
    format=pyaudio.paInt16,
    channels=1,
    rate=16000,
    input=True,
    frames_per_buffer=8192
)
stream.start_stream()
print("[SERVER] Microphone ready ✅")

# ── WebSocket ─────────────────────────────────────────────────────────────────
connected_clients = set()

async def ws_handler(websocket):
    connected_clients.add(websocket)
    print(f"[WS] Webots connected  (total: {len(connected_clients)})")
    try:
        await websocket.wait_closed()
    finally:
        connected_clients.discard(websocket)
        print(f"[WS] Webots disconnected (total: {len(connected_clients)})")

async def broadcast(command):
    if not connected_clients:
        print(f"[WS] No Webots client connected — command dropped: {command}")
        return

    msg = json.dumps({"command": command})
    print(f"[WS] Sending → {command}")

    results = await asyncio.gather(
        *[c.send(msg) for c in list(connected_clients)],
        return_exceptions=True
    )

    dead_clients = []
    for client, result in zip(list(connected_clients), results):
        if isinstance(result, Exception):
            print(f"[WS] Send error: {result}")
            dead_clients.append(client)

    for client in dead_clients:
        connected_clients.discard(client)

# ── Ollama ────────────────────────────────────────────────────────────────────
def ask_ollama(text: str) -> str:
    try:
        r = requests.post(
            OLLAMA_URL,
            json={
                "model": OLLAMA_MODEL,
                "prompt": text,
                "system": SYSTEM_PROMPT,
                "stream": False,
                "options": {"temperature": 0},
            },
            timeout=10
        )
        raw = r.json().get("response", "").strip().upper()
        return raw.split()[0] if raw.split() else ""
    except Exception as e:
        print(f"[OLLAMA] Error: {e}")
        return ""

def map_text_to_command(text: str) -> str:
    direct = normalize_direct_command(text)
    if direct in VALID_COMMANDS:
        return direct
    return ask_ollama(text)

# ── Voice loop ────────────────────────────────────────────────────────────────
_audio_q: queue.Queue = queue.Queue()

def _mic_reader():
    while True:
        try:
            _audio_q.put(stream.read(4096, exception_on_overflow=False))
        except Exception:
            pass

threading.Thread(target=_mic_reader, daemon=True).start()

async def voice_loop():
    print("[SERVER] Listening for voice commands... Speak now!")
    while True:
        try:
            data = _audio_q.get_nowait()
        except queue.Empty:
            await asyncio.sleep(0.01)
            continue

        if recognizer.AcceptWaveform(data):
            text = json.loads(recognizer.Result()).get("text", "").strip()
            if not text:
                continue

            print(f"\n[VOICE]  Heard  : '{text}'")
            cmd = map_text_to_command(text)
            print(f"[OLLAMA] Mapped  : '{cmd}'")

            if cmd in VALID_COMMANDS:
                await broadcast(cmd)
            else:
                print(f"[OLLAMA] No match for: '{text}'")

# ── Typed text loop ───────────────────────────────────────────────────────────
def typed_input_loop(loop):
    print("[TEXT] Typed command mode enabled.")
    print("[TEXT] Type natural language or a direct command.")
    print("[TEXT] Type 'help' to list commands, 'quit' to exit.\n")

    while True:
        try:
            user_text = input("TEXT> ").strip()
        except EOFError:
            print("[TEXT] Input closed.")
            break
        except Exception as e:
            print(f"[TEXT] Input error: {e}")
            continue

        if not user_text:
            continue

        lower = user_text.lower().strip()

        if lower == "help":
            print("\n[TEXT] Available commands:")
            for cmd in VALID_COMMANDS:
                print("  ", cmd)
            print()
            continue

        if lower in ("quit", "exit"):
            print("[TEXT] Shutting down server...")
            loop.call_soon_threadsafe(loop.stop)
            break

        cmd = map_text_to_command(user_text)
        print(f"[TEXT] Input   : '{user_text}'")
        print(f"[TEXT] Mapped  : '{cmd}'")

        if cmd in VALID_COMMANDS:
            future = asyncio.run_coroutine_threadsafe(broadcast(cmd), loop)
            try:
                future.result(timeout=5)
            except Exception as e:
                print(f"[TEXT] Send error: {e}")
        else:
            print("[TEXT] Could not map that to a valid command.")

# ── Main ──────────────────────────────────────────────────────────────────────
async def main():
    print(f"\n{'='*55}")
    print("  Multi-Task NAO Voice Server  —  my_project12")
    print(f"  WebSocket : ws://{WS_HOST}:{WS_PORT}")
    print(f"  Ollama    : {OLLAMA_URL}  [{OLLAMA_MODEL}]")
    print(f"  Commands  : {len(VALID_COMMANDS)} available")
    print(f"{'='*55}\n")

    async with websockets.serve(ws_handler, WS_HOST, WS_PORT):
        loop = asyncio.get_running_loop()
        threading.Thread(target=typed_input_loop, args=(loop,), daemon=True).start()
        await voice_loop()

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except RuntimeError as e:
        if "Event loop stopped before Future completed" in str(e):
            print("[SERVER] Server stopped.")
        else:
            raise