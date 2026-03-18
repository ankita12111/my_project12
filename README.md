# Multi-Task NAO Arena — my_project12

A Webots R2025a simulation where a **NAO humanoid robot** performs multiple tasks controlled by **voice commands** or **typed text**. The robot can play football, pick and carry objects, patrol between locations, express emotions, dance, and more — all from a single unified controller.

---

## Demo Features

| Command | What the Robot Does |
|---|---|
| `PLAY_BALL` | Continuously chases and kicks the football until told to stop |
| `SCORE` | Walks to the ball and kicks it in one seamless action |
| `GRAB_BOX` | Bends down, picks up the orange box, holds it at waist level |
| `DELIVER` | Picks up the box and carries it to Chair 1 |
| `PATROL` | Walks between Chair 1, the Ball, and Chair 2 on repeat |
| `TOUR` | Visits every location in order, waving at each stop |
| `DANCE` | Performs a multi-motion dance routine |
| `HAPPY / SAD / ANGRY` | Full-body emotion expressions |
| `WAVE / TAICHI / CELEBRATE` | Individual gesture motions |
| `GO_TO_CHAIR1/2` | Navigates to a specific chair |
| `KICK_BALL` | Walks to the ball and kicks it once |
| `PLACE_BOX` | Sets the carried box down |
| `STOP` | Immediately halts any active task |
| `STAND_UP` | Gets up from a fall |

---

## Project Structure

```
my_project12/
├── worlds/
│   └── unified.wbt                  # Webots world (arena, NAO, ball, box, chairs)
├── controllers/
│   └── unified_controller/
│       ├── unified_controller.py    # Main robot controller
│       └── runtime.ini              # Tells Webots to use Python
├── voice_server.py                  # Voice + text command server
├── vosk-model-small-en-us-0.15/    # Offline speech recognition model
└── README.md
```

> **Motion files and YOLO models** are loaded automatically from `my_project10`:
> `C:\Users\HP\Documents\my_project10\controllers\ball\`

---

## Requirements

### Python packages
```bash
pip install vosk pyaudio websockets requests ultralytics numpy
```

### External tools
| Tool | Purpose | Download |
|---|---|---|
| **Webots R2025a** | Robot simulation | [cyberbotics.com](https://cyberbotics.com) |
| **Ollama + llama3.2** | Maps speech/text to commands | [ollama.com](https://ollama.com) |
| **VOSK model** | Offline speech recognition | Already in `vosk-model-small-en-us-0.15/` |

### Folder dependencies
- `my_project10` must exist at `C:\Users\HP\Documents\my_project10\`
- It must contain trained YOLO weights at:
  - `controllers\ball\runs\detect\ball_detector\weights\best.pt`
  - `controllers\ball\runs\detect\box_detector\weights\best.pt`
- Motion `.motion` files must be directly inside `controllers\ball\`

---

## How to Run

**Step 1 — Start Ollama**
```bash
ollama run llama3.2
```

**Step 2 — Start the voice/text server**
```bash
cd C:\Users\HP\Documents\my_project12
python voice_server.py
```

**Step 3 — Open Webots**
- Open `worlds\unified.wbt`
- Press **Play ▶**
- The robot will say *"Ready! Waiting for your commands."*

---

## Giving Commands

### By Voice
Just speak naturally after the server starts. Examples:
- *"play football"* → robot chases and kicks the ball continuously
- *"go to chair one"* → robot walks to Chair 1
- *"dance"* → robot does a dance routine
- *"stop"* → stops whatever it's doing

### By Text
Type in the terminal where `voice_server.py` is running:
```
TEXT> play ball
TEXT> STOP
TEXT> deliver the box to the chair
TEXT> help        ← lists all commands
TEXT> quit        ← shuts down the server
```
You can type natural phrases or direct command names — both work.

---

## World Layout

```
         CHAIR1 (0, 3)
              |
   ROBOT ─────┼───── CHAIR2 (3, 0)
              |
         BALL (~3, -1.5)
         PICK_OBJECT (orange box)
```

- **Blue box** = Chair 1
- **Green box** = Chair 2
- **Soccer ball** = Football
- **Orange box** = Pick/place object

---

## Architecture

```
Microphone / Keyboard
        │
   voice_server.py
        │  VOSK  →  speech-to-text
        │  Ollama →  text-to-command (llama3.2)
        │
   WebSocket (ws://localhost:8765)
        │
   unified_controller.py   (runs inside Webots)
        │
   NAO Robot
     ├── GPS + IMU navigation
     ├── YOLO object detection (camera)
     ├── Motion files (walk, turn, kick, wave...)
     └── Supervisor API (grab/carry objects)
```

---

## Troubleshooting

| Problem | Fix |
|---|---|
| `Device "right_hand_connector" was not found` | Ignore — grab is supervisor-based, no connector needed |
| TTS not working | Ensure `C:\Windows\System32\WindowsPowerShell\v1.0\powershell.exe` exists |
| YOLO models not found | Check `my_project10` path in `unified_controller.py` line 50 |
| Robot falls during navigation | Say `stand up` or restart Webots |
| Ollama slow / timeout | Make sure `ollama run llama3.2` is running before starting the server |
| Voice not recognized | Speak clearly; check microphone is set as default in Windows Sound settings |
