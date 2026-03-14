"""
unified_controller.py  —  my_project12
=======================================
Multi-task NAO Robot Controller.

Motions & YOLO models are loaded from my_project10 automatically.
Just make sure my_project10 is at: C:/Users/HP/Documents/my_project10

Voice Commands (send via voice_server.py):
  GO_TO_CHAIR1  GO_TO_CHAIR2  GO_TO_BALL1  GO_TO_BALL2
  KICK_BALL     GRAB_BOX      PLACE_BOX
  HAPPY         SAD           ANGRY
  WAVE          TAICHI        CELEBRATE
  PATROL        STOP          STAND_UP
"""

import sys
import math
import os
import threading
import asyncio
import json
import queue

import numpy as np

# ── TTS (optional) ────────────────────────────────────────────────────────────
try:
    import pyttsx3
    _tts_engine = pyttsx3.init()
    _tts_engine.setProperty('rate', 150)
    TTS_AVAILABLE = True
except Exception:
    TTS_AVAILABLE = False

from controller import Supervisor, Motion
from ultralytics import YOLO
import websockets

# =========================================
# PATHS  — pointing to my_project10 resources
# =========================================
_PROJECT10    = r"C:\Users\HP\Documents\my_project10\controllers\ball"
_MOTION_DIR   = os.path.join(_PROJECT10, "motions")
_DETECT_DIR   = os.path.join(_PROJECT10, "runs", "detect")
_BALL_PT      = os.path.join(_DETECT_DIR, "ball_detector", "weights", "best.pt")
_BOX_PT       = os.path.join(_DETECT_DIR, "box_detector",  "weights", "best.pt")

# =========================================
# ROBOT SETUP
# =========================================
robot     = Supervisor()
TIME_STEP = int(robot.getBasicTimeStep())

def step():
    if robot.step(TIME_STEP) == -1:
        sys.exit(0)

def settle(frames=20):
    for _ in range(frames):
        step()

# =========================================
# YOLO MODELS
# =========================================
ball_model = YOLO(_BALL_PT) if os.path.exists(_BALL_PT) else None
box_model  = YOLO(_BOX_PT)  if os.path.exists(_BOX_PT)  else None

print(f"[YOLO] ball_detector : {'loaded ✅' if ball_model else 'NOT FOUND ❌  →  ' + _BALL_PT}")
print(f"[YOLO] box_detector  : {'loaded ✅' if box_model  else 'NOT FOUND ❌  →  ' + _BOX_PT}")

# =========================================
# SENSORS
# =========================================
gps = robot.getDevice("gps")
gps.enable(TIME_STEP)

imu = robot.getDevice("inertial unit")
imu.enable(TIME_STEP)

camera_top = robot.getDevice("CameraTop")
camera_bot = robot.getDevice("CameraBottom")
if camera_top: camera_top.enable(TIME_STEP)
if camera_bot: camera_bot.enable(TIME_STEP)

CAM_WIDTH      = camera_bot.getWidth()  if camera_bot else (camera_top.getWidth()  if camera_top else 160)
CAM_HEIGHT     = camera_bot.getHeight() if camera_bot else (camera_top.getHeight() if camera_top else 120)
CAM_CENTER_TOL = CAM_WIDTH // 10
SIDESTEP_TOL   = CAM_WIDTH // 5

# Supervisor DEF nodes
soccer_node = robot.getFromDef("SOCCER_BALL")
box_node    = robot.getFromDef("PICK_OBJECT")

# =========================================
# MOTORS
# =========================================
MOTOR_NAMES = [
    "HeadPitch", "HeadYaw",
    "LShoulderPitch", "LShoulderRoll", "LElbowYaw", "LElbowRoll",
    "RShoulderPitch", "RShoulderRoll", "RElbowYaw", "RElbowRoll",
    "LHipYawPitch", "LHipRoll", "LHipPitch", "LKneePitch", "LAnklePitch", "LAnkleRoll",
    "RHipYawPitch", "RHipRoll", "RHipPitch", "RKneePitch", "RAnklePitch", "RAnkleRoll",
]
NUMBER_OF_JOINTS = len(MOTOR_NAMES)
motors = [robot.getDevice(n) for n in MOTOR_NAMES]

STABLE_LEGS = [
    0.0, -0.03, -0.50, 1.05, -0.55, 0.03,
    0.0, -0.03, -0.50, 1.05, -0.55, 0.03,
]

# =========================================
# CONNECTOR
# =========================================
hand_connector = robot.getDevice("right_hand_connector")
hand_connector.enablePresence(TIME_STEP)
_carrying = False

def object_in_reach():
    return hand_connector.getPresence() == 1

def grab_object():
    global _carrying
    if object_in_reach():
        hand_connector.lock()
        _carrying = True
        print("[GRAB] Locked!")
        return True
    print("[GRAB] Nothing in reach.")
    return False

def release_object():
    global _carrying
    hand_connector.unlock()
    if box_node:
        box_node.resetPhysics()
    _carrying = False
    print("[RELEASE] Released.")

# =========================================
# LOAD ALL MOTIONS FROM my_project10
# =========================================
def _load(name):
    path = os.path.join(_MOTION_DIR, name)
    if os.path.exists(path):
        print(f"[MOTION] ✅ {name}")
        return Motion(path)
    print(f"[MOTION] ❌ {name}  (looked in {path})")
    return None

# Walking
forward50_motion  = _load("Forwards50.motion")
forward_motion    = _load("Forwards.motion")
backward_motion   = _load("Backwards.motion")

# Side steps
sidestep_L_motion = _load("SideStepLeft.motion")
sidestep_R_motion = _load("SideStepRight.motion")

# Turns — full 5° increments (5,10,15,...,60) + 180
left_motions  = {}
right_motions = {}

for deg in range(5, 65, 5):
    m = _load(f"TurnLeft{deg}.motion")
    if m: left_motions[deg] = m

m180 = _load("TurnLeft180.motion")
if m180: left_motions[180] = m180

for deg in range(5, 65, 5):
    m = _load(f"TurnRight{deg}.motion")
    if m: right_motions[deg] = m

print(f"[MOTIONS] Left  turns: {sorted(left_motions.keys())}")
print(f"[MOTIONS] Right turns: {sorted(right_motions.keys())}")

# Actions
shoot_motion   = _load("Shoot.motion")
standup_motion = _load("StandUpFromFront.motion")

# Gestures
wave_motion    = _load("HandWave.motion")
taichi_motion  = _load("TaiChi.motion")
wipe_motion    = _load("WipeForehead.motion")

# =========================================
# TTS
# =========================================
def speak(text):
    print(f"[SPEAK] {text}")
    if not TTS_AVAILABLE:
        return
    def _run():
        _tts_engine.say(text)
        _tts_engine.runAndWait()
    threading.Thread(target=_run, daemon=True).start()

# =========================================
# MOVEMENT HELPERS
# =========================================
def movement_decomposition(target, duration):
    n    = max(1, int(duration * 1000 / TIME_STEP))
    cur  = [motors[i].getTargetPosition() for i in range(NUMBER_OF_JOINTS)]
    diff = [(target[i] - cur[i]) / n for i in range(NUMBER_OF_JOINTS)]
    for _ in range(n):
        for j in range(NUMBER_OF_JOINTS):
            cur[j] += diff[j]
            motors[j].setPosition(cur[j])
        step()

def stand_up(duration=3.0):
    movement_decomposition(
        [0.0, 0.0, 1.5, 0.2, -1.4, -0.5, 1.5, -0.2, 1.4, 0.5] + STABLE_LEGS,
        duration)

def look_down():
    motors[0].setPosition(0.4)
    settle(20)

def head_center():
    motors[0].setPosition(0.0)
    motors[1].setPosition(0.0)
    settle(20)

def walk_forward():
    m = forward50_motion
    if m:
        m.stop(); m.play()
        while not m.isOver(): step()
        settle(20)

def walk_forward_short():
    m = forward_motion or forward50_motion
    if m:
        m.stop(); m.play()
        while not m.isOver(): step()
        settle(20)

def walk_backward():
    if backward_motion:
        backward_motion.stop(); backward_motion.play()
        while not backward_motion.isOver(): step()
        settle(20)

def sidestep(direction):
    m = sidestep_L_motion if direction == 'left' else sidestep_R_motion
    if m:
        m.stop(); m.play()
        while not m.isOver(): step()
        settle(20)
        print(f"[SIDESTEP] {direction}")

def play_turn(degrees, direction):
    m_map = left_motions if direction == 'left' else right_motions
    if not m_map:
        return
    # Snap to nearest 5°, clamp 5-60; use 180 for large left turns
    deg = max(5, min(60, round(abs(degrees) / 5) * 5))
    if direction == 'left' and abs(degrees) > 120 and 180 in left_motions:
        deg = 180
    if deg not in m_map:
        deg = min(sorted(m_map.keys()), key=lambda x: abs(x - deg))
    m = m_map[deg]
    m.stop(); m.play()
    while not m.isOver(): step()
    settle(50)
    print(f"[TURN] {direction} {deg}°")

def recover_from_fall():
    print("[RECOVER] Getting up...")
    if standup_motion:
        standup_motion.stop(); standup_motion.play()
        while not standup_motion.isOver(): step()
        settle(40)
    else:
        stand_up(4.0)

# =========================================
# NAVIGATION HELPERS
# =========================================
def get_pos():
    v = gps.getValues(); return v[0], v[1]

def get_heading():
    _, _, yaw = imu.getRollPitchYaw()
    return math.degrees(yaw)

def get_node_pos(node):
    p = node.getPosition(); return p[0], p[1]

def gps_angle_dist(tx, ty):
    rx, ry = get_pos()
    dx, dy = tx - rx, ty - ry
    return math.degrees(math.atan2(dy, dx)), math.hypot(dx, dy)

def face_angle(target_deg):
    for _ in range(15):
        corr = target_deg - get_heading()
        while corr >  180: corr -= 360
        while corr < -180: corr += 360
        if abs(corr) <= 5: break
        play_turn(abs(corr), 'left' if corr > 0 else 'right')
        settle(30)

def check_fallen():
    roll, pitch, _ = imu.getRollPitchYaw()
    return abs(math.degrees(pitch)) > 45 or abs(math.degrees(roll)) > 45

# =========================================
# YOLO DETECTION
# =========================================
def _cam_img(cam):
    raw = cam.getImage()
    img = np.frombuffer(raw, dtype=np.uint8).reshape(
          (cam.getHeight(), cam.getWidth(), 4))
    return img[:, :, :3]

def _detect(cam, model, thresh=0.3):
    if cam is None or model is None: return None
    results = model(_cam_img(cam), verbose=False)
    best, best_conf = None, thresh
    for r in results:
        for box in r.boxes:
            conf = float(box.conf)
            if conf > best_conf:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                best = ((x1+x2)/2, (y1+y2)/2, (x2-x1)*(y2-y1), conf)
                best_conf = conf
    return best

def best_detection(model):
    return _detect(camera_bot, model) or _detect(camera_top, model)

# =========================================
# COMBINED GPS + YOLO APPROACH
# =========================================
GRAB_DIST      = 0.38
WALK_STOP_AREA = 9000

def combined_approach(target_node, yolo_model):
    look_down()
    no_det = 0; last_dist = 999.0; stuck = 0

    for i in range(200):
        if check_fallen(): recover_from_fall(); continue

        tx, ty = get_node_pos(target_node)
        angle, dist = gps_angle_dist(tx, ty)
        print(f"[APPROACH {i:02d}] dist={dist:.3f}  angle={angle:.1f}")

        if dist >= last_dist - 0.01: stuck += 1
        else: stuck = 0
        last_dist = dist

        if stuck >= 8:
            face_angle(angle); walk_forward(); stuck = 0; continue

        if dist <= GRAB_DIST:
            print("[APPROACH] In grab range!"); return True

        det = best_detection(yolo_model)
        if det:
            no_det = 0
            cx, cy, area, conf = det
            offset = cx - (CAM_WIDTH / 2)

            if area < 6000:
                gps_err = angle - get_heading()
                while gps_err >  180: gps_err -= 360
                while gps_err < -180: gps_err += 360
                face_angle(angle) if abs(gps_err) > 5 else walk_forward()
            elif abs(offset) <= CAM_CENTER_TOL:
                if area < WALK_STOP_AREA: walk_forward()
            elif abs(offset) <= SIDESTEP_TOL:
                sidestep('right' if offset > 0 else 'left')
            else:
                ang_corr = (offset / CAM_WIDTH) * 60.0
                play_turn(abs(ang_corr), 'right' if ang_corr > 0 else 'left')
        else:
            no_det += 1
            face_angle(angle); walk_forward()
            if no_det >= 15: break

    return True

# =========================================
# GRAB / LIFT
# =========================================
def reach_for_object():
    movement_decomposition(
        [0.4, 0.0, 1.3, 0.15, -1.4, -0.5, 1.3, -0.15, 1.4, 0.5] + STABLE_LEGS, 1.5)
    settle(10)
    movement_decomposition(
        [0.5, 0.0, 1.57, 0.10, -1.4, -0.3, 1.57, -0.10, 1.4, 0.3] + STABLE_LEGS, 1.5)
    for _ in range(int(3000 / TIME_STEP)):
        step()
        if object_in_reach(): return grab_object()
    return False

def lift_object():
    movement_decomposition(
        [0.0, 0.0, 1.5, 0.1, -1.4, -0.5, 1.5, -0.1, 1.4, 0.5] + STABLE_LEGS, 2.0)
    settle(20)
    movement_decomposition(
        [0.0, 0.0, 1.2, -0.10, -1.4, -0.9, 1.2, 0.10, 1.4, 0.9] + STABLE_LEGS, 2.0)
    settle(40)

# =========================================
# TASK: GRAB BOX  (box_detector YOLO)
# =========================================
def task_grab_box():
    if _carrying: speak("Already carrying something."); return
    if box_node is None: speak("Cannot find the box."); return

    speak("I will grab the orange box.")

    for _ in range(300):
        tx, ty = get_node_pos(box_node)
        angle, dist = gps_angle_dist(tx, ty)
        if dist <= 1.5: break
        face_angle(angle); walk_forward()

    combined_approach(box_node, box_model)

    angle, dist = gps_angle_dist(*get_node_pos(box_node))
    face_angle(angle)
    if dist > 0.22:
        walk_forward_short()
        face_angle(gps_angle_dist(*get_node_pos(box_node))[0])

    stand_up(2.0); settle(30)
    grabbed = reach_for_object()
    if grabbed:
        speak("Got it! Lifting the box.")
        lift_object(); settle(100)
    else:
        speak("Could not grab. Retrying.")
        stand_up(3.0); settle(150)

# =========================================
# TASK: PLACE BOX
# =========================================
def task_place_box():
    if not _carrying: speak("Not carrying anything."); return
    speak("Putting the box down.")
    movement_decomposition(
        [0.0, 0.0, 1.5, 0.1, -1.4, -0.5, 1.5, -0.1, 1.4, 0.5] + STABLE_LEGS, 1.5)
    settle(20); release_object(); settle(60)
    walk_backward(); settle(20)
    stand_up(2.0); settle(100)
    speak("Box placed.")

# =========================================
# TASK: KICK BALL  (ball_detector YOLO)
# =========================================
def task_kick_ball():
    if soccer_node is None: speak("Cannot find the soccer ball."); return

    speak("I will kick the ball!")
    look_down()

    for i in range(200):
        tx, ty = get_node_pos(soccer_node)
        angle, dist = gps_angle_dist(tx, ty)
        if dist <= 1.5: break
        face_angle(angle); walk_forward()

    no_det = 0
    for i in range(150):
        if check_fallen(): recover_from_fall(); continue
        tx, ty = get_node_pos(soccer_node)
        angle, dist = gps_angle_dist(tx, ty)
        if dist <= 0.55: break

        det = best_detection(ball_model)
        if det:
            no_det = 0
            cx, cy, area, conf = det
            offset = cx - (CAM_WIDTH / 2)
            if abs(offset) <= CAM_CENTER_TOL:
                walk_forward()
            elif abs(offset) <= SIDESTEP_TOL:
                sidestep('right' if offset > 0 else 'left')
            else:
                ang_corr = (offset / CAM_WIDTH) * 60.0
                play_turn(abs(ang_corr), 'right' if ang_corr > 0 else 'left')
        else:
            no_det += 1; face_angle(angle); walk_forward()
            if no_det >= 15: break

    face_angle(gps_angle_dist(*get_node_pos(soccer_node))[0])
    speak("Shooting!")
    if shoot_motion:
        shoot_motion.stop(); shoot_motion.play()
        while not shoot_motion.isOver(): step()
        settle(50)

    stand_up(2.0); settle(80); head_center()
    speak("Goal!")
    if wipe_motion:
        wipe_motion.stop(); wipe_motion.play()
        while not wipe_motion.isOver(): step()
        settle(30)
    stand_up(2.0)

# =========================================
# TASK: NAVIGATE
# =========================================
NAV_STOP_DIST = 0.7

LOCATIONS = {
    "CHAIR1": (0.0,  3.0),
    "CHAIR2": (3.0,  0.0),
    "BALL1":  (3.0,  3.0),
    "BALL2":  (2.4, -1.5),
}

def navigate_to(tx, ty, label="target"):
    speak(f"Going to {label}.")
    for i in range(300):
        if check_fallen(): recover_from_fall(); continue
        angle, dist = gps_angle_dist(tx, ty)
        print(f"[NAV {i:02d}] dist={dist:.2f}")
        if dist <= NAV_STOP_DIST:
            speak(f"Arrived at {label}!")
            if wave_motion:
                wave_motion.stop(); wave_motion.play()
                while not wave_motion.isOver(): step()
                settle(20)
            stand_up(1.5); return True
        face_angle(angle); walk_forward()
    speak(f"Could not reach {label}.")
    return False

# =========================================
# TASK: EMOTIONS
# =========================================
def task_happy():
    speak("I am feeling very happy!")
    for _ in range(3):
        movement_decomposition(
            [0.0, 0.0, 0.5, 0.8, -1.4, -0.5, 0.5, -0.8, 1.4, 0.5] + STABLE_LEGS, 0.35)
        movement_decomposition(
            [0.0, 0.0, 1.5, 0.2, -1.4, -0.5, 1.5, -0.2, 1.4, 0.5] + STABLE_LEGS, 0.35)
    stand_up(1.5)

def task_sad():
    speak("I am feeling very sad.")
    movement_decomposition(
        [-0.3, 0.0, 1.8, 0.1, -1.4, -0.5, 1.8, -0.1, 1.4, 0.5] + [
            0.0, -0.03, -0.35, 0.75, -0.40, 0.03,
            0.0, -0.03, -0.35, 0.75, -0.40, 0.03,
        ], 2.5)
    settle(80); stand_up(2.0)

def task_angry():
    speak("I am feeling very angry!")
    for _ in range(4):
        movement_decomposition(
            [0.0, 0.0, 0.8, 0.5, -1.4, -0.8, 0.8, -0.5, 1.4, 0.8] + STABLE_LEGS, 0.18)
        movement_decomposition(
            [0.0, 0.0, 1.2, 0.2, -1.4, -0.4, 1.2, -0.2, 1.4, 0.4] + STABLE_LEGS, 0.18)
    stand_up(1.5)

# =========================================
# TASK: GESTURES
# =========================================
def task_wave():
    speak("Hello there!")
    if wave_motion:
        wave_motion.stop(); wave_motion.play()
        while not wave_motion.isOver(): step()
        settle(20)
    stand_up(2.0)

def task_taichi():
    speak("Let me do some Tai Chi.")
    if taichi_motion:
        taichi_motion.stop(); taichi_motion.play()
        while not taichi_motion.isOver(): step()
        settle(30)
    stand_up(2.0)

def task_celebrate():
    speak("Yes! Let me celebrate!")
    if wipe_motion:
        wipe_motion.stop(); wipe_motion.play()
        while not wipe_motion.isOver(): step()
        settle(20)
    stand_up(2.0)

def task_standup():
    speak("Getting up!")
    recover_from_fall()

# =========================================
# WEBSOCKET CLIENT  (from voice_server.py)
# =========================================
cmd_queue = queue.Queue()

VALID_COMMANDS = {
    "GO_TO_CHAIR1", "GO_TO_CHAIR2", "GO_TO_BALL1", "GO_TO_BALL2",
    "KICK_BALL", "GRAB_BOX", "PLACE_BOX",
    "HAPPY", "SAD", "ANGRY",
    "WAVE", "TAICHI", "CELEBRATE",
    "PATROL", "STOP", "STAND_UP",
}

def _ws_thread():
    async def _listen():
        uri = "ws://localhost:8765"
        while True:
            try:
                async with websockets.connect(uri) as ws:
                    print(f"[WS] Connected to {uri}")
                    speak("Voice control ready.")
                    async for msg in ws:
                        try:
                            cmd = json.loads(msg).get("command", "").upper().strip()
                            if cmd in VALID_COMMANDS:
                                print(f"[WS] → {cmd}")
                                cmd_queue.put(cmd)
                        except json.JSONDecodeError:
                            pass
            except Exception as e:
                print(f"[WS] Disconnected ({e}). Retry in 3s...")
                await asyncio.sleep(3)
    asyncio.run(_listen())

threading.Thread(target=_ws_thread, daemon=True).start()

# =========================================
# STARTUP
# =========================================
print("[NAO] Starting up — multi-task unified controller")
print(f"[NAO] Motions from : {_MOTION_DIR}")
print(f"[NAO] YOLO from    : {_DETECT_DIR}")
stand_up(4.0)
settle(80)
speak("Ready! Waiting for your commands.")
print(f"[NAO] Heading = {get_heading():.1f}°")
print(f"[NAO] Commands: {sorted(VALID_COMMANDS)}")

# =========================================
# MAIN LOOP
# =========================================
_patrol_active = False
_patrol_idx    = 0
_patrol_order  = ["CHAIR1", "CHAIR2"]

while True:
    cmd = None
    try:
        cmd = cmd_queue.get_nowait()
    except queue.Empty:
        pass

    if cmd:
        print(f"\n{'='*40}\n  CMD → {cmd}\n{'='*40}")
        _patrol_active = False

        if   cmd == "STOP":         speak("Stopping."); stand_up(2.0)
        elif cmd == "STAND_UP":     task_standup()
        elif cmd == "HAPPY":        task_happy()
        elif cmd == "SAD":          task_sad()
        elif cmd == "ANGRY":        task_angry()
        elif cmd == "WAVE":         task_wave()
        elif cmd == "TAICHI":       task_taichi()
        elif cmd == "CELEBRATE":    task_celebrate()
        elif cmd == "GO_TO_CHAIR1": navigate_to(*LOCATIONS["CHAIR1"], "Chair 1")
        elif cmd == "GO_TO_CHAIR2": navigate_to(*LOCATIONS["CHAIR2"], "Chair 2")
        elif cmd == "GO_TO_BALL1":  navigate_to(*LOCATIONS["BALL1"],  "Ball 1")
        elif cmd == "GO_TO_BALL2":  navigate_to(*LOCATIONS["BALL2"],  "Ball 2")
        elif cmd == "KICK_BALL":    task_kick_ball()
        elif cmd == "GRAB_BOX":     task_grab_box()
        elif cmd == "PLACE_BOX":    task_place_box()
        elif cmd == "PATROL":
            _patrol_active = True
            speak("Starting patrol between chairs.")

    elif _patrol_active:
        name = _patrol_order[_patrol_idx % len(_patrol_order)]
        navigate_to(*LOCATIONS[name], name)
        _patrol_idx += 1
        try:
            cmd = cmd_queue.get_nowait()
            if cmd:
                _patrol_active = False
                cmd_queue.put(cmd)
        except queue.Empty:
            pass
    else:
        step()
