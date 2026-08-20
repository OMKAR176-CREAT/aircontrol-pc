# AIR CONTROL - FAST GESTURE CONTROLLED PC
import cv2
import mediapipe as mp
import numpy as np
import pyautogui
import time
import os
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# Settings
CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480
SMOOTHING = 0.70
PINCH_DISTANCE = 35
CLICK_DELAY = 0.30

# Load MediaPipe hand model
MODEL = os.path.join(os.path.dirname(os.path.abspath(__file__)), "hand_landmarker.task")
if not os.path.isfile(MODEL):
    print("ERROR: hand_landmarker.task not found!")
    print("Expected:", MODEL)
    exit()

options = vision.HandLandmarkerOptions(
    base_options=python.BaseOptions(model_asset_path=MODEL),
    running_mode=vision.RunningMode.VIDEO,
    num_hands=1,
    min_hand_detection_confidence=0.7,
    min_hand_presence_confidence=0.7,
    min_tracking_confidence=0.7
)
hands = vision.HandLandmarker.create_from_options(options)

# Setup PyAutoGUI
screen_width, screen_height = pyautogui.size()
pyautogui.FAILSAFE = True
pyautogui.PAUSE = 0.005

# Setup webcam
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("ERROR: Webcam not accessible!")
    hands.close()
    exit()
cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)
cap.set(cv2.CAP_PROP_FPS, 60)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

# Program variables
timestamp = 0
control_enabled = True
mode = "MOUSE"
mouse_x = screen_width // 2
mouse_y = screen_height // 2
pinching = False
dragging = False
pinch_start = 0
last_click = 0
previous_scroll_y = None
last_time = time.time()
fps = 0

def distance(p1, p2):
    """Return distance between two points."""
    return np.hypot(p1[0] - p2[0], p1[1] - p2[1])

def finger_up(hand, tip, joint):
    """Check whether a finger is raised."""
    return hand[tip].y < hand[joint].y

def draw_hand(frame, hand):
    """Draw the detected hand landmarks."""
    h, w, _ = frame.shape
    points = [(int(p.x * w), int(p.y * h)) for p in hand]
    connections = [(0,1),(1,2),(2,3),(3,4),(0,5),(5,6),(6,7),(7,8),(5,9),(9,10),(10,11),(11,12),(9,13),(13,14),(14,15),(15,16),(13,17),(17,18),(18,19),(19,20),(0,17)]
    for start, end in connections:
        cv2.line(frame, points[start], points[end], (0,255,0), 2)
    for point in points:
        cv2.circle(frame, point, 4, (0,255,0), -1)

def to_screen(x, y):
    """Convert camera coordinates to screen coordinates."""
    x = np.interp(x, [20, CAMERA_WIDTH - 20], [0, screen_width])
    y = np.interp(y, [20, CAMERA_HEIGHT - 20], [0, screen_height])
    return int(x), int(y)

def move_cursor(x, y):
    """Move the PC cursor using the index finger."""
    global mouse_x, mouse_y
    target_x, target_y = to_screen(x, y)
    mouse_x += (target_x - mouse_x) * SMOOTHING
    mouse_y += (target_y - mouse_y) * SMOOTHING
    pyautogui.moveTo(int(mouse_x), int(mouse_y), _pause=False)

print("==========================================")
print("       AIR CONTROL - FAST MODE")
print("==========================================")
print("E = Enable / Disable")
print("M = Mouse mode")
print("S = Scroll mode")
print("P = Media mode")
print("ESC = Exit")
print("==========================================")

try:
    while True:
        success, frame = cap.read()
        if not success:
            print("Camera frame error.")
            break

        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape

        # Convert camera frame for MediaPipe
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        timestamp += 16
        result = hands.detect_for_video(image, timestamp)

        gesture = "No Hand"
        action = "Waiting"

        if result.hand_landmarks:
            hand = result.hand_landmarks[0]
            draw_hand(frame, hand)

            # Get important finger positions
            index = (int(hand[8].x * w), int(hand[8].y * h))
            thumb = (int(hand[4].x * w), int(hand[4].y * h))
            middle = (int(hand[12].x * w), int(hand[12].y * h))

            # Detect raised fingers
            index_up = finger_up(hand, 8, 6)
            middle_up = finger_up(hand, 12, 10)
            ring_up = finger_up(hand, 16, 14)
            pinky_up = finger_up(hand, 20, 18)

            one_finger = index_up and not middle_up and not ring_up and not pinky_up
            two_fingers = index_up and middle_up and not ring_up and not pinky_up
            open_hand = index_up and middle_up and ring_up and pinky_up

            # Detect pinches
            index_pinch = distance(index, thumb)
            middle_pinch = distance(middle, thumb)

            # Open palm pauses the system
            if open_hand:
                gesture = "Open Palm"
                action = "PAUSED"
                if dragging:
                    pyautogui.mouseUp()
                    dragging = False
                pinching = False
                previous_scroll_y = None

            # Disable control
            elif not control_enabled:
                gesture = "Disabled"
                action = "CONTROL OFF"
                if dragging:
                    pyautogui.mouseUp()
                    dragging = False
                pinching = False
                previous_scroll_y = None

            # Mouse mode
            elif mode == "MOUSE":
                # Thumb + middle = right click
                if middle_pinch < PINCH_DISTANCE:
                    gesture = "Middle Pinch"
                    action = "RIGHT CLICK"
                    if not pinching:
                        now = time.time()
                        if now - last_click > CLICK_DELAY:
                            pyautogui.rightClick()
                            last_click = now
                    pinching = True

                # Thumb + index = left click or drag
                elif index_pinch < PINCH_DISTANCE:
                    gesture = "Index Pinch"
                    if not pinching:
                        pinch_start = time.time()
                        pinching = True
                    if time.time() - pinch_start > 0.45:
                        if not dragging:
                            pyautogui.mouseDown()
                            dragging = True
                        move_cursor(index[0], index[1])
                        action = "DRAGGING"
                    else:
                        action = "READY"

                # Release pinch
                else:
                    if pinching:
                        if dragging:
                            pyautogui.mouseUp()
                            dragging = False
                            action = "DRAG END"
                        elif time.time() - pinch_start < 0.45:
                            now = time.time()
                            if now - last_click > CLICK_DELAY:
                                pyautogui.click()
                                last_click = now
                                action = "LEFT CLICK"
                    pinching = False

                    # Index finger = move cursor
                    if one_finger:
                        gesture = "Index Finger"
                        action = "MOVE CURSOR"
                        move_cursor(index[0], index[1])
                    else:
                        action = "Ready"

            # Scroll mode
            elif mode == "SCROLL":
                if two_fingers:
                    gesture = "Two Fingers"
                    action = "SCROLL"
                    current_y = index[1]
                    if previous_scroll_y is not None:
                        movement = previous_scroll_y - current_y
                        if abs(movement) > 3:
                            scroll = max(-10, min(10, int(movement / 6)))
                            pyautogui.scroll(scroll)
                    previous_scroll_y = current_y
                else:
                    gesture = "Scroll"
                    action = "Show Two Fingers"
                    previous_scroll_y = None

            # Media mode
            elif mode == "MEDIA":
                thumb_up = hand[4].y < hand[3].y and not index_up and not middle_up and not ring_up and not pinky_up

                if thumb_up:
                    gesture = "Thumb Up"
                    action = "PLAY / PAUSE"
                    now = time.time()
                    if now - last_click > 1:
                        pyautogui.press("playpause")
                        last_click = now
                elif one_finger:
                    gesture = "Index"
                    action = "VOLUME UP"
                    pyautogui.press("volumeup")
                elif two_fingers:
                    gesture = "Two Fingers"
                    action = "VOLUME DOWN"
                    pyautogui.press("volumedown")
                else:
                    gesture = "Media"
                    action = "Ready"

        else:
            gesture = "No Hand"
            action = "Waiting"
            previous_scroll_y = None
            if dragging:
                pyautogui.mouseUp()
                dragging = False
            pinching = False

        # Calculate FPS
        now = time.time()
        elapsed = now - last_time
        if elapsed > 0:
            fps = int(1 / elapsed)
        last_time = now

        # Display information
        cv2.rectangle(frame, (0,0), (640,110), (0,0,0), -1)
        status = "ON" if control_enabled else "OFF"

        cv2.putText(frame, f"AIR CONTROL | {status}", (10,25), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255,255,255), 2)
        cv2.putText(frame, f"Mode: {mode} | Gesture: {gesture}", (10,52), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (255,255,255), 1)
        cv2.putText(frame, f"Action: {action} | FPS: {fps}", (10,78), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (255,255,255), 1)
        cv2.putText(frame, "E: ON/OFF | M: Mouse | S: Scroll | P: Media | ESC: Exit", (10,100), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (200,200,200), 1)

        # Show camera
        cv2.imshow("AIR CONTROL - FAST", frame)

        # Keyboard controls
        key = cv2.waitKey(1) & 0xFF

        if key == 27:
            break
        elif key == ord("e"):
            control_enabled = not control_enabled
            if not control_enabled and dragging:
                pyautogui.mouseUp()
                dragging = False
            print("Control:", "ON" if control_enabled else "OFF")
        elif key == ord("m"):
            mode = "MOUSE"
            previous_scroll_y = None
            print("Mode: MOUSE")
        elif key == ord("s"):
            mode = "SCROLL"
            previous_scroll_y = None
            print("Mode: SCROLL")
        elif key == ord("p"):
            mode = "MEDIA"
            previous_scroll_y = None
            print("Mode: MEDIA")

except pyautogui.FailSafeException:
    print("PyAutoGUI safety activated.")
except Exception as error:
    print("PROGRAM ERROR:", error)
finally:
    try:
        if dragging:
            pyautogui.mouseUp()
    except:
        pass
    cap.release()
    hands.close()
    cv2.destroyAllWindows()
    print("AIR CONTROL CLOSED.")
