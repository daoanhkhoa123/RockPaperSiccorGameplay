import time
import cv2
import numpy as np

CLASSES = ["Rock", "Paper", "Scissors"]

def preprocess(frame):
    img = cv2.resize(frame, (224, 224))
    img = img.astype("float32") / 255.0
    return np.expand_dims(img, axis=0)

import numpy as np
import tensorflow as tf

def get_prediction(interpreter, input_data, *, class_names=CLASSES, logging=None ,mgs=""):
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    input_data = input_data.astype(np.float32)/255.0
    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()
    
    output_data = interpreter.get_tensor(output_details[0]['index'])
    
    pred_class = np.argmax(output_data, axis=1)[0]
    confidence = float(output_data[0][pred_class])

    if logging:
        logging.info(f"Logits {mgs}: {np.round(output_data,2)}")
    
    if class_names is not None:
        return class_names[pred_class], confidence
    return pred_class, confidence

def decide_winner(m1, m2):
    if m1 == m2:
        return "Draw"
    rules = {"Rock": "Scissors", "Scissors": "Paper", "Paper": "Rock"}
    if rules[m1] == m2:
        return "Player 1 Wins!"
    else:
        return "Player 2 Wins!"
    
def extract_hand_rois(frame, results, target_size=224, scale=1.5):
    h, w, _ = frame.shape
    rois = []

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            xs = [lm.x for lm in hand_landmarks.landmark]
            ys = [lm.y for lm in hand_landmarks.landmark]

            x1, y1 = int(min(xs) * w), int(min(ys) * h)
            x2, y2 = int(max(xs) * w), int(max(ys) * h)

            bw, bh = x2 - x1, y2 - y1
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

            # longest side
            side = max(bw, bh)
            side = int(side * scale)

            # make square ROI centered on hand
            x1 = max(0, cx - side // 2)
            x2 = min(w, cx + side // 2)
            y1 = max(0, cy - side // 2)
            y2 = min(h, cy + side // 2)

            hand_img = frame[y1:y2, x1:x2]
            if hand_img.size == 0:
                continue
            hand_img = cv2.resize(hand_img, (target_size, target_size))
            hand_img = hand_img[None, ...]
            rois.append((hand_img, (x1, y1, x2, y2)))

    rois.sort(key=lambda r: r[1][0])
    return rois


def draw_with_hands(frame, hands):
    """Process frame with Mediapipe, return frame + rois"""
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)
    rois = extract_hand_rois(rgb, results)

    for idx, (_, (x1, y1, x2, y2)) in enumerate(rois):
        # Player 1 → Red, Player 2 → Blue
        color = (0, 0, 255) if idx == 0 else (255, 0, 0)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

    return frame, rois


def countdown_overlay(cap,hands, countdown=3):
    start = time.time()
    while True:
        ret, frame = cap.read()
        if not ret:
            return False
        frame = cv2.flip(frame, 1)

        # 🔹 Draw landmarks + bounding boxes
        frame, _ = draw_with_hands(frame, hands)

        # Compute time left
        elapsed = time.time() - start
        remaining = countdown - int(elapsed)

        if remaining > 0:
            cv2.putText(frame, str(remaining), (frame.shape[1]//2 - 30, frame.shape[0]//2),
                        cv2.FONT_HERSHEY_SIMPLEX, 4, (0, 0, 255), 8)
        else:
            # Final "GO!"
            cv2.putText(frame, "GO!", (frame.shape[1]//2 - 80, frame.shape[0]//2),
                        cv2.FONT_HERSHEY_SIMPLEX, 4, (0, 255, 0), 8)
            cv2.imshow("RPS", frame)
            cv2.waitKey(1000)  # show GO! for 1 sec
            return True

        cv2.imshow("RPS", frame)

        # Stop if user presses 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            return False


def draw_overlay(frame, score1, score2):
    h, w, _ = frame.shape

    # --- Header bar (top) ---
    cv2.rectangle(frame, (0, 0), (w, 70), (0, 0, 0), -1)  # Black filled rect

    # Player labels
    cv2.putText(frame, "Player 1", (30, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)  # Red
    cv2.putText(frame, "Player 2", (w - 200, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 0, 0), 3)  # Blue

    # --- Footer bar (bottom) ---
    cv2.rectangle(frame, (0, h - 70), (w, h), (0, 0, 0), -1)

    # Scores
    cv2.putText(frame, f"Score P1: {score1:.1f}", (30, h - 25),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    cv2.putText(frame, f"Score P2: {score2:.1f}", (w - 250, h - 25),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

    return frame

def draw_result(frame, text, color=(0, 255, 0)):
    """
    Draws the round result centered in the frame with a black background box.
    """
    h, w, _ = frame.shape

    # Get text size for centering
    (text_w, text_h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1.5, 4)
    x = (w - text_w) // 2
    y = (h // 2) + (text_h // 2)

    # Black rectangle behind text (padding 20px)
    cv2.rectangle(frame,
                  (x - 20, y - text_h - 20),
                  (x + text_w + 20, y + 20),
                  (0, 0, 0), -1)

    # Draw text
    cv2.putText(frame, text, (x, y),
                cv2.FONT_HERSHEY_SIMPLEX, 1.5, color, 4)

    return frame
