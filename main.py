import cv2
import tensorflow as tf
import logging
from mediapipe.python.solutions import hands as mp_hands

from ultils import get_prediction, decide_winner, countdown_overlay, draw_with_hands, draw_overlay, draw_result

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger("RPS")


MODEL_PATH = r"simple_cnn_model_20251001_150247.tflite"
interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)


def run_game():
    cap = cv2.VideoCapture(0)
    score1, score2 = 0, 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.flip(frame, 1)

        # Draw landmarks and ROIs
        frame, rois = draw_with_hands(frame,hands)
        frame = draw_overlay(frame, score1, score2)  # header/footer

        # Instruction
        cv2.putText(frame, "Press SPACE to start round", (20, 100),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
        cv2.imshow("RPS", frame)

        key = cv2.waitKey(1) & 0xFF

        if key == 32:  # SPACE
            if countdown_overlay(cap,hands):  # countdown with ROIs
                ret, snapshot = cap.read()
                if not ret:
                    continue
                snapshot = cv2.flip(snapshot, 1)
                snapshot, rois = draw_with_hands(snapshot,hands)

                if len(rois) >= 2:
                    (hand2, _), (hand1, _) = rois[:2]
                    # Predictions
                    move1, conf1 = get_prediction(interpreter, hand1, logging=logger, mgs="player1")
                    move2, conf2 = get_prediction(interpreter, hand2, logging=logger, mgs="player2")
                    result = decide_winner(move1, move2)

                    # Score update
                    if "Player 1" in result:
                        score1 += 1 * (1 + conf1 - conf2)
                        color = (0, 0, 255)  # red
                    elif "Player 2" in result:
                        score2 += 1 * (1 + conf2 - conf1)
                        color = (255, 0, 0)  # blue
                    else:
                        color = (0, 255, 0)  # tie = green

                    # Overlay moves + scores
                    snapshot = draw_overlay(snapshot, score1, score2)
                    h, w, _ = snapshot.shape
                    cv2.putText(snapshot, f"P1: {move1} ({conf1:.2f})", (20, 150),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
                    cv2.putText(snapshot, f"P2: {move2} ({conf2:.2f})", (w-300, 150),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

                    # Big result box
                    snapshot = draw_result(snapshot, result, color)

                    cv2.imshow("RPS", snapshot)
                    cv2.waitKey(2000)

                else:
                    # Not enough hands detected
                    snapshot = draw_overlay(snapshot, score1, score2)
                    snapshot = draw_result(snapshot, "Need 2 hands!", (0, 0, 255))
                    cv2.imshow("RPS", snapshot)
                    cv2.waitKey(1500)

        if key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    run_game()
