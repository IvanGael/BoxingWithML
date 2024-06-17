import mediapipe as mp
import cv2
from pynput.keyboard import Controller as KeyboardController
from scipy.stats import linregress
import numpy as np

mp_hands = mp.solutions.hands
hands = mp_hands.Hands()
hands_area = [[], []]
counter = 0

leftHand_positions = [np.array([0, 0]), np.array([0, 0])]
rightHand_positions = [np.array([0, 0]), np.array([0, 0])]
damping_factory = 0.15

key = KeyboardController()
cap = cv2.VideoCapture(0)

overlay_box_size = 200  # Size of the overlay box

def is_fist(hand_landmarks):
    # Check if the distance between the tips of the fingers and the base of the fingers are close
    # Indicating that the hand is in a fist position
    thumb_tip = hand_landmarks.landmark[4]
    index_tip = hand_landmarks.landmark[8]
    middle_tip = hand_landmarks.landmark[12]
    ring_tip = hand_landmarks.landmark[16]
    pinky_tip = hand_landmarks.landmark[20]

    thumb_base = hand_landmarks.landmark[2]
    index_base = hand_landmarks.landmark[5]
    middle_base = hand_landmarks.landmark[9]
    ring_base = hand_landmarks.landmark[13]
    pinky_base = hand_landmarks.landmark[17]

    tips = [thumb_tip, index_tip, middle_tip, ring_tip, pinky_tip]
    bases = [thumb_base, index_base, middle_base, ring_base, pinky_base]

    distances = [np.linalg.norm([tip.x - base.x, tip.y - base.y, tip.z - base.z]) for tip, base in zip(tips, bases)]

    return all(dist < 0.1 for dist in distances)

while cap.isOpened():
    success, img = cap.read()
    img = cv2.flip(img, 2)

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img.flags.writeable = False
    results = hands.process(img)
    img.flags.writeable = True
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    overlay_box_top_left = (img.shape[1] - overlay_box_size - 10, 10)  # Top right corner of the screen

    # Draw the overlay box
    cv2.rectangle(
        img, overlay_box_top_left,
        (overlay_box_top_left[0] + overlay_box_size, overlay_box_top_left[1] + overlay_box_size),
        (127, 0, 255), 2
    )

    fist_detected = False
    fist_label = ""

    if results.multi_hand_landmarks:
        for hand_id, hand_landmarks in enumerate(results.multi_hand_landmarks):
            x_list, y_list, z_list = [], [], []
            for lm in hand_landmarks.landmark:
                h, w, c = img.shape
                cx, cy, cz = int(lm.x * w), int(lm.y * h), lm.z
                x_list.append(cx)
                y_list.append(cy)
                z_list.append(cz)

            x_min, x_max = min(x_list), max(x_list)
            y_min, y_max = min(y_list), max(y_list)

            top_left_point_hand = np.array([x_min, y_min])
            bottom_right_point_hand = np.array([x_max, y_max])

            handedness = results.multi_handedness[hand_id].classification[0].label
            handedness_id = -1
            hand_positions = []

            if handedness == "Left":
                current_top_left_point_hand = leftHand_positions[0] + (
                            top_left_point_hand - leftHand_positions[0]) * damping_factory
                current_bottom_right_point_hand = leftHand_positions[1] + (
                            bottom_right_point_hand - leftHand_positions[1]) * damping_factory

                handedness_id = 0
                hand_positions = leftHand_positions
            else:
                current_top_left_point_hand = rightHand_positions[0] + (
                            top_left_point_hand - rightHand_positions[0]) * damping_factory
                current_bottom_right_point_hand = rightHand_positions[1] + (
                            bottom_right_point_hand - rightHand_positions[1]) * damping_factory

                handedness_id = 1
                hand_positions = rightHand_positions

            max_width_hand = (current_bottom_right_point_hand[0] - current_top_left_point_hand[0])
            max_height_hand = (current_bottom_right_point_hand[1] - current_top_left_point_hand[1])
            hands_area[handedness_id] += [max_width_hand * max_height_hand]

            cv2.rectangle(
                img, (int(current_top_left_point_hand[0]), int(current_top_left_point_hand[1])),
                (int(current_bottom_right_point_hand[0]), int(current_bottom_right_point_hand[1])), (255, 255, 255),
                2
            )

            hand_positions[0] = current_top_left_point_hand
            hand_positions[1] = current_bottom_right_point_hand

            # Draw landmarks in the overlay box
            overlay_x_min, overlay_y_min = overlay_box_top_left
            overlay_x_max, overlay_y_max = overlay_x_min + overlay_box_size, overlay_y_min + overlay_box_size

            for lm in hand_landmarks.landmark:
                cx = int(lm.x * overlay_box_size) + overlay_x_min
                cy = int(lm.y * overlay_box_size) + overlay_y_min
                cv2.circle(img, (cx, cy), 2, (0, 0, 255), -1)

            if is_fist(hand_landmarks):
                fist_detected = True
                fist_label = f"{handedness} Fist"

    if fist_detected:
        cv2.putText(img, fist_label, (overlay_box_top_left[0] + 10, overlay_box_top_left[1] + 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

    if len(hands_area[0]) >= 5 and len(hands_area[1]) >= 5:
        left = hands_area[0][-5:]
        right = hands_area[1][-5:]

        x = list(range(5))

        left_hand_slope, _, _, _, _ = linregress(x, left)
        right_hand_slope, _, _, _, _ = linregress(x, right)

        if left_hand_slope > 600:
            key.press('y')
            key.release('y')

            print(f"Left Click {counter}: {left_hand_slope}")
            counter += 1
        elif right_hand_slope > 600:
            key.press('u')
            key.release('u')

            print(f"Right Click {counter}: {right_hand_slope}")
            counter += 1

        hands_area[0] = []
        hands_area[1] = []

    cv2.imshow("Screen", img)

    pressed_key = cv2.waitKey(8)
    if pressed_key == ord('q'):
        cap.release()
        break

cv2.destroyAllWindows()
