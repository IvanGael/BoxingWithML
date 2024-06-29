import mediapipe as mp
import cv2
from pynput.keyboard import Controller as KeyboardController
from scipy.stats import linregress
import numpy as np
import time

mp_hands = mp.solutions.hands
hands = mp_hands.Hands()
mp_drawing = mp.solutions.drawing_utils
hands_area = [[], []]
counter = 0

leftHand_positions = [np.array([0, 0]), np.array([0, 0])]
rightHand_positions = [np.array([0, 0]), np.array([0, 0])]
damping_factory = 0.15

key = KeyboardController()
cap = cv2.VideoCapture("video.mp4")  # Use 0 for the webcam, or "video.mp4" for a video file

fps = cap.get(cv2.CAP_PROP_FPS)
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('output.avi', fourcc, fps, (int(cap.get(3)), int(cap.get(4))))

overlay_box_size = 200  # Size of the overlay box

# Load the fist image
fist_img = cv2.imread('fist.png')
fist_img = cv2.resize(fist_img, (50, 50))

# Variables for punch counting
punch_count = 0
previous_fist_state = False
fist_start_time = None
fist_delay = 1.4  # Delay in seconds

def get_handedness(hand_landmarks):
    # Calculate the center of the hand
    center_x = sum([lm.x for lm in hand_landmarks.landmark]) / len(hand_landmarks.landmark)
    center_y = sum([lm.y for lm in hand_landmarks.landmark]) / len(hand_landmarks.landmark)

    # Calculate the wrist position
    wrist_x = hand_landmarks.landmark[0].x
    wrist_y = hand_landmarks.landmark[0].y

    # If the wrist is to the left of the center, it's a right hand
    if wrist_x < center_x:
        return "Right"
    else:
        return "Left"

def is_fist(hand_landmarks):
    # Check if the thumb is close to the palm
    thumb_tip = hand_landmarks.landmark[4]
    thumb_ip = hand_landmarks.landmark[3]
    thumb_mcp = hand_landmarks.landmark[2]
    thumb_cmc = hand_landmarks.landmark[1]

    if (
        np.linalg.norm([thumb_tip.x - thumb_ip.x, thumb_tip.y - thumb_ip.y, thumb_tip.z - thumb_ip.z]) < 0.1
        and np.linalg.norm([thumb_ip.x - thumb_mcp.x, thumb_ip.y - thumb_mcp.y, thumb_ip.z - thumb_mcp.z]) < 0.1
        and np.linalg.norm([thumb_mcp.x - thumb_cmc.x, thumb_mcp.y - thumb_cmc.y, thumb_mcp.z - thumb_cmc.z]) < 0.1
    ):
        # Check if the other fingers are close to the palm
        for finger_id in [8, 12, 16, 20]:
            finger_tip = hand_landmarks.landmark[finger_id]
            finger_pip = hand_landmarks.landmark[finger_id - 2]
            finger_mcp = hand_landmarks.landmark[finger_id - 3]

            if (
                np.linalg.norm([finger_tip.x - finger_pip.x, finger_tip.y - finger_pip.y, finger_tip.z - finger_pip.z]) > 0.2
                or np.linalg.norm([finger_pip.x - finger_mcp.x, finger_pip.y - finger_mcp.y, finger_pip.z - finger_mcp.z]) > 0.2
            ):
                return False

        return True

    return False

def segment_hand(image, hand_landmarks):
    mask = np.zeros(image.shape[:2], dtype=np.uint8)
    points = []
    for lm in hand_landmarks.landmark:
        h, w, _ = image.shape
        cx, cy = int(lm.x * w), int(lm.y * h)
        points.append((cx, cy))

    points = np.array(points, dtype=np.int32)
    cv2.fillPoly(mask, [points], 255)
    segmented_hand = cv2.bitwise_and(image, image, mask=mask)
    return segmented_hand, mask

while cap.isOpened():
    success, img = cap.read()

    if not success:
        break

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
        (255, 255, 255), -1
    )

    fist_detected = False
    fist_label = ""

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            handedness = get_handedness(hand_landmarks)

            if handedness == "Left":
                handedness_id = 0
                hand_positions = leftHand_positions
            else:
                handedness_id = 1
                hand_positions = rightHand_positions

            # Segment the hand
            segmented_hand, hand_mask = segment_hand(img, hand_landmarks)

            # Find contours
            contours, _ = cv2.findContours(hand_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if contours:
                largest_contour = max(contours, key=cv2.contourArea)
                x, y, w, h = cv2.boundingRect(largest_contour)
                hand_bbox = np.array([x, y, x + w, y + h])

                # Update hand positions using the calculated bounding box
                hand_positions[0] = np.array([hand_bbox[0], hand_bbox[1]])
                hand_positions[1] = np.array([hand_bbox[2], hand_bbox[3]])

                current_top_left_point_hand = hand_positions[0] + (np.array([hand_bbox[0], hand_bbox[1]]) - hand_positions[0]) * damping_factory
                current_bottom_right_point_hand = hand_positions[1] + (np.array([hand_bbox[2], hand_bbox[3]]) - hand_positions[1]) * damping_factory

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

                # Draw landmarks on the segmented hand image
                mp_drawing.draw_landmarks(img, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                if is_fist(hand_landmarks):
                    fist_detected = True
                    fist_label = f"{handedness} Fist"

    # Punch counting logic with delay
    current_time = time.time()
    if fist_detected:
        if not previous_fist_state:
            fist_start_time = current_time
        elif current_time - fist_start_time >= fist_delay:
            punch_count += 1
            fist_start_time = current_time  # Reset the start time to avoid multiple counts for the same punch
    previous_fist_state = fist_detected


    if fist_detected:
        # Calculate the center coordinates of the overlay box
        overlay_center_x = overlay_box_top_left[0] + overlay_box_size // 2
        overlay_center_y = overlay_box_top_left[1] + overlay_box_size // 2

        # Calculate the top-left corner coordinates to position the fist_img at the center
        fist_x_min = overlay_center_x - fist_img.shape[1] // 2
        fist_y_min = overlay_center_y - fist_img.shape[0] // 2

        # Replace the portion of the image with the fist image
        img[fist_y_min:fist_y_min+fist_img.shape[0], fist_x_min:fist_x_min+fist_img.shape[1]] = fist_img

        cv2.putText(img, fist_label, (overlay_box_top_left[0] + 10, overlay_box_top_left[1] + 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 20, 147), 2, cv2.LINE_AA)

    # Display punch count
    cv2.putText(img, f"Punches: {punch_count}", (10, 30), 
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

            print(f"Left {counter}: {left_hand_slope}")
            counter += 1
        elif right_hand_slope > 600:
            key.press('u')
            key.release('u')

            print(f"Right {counter}: {right_hand_slope}")
            counter += 1

        hands_area[0] = []
        hands_area[1] = []

    # Write the frame to the output video file
    out.write(img)

    cv2.namedWindow('Screen', cv2.WINDOW_NORMAL)

    cv2.imshow("Screen", img)

    pressed_key = cv2.waitKey(8)
    if pressed_key == ord('q'):
        cap.release()
        break

cv2.destroyAllWindows()
