import cv2
import mediapipe as mp
import numpy as np

WHITE = (255, 255, 255)

FONT = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE = 3
FONT_THICK = 3

CAPTION = "HandPosition"
cv2.namedWindow(CAPTION, cv2.WINDOW_FREERATIO)

def RefreshWindow(hand_up, text = None):
    if hand_up:
        window_x, window_y, window_w, window_h = cv2.getWindowImageRect(CAPTION)
        frame = np.zeros((window_w, window_h, 3), np.uint8)
        if text:
            text_size, text_x  = cv2.getTextSize(text, FONT, FONT_SCALE, FONT_THICK)
            cv2.putText(frame, text, (int(window_w/2 - text_size[0]/2), int(window_h/2 + text_size[1]/2)), FONT, FONT_SCALE, WHITE, FONT_THICK)
        cv2.imshow(CAPTION, frame)
        hand_up = False

# Better use a class
def DetectHandPos(keypoint):
    keypoint_x = keypoint[0]
    keypoint_y = keypoint[1]

    # print(keypoint_y)
    hand_up = False
    # if keypoint_y >= 0.5:
        # RefreshWindow("Down")
    if keypoint_y < 0.7:
        hand_up = True
        # RefreshWindow("UP")

    if keypoint_x < 0.2:
        RefreshWindow(hand_up, "d")
    elif keypoint_x >= 0.2 and  keypoint_x < 0.5:
        RefreshWindow(hand_up, "f")
    elif keypoint_x >= 0.5 and  keypoint_x < 0.8:
        RefreshWindow(hand_up, "j")
    else:
        RefreshWindow(hand_up, "k")

    

def GetHandPos():
    cap = cv2.VideoCapture(0)
    if cap.isOpened():
        mpHands = mp.solutions.hands
        hands = mpHands.Hands(
                static_image_mode = False,
                max_num_hands = 2,
                min_detection_confidence = 0.5,
                min_tracking_confidence = 0.5)

        while True:
            ret, frame = cap.read()
            if ret:
                frame = cv2.flip(frame, 1)
                frame_RGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                result = hands.process(frame_RGB)

                if result.multi_hand_landmarks:
                    hand_num = len(result.multi_hand_landmarks) 
                    # print(hand_num)
                    for i in range(hand_num):
                        for hands_LMS in result.multi_hand_landmarks:
                            keypoint_pos_rate = []
                            for i in range(21):
                                x = hands_LMS.landmark[i].x
                                y = hands_LMS.landmark[i].y
                                keypoint_pos_rate.append((x, y))
                            if keypoint_pos_rate:
                                DetectHandPos(keypoint_pos_rate[12])
                            
            if cv2.waitKey(1) == ord('q'):
                break
        cap.release()

GetHandPos()
