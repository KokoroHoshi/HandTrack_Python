import cv2, math, time
import mediapipe as mp

WINDOW_TITLE = "HandTrack"

# BGR_COLOR
BGR_BLUE = (255, 0, 0)
BGR_GREEN = (0, 255, 0)
BGR_RED = (0, 0, 255)
BGR_WHITE = (255, 255, 255)

cap = cv2.VideoCapture(0)
# cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
# cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 960)
cv2.namedWindow(WINDOW_TITLE, cv2.WINDOW_NORMAL)

mpHands = mp.solutions.hands
hands = mpHands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.9,
            min_tracking_confidence=0.3)

mpDraw = mp.solutions.drawing_utils
HandLMS_style = mpDraw.DrawingSpec(color = BGR_RED, thickness = 3)
HandCON_style = mpDraw.DrawingSpec(color = BGR_WHITE, thickness = 2)

cTime = 0
pTime = 0

def mosaic(img, left_up, right_down):
    new_img = img.copy()
    size = 10
    for i in range(left_up[1], right_down[1]-size-1, size):
        for j in range(left_up[0], right_down[0]-size-1, size):
            try:
                new_img[i:i + size, j:j + size] = img[i, j, :]
            except:
                pass
    return new_img

def vector_2d_angle(v1,v2):
    v1_x=v1[0]
    v1_y=v1[1]
    v2_x=v2[0]
    v2_y=v2[1]
    try:
        angle_= math.degrees(math.acos((v1_x*v2_x+v1_y*v2_y)/(((v1_x**2+v1_y**2)**0.5)*((v2_x**2+v2_y**2)**0.5))))
    except:
        angle_ = -1
    return angle_

def hand_angle(hand_):
    angle_list = []
    # thumb
    angle_ = vector_2d_angle(
        ((int(hand_[0][0])- int(hand_[2][0])),(int(hand_[0][1])-int(hand_[2][1]))),
        ((int(hand_[3][0])- int(hand_[4][0])),(int(hand_[3][1])- int(hand_[4][1])))
        )
    angle_list.append(angle_)
    # index
    angle_ = vector_2d_angle(
        ((int(hand_[0][0])-int(hand_[6][0])),(int(hand_[0][1])- int(hand_[6][1]))),
        ((int(hand_[7][0])- int(hand_[8][0])),(int(hand_[7][1])- int(hand_[8][1])))
        )
    angle_list.append(angle_)
    # middle
    angle_ = vector_2d_angle(
        ((int(hand_[0][0])- int(hand_[10][0])),(int(hand_[0][1])- int(hand_[10][1]))),
        ((int(hand_[11][0])- int(hand_[12][0])),(int(hand_[11][1])- int(hand_[12][1])))
        )
    angle_list.append(angle_)
    # ring
    angle_ = vector_2d_angle(
        ((int(hand_[0][0])- int(hand_[14][0])),(int(hand_[0][1])- int(hand_[14][1]))),
        ((int(hand_[15][0])- int(hand_[16][0])),(int(hand_[15][1])- int(hand_[16][1])))
        )
    angle_list.append(angle_)
    # pink
    angle_ = vector_2d_angle(
        ((int(hand_[0][0])- int(hand_[18][0])),(int(hand_[0][1])- int(hand_[18][1]))),
        ((int(hand_[19][0])- int(hand_[20][0])),(int(hand_[19][1])- int(hand_[20][1])))
        )
    angle_list.append(angle_)
    return angle_list

def hand_gesture(angle_list):
    gesture_str = None
    if -1 not in angle_list:
        if (angle_list[1]>40) and (angle_list[2]<40) and (angle_list[3]>40) and (angle_list[4]>40):
            gesture_str = "middle"
    return gesture_str


while True:
    window_x, window_y, window_w, window_h = cv2.getWindowImageRect(WINDOW_TITLE)

    ret, img = cap.read()
    if ret:
        img = cv2.flip(img, 1)
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        result = hands.process(imgRGB)

        if result.multi_hand_landmarks:
            for handLMS in result.multi_hand_landmarks:
                mpDraw.draw_landmarks(img, handLMS, mpHands.HAND_CONNECTIONS, HandLMS_style, HandCON_style)
                keypoint_pos = []
                for i in range(21):
                    x = handLMS.landmark[i].x*img.shape[1]
                    y = handLMS.landmark[i].y*img.shape[0]
                    keypoint_pos.append((x,y))
                if keypoint_pos:
                    angle_list = hand_angle(keypoint_pos)

                    cv2.putText(img, f"middle angle:{int(angle_list[2])}", (30, 50), cv2.FONT_HERSHEY_COMPLEX, 1, BGR_WHITE, 1)

                    gesture_str = hand_gesture(angle_list)
                    if gesture_str == "middle":
                        for node in range(9, 13):
                            center_x = int(keypoint_pos[node][0])
                            center_y = int(keypoint_pos[node][1])
                            img = mosaic(img, [center_x - 15 , center_y - 10], [center_x + 30, center_y + 50])
        
        cTime = time.time()
        fps = 1/(cTime-pTime)
        pTime = cTime
        cv2.putText(img, f"fps:{int(fps)}", (550, 460), cv2.FONT_HERSHEY_COMPLEX, 0.8, BGR_WHITE, 1)
        cv2.imshow(WINDOW_TITLE, img)

    if cv2.waitKey(1) == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()