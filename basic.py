import cv2
import mediapipe as mp
import time
import math

def zoom_out(image, zoom_factor):
    h, w = image.shape[:2]
    new_h, new_w = int(h / zoom_factor), int(w / zoom_factor)
    resized_img = cv2.resize(image, (new_w, new_h))
    return resized_img

cap = cv2.VideoCapture(0)

mpHands = mp.solutions.hands
hands = mpHands.Hands()
mpDraw = mp.solutions.drawing_utils

pTime = 0
cTime = 0
zoom_factor = 1.0

while True:
    success, img = cap.read()
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(imgRGB)

    if results.multi_hand_landmarks:
        for handLms in results.multi_hand_landmarks:
            thumb_tip = handLms.landmark[mpHands.HandLandmark.THUMB_TIP]
            index_finger_tip = handLms.landmark[mpHands.HandLandmark.INDEX_FINGER_TIP]
            thumb_x, thumb_y = int(thumb_tip.x * img.shape[1]), int(thumb_tip.y * img.shape[0])
            index_finger_x, index_finger_y = int(index_finger_tip.x * img.shape[1]), int(index_finger_tip.y * img.shape[0])
            distance = math.sqrt((index_finger_x - thumb_x) ** 2 + (index_finger_y - thumb_y) ** 2)
            print("Distance:", distance)

            if distance > 200:
                zoom_factor = max(zoom_factor * 0.9, 1.0)
            cv2.circle(img, (thumb_x, thumb_y), 15, (255, 0, 255), cv2.FILLED)
            cv2.circle(img, (index_finger_x, index_finger_y), 15, (255, 0, 255), cv2.FILLED)

            mpDraw.draw_landmarks(img, handLms, mpHands.HAND_CONNECTIONS)
    resized_image = zoom_out(img, zoom_factor)

    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime

    cv2.putText(resized_image, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3)

    cv2.imshow("Image", resized_image)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
