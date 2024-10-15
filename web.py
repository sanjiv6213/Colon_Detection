import cv2
import numpy as np
from keras.preprocessing import image
from keras.models import load_model
from cvzone.HandTrackingModule import HandDetector
import matplotlib.pyplot as plt

model = load_model("colon1.h5")
class_labels = {
    0: "Normal",
    1: "Ulcerative colitis",
    2: "Polyp",
    3: "Esophagitis"
}

def preprocess_image(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (224, 224))
    img_array = np.expand_dims(img, axis=0)
    img_array = img_array / 255.0  
    return img_array

def predict_colon_class(img):
    img_array = preprocess_image(img)
    predictions = model.predict(img_array)
    predicted_class_index = np.argmax(predictions)
    predicted_class_label = class_labels[predicted_class_index]
    return predicted_class_label

cap = cv2.VideoCapture(0)
cap.set(3, 1280)
cap.set(4, 720)

detector = HandDetector(detectionCon=0.7)
startDist = None
scale = 0
cx, cy = 500, 500

while True:
    success, img = cap.read()

    if not success:
        print("Failed to capture frame")
        break

    hands, img = detector.findHands(img)
    if len(hands) == 2:
        if detector.fingersUp(hands[0]) == [1, 1, 0, 0, 0] and \
                detector.fingersUp(hands[1]) == [1, 1, 0, 0, 0]:
            lmList1 = hands[0]["lmList"]
            lmList2 = hands[1]["lmList"]
            if startDist is None:
                length, info, img = detector.findDistance(hands[0]["center"], hands[1]["center"], img)
                startDist = length

            length, info, img = detector.findDistance(hands[0]["center"], hands[1]["center"], img)

            scale = int((length - startDist) // 2)
            cx, cy = info[4:]
            print(scale)
    else:
        startDist = None

    try:
        colon_img = cv2.imread("train.jpg")
        colon_class = predict_colon_class(colon_img)
        cv2.putText(img, colon_class, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
    except Exception as e:
        print("Error:", e)

    cv2.imshow("Image", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
