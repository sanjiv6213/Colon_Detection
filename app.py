import cv2
import mediapipe as mp

def zoom_in(image, zoom_factor):
    h, w = image.shape[:2]
    new_h, new_w = int(h * zoom_factor), int(w * zoom_factor)
    resized_img = cv2.resize(image, (new_w, new_h))
    return resized_img

def zoom_out(image, zoom_factor):
    h, w = image.shape[:2]
    new_h, new_w = int(h / zoom_factor), int(w / zoom_factor)
    resized_img = cv2.resize(image, (new_w, new_h))
    return resized_img

def main():
    # Initialize mediapipe hands module
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands()

    # Initialize camera
    cap = cv2.VideoCapture(0)

    # Initialize zoom factor
    zoom_factor = 1.0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Convert the image to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Detect hands in the frame
        results = hands.process(frame_rgb)

        # Check if hand is detected
        if results.multi_hand_landmarks:
            # Get the landmarks of the first hand
            hand_landmarks = results.multi_hand_landmarks[0]

            # Get the index finger tip coordinates
            index_finger_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
            index_finger_x = int(index_finger_tip.x * frame.shape[1])
            index_finger_y = int(index_finger_tip.y * frame.shape[0])

            # Perform zoom in or out based on finger position
            if index_finger_y < frame.shape[0] // 3:  # Zoom in
                zoom_factor *= 1.1
            elif index_finger_y > 2 * frame.shape[0] // 3:  # Zoom out
                zoom_factor = max(zoom_factor / 1.1, 1.0)  # Ensure zoom factor doesn't go below 1.0

        # Resize the image based on the zoom factor
        resized_image = zoom_out(frame, zoom_factor)

        # Display the resized image
        cv2.imshow('Frame', resized_image)

        # Break the loop when 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
