import cv2
import mediapipe as mp
import numpy as np

# Initialize MediaPipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, refine_landmarks=True)

# Load Pi symbol image
pi_symbol = cv2.imread("pi_symbol.png", cv2.IMREAD_UNCHANGED)  # Load with alpha channel if available

# Open webcam
cap = cv2.VideoCapture(0)


while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Convert frame to RGB (required for MediaPipe)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb_frame)

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            # Get eye landmark coordinates
            left_eye = [face_landmarks.landmark[i] for i in [33, 133]]  # Left eye (two key points)
            right_eye = [face_landmarks.landmark[i] for i in [362, 263]]  # Right eye (two key points)

            h, w, _ = frame.shape  # Get frame dimensions

            for eye_landmarks in [left_eye, right_eye]:  # Loop for both eyes
                x1, y1 = int(eye_landmarks[0].x * w), int(eye_landmarks[0].y * h)
                x2, y2 = int(eye_landmarks[1].x * w), int(eye_landmarks[1].y * h)

                eye_width = abs(x2 - x1)
                eye_height = int(eye_width * 0.6)  # Adjust height proportionally

                # Resize Pi symbol to match eye size
                pi_resized = cv2.resize(pi_symbol, (eye_width, eye_height))

                if pi_resized.shape[2] == 4:  # Check if image has 4 channels (RGBA)
                    pi_resized = cv2.cvtColor(pi_resized, cv2.COLOR_BGRA2BGR)  # Convert to 3 channels (RGB)

                # Overlay Pi symbol
                x_offset, y_offset = x1, y1
                frame[y_offset:y_offset+eye_height, x_offset:x_offset+eye_width] = pi_resized



    # Show result
    cv2.imshow("PiMoji", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
