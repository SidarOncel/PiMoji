import cv2
import mediapipe as mp
import numpy as np

# Initialize MediaPipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=10, refine_landmarks=True)

# Load Pi symbol image
pi_symbol = cv2.imread("pi_symbol_blue.png", cv2.IMREAD_UNCHANGED)  # Load with alpha channel if available

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
            if not face_landmarks or not face_landmarks.landmark:
                continue  # Skip processing if no landmarks are detected

            # Get eye landmark coordinates
            try:
                left_eye = [face_landmarks.landmark[i] for i in [33, 133]]
                right_eye = [face_landmarks.landmark[i] for i in [362, 263]]
            except IndexError:
                continue  # Skip if landmark indices are out of range

            h, w, _ = frame.shape  # Get frame dimensions

            for eye_landmarks in [left_eye, right_eye]:  # Loop for both eyes
                x1, y1 = int(eye_landmarks[0].x * w), int(eye_landmarks[0].y * h)
                x2, y2 = int(eye_landmarks[1].x * w), int(eye_landmarks[1].y * h)

                # Ensure valid coordinates
                if x1 < 0 or y1 < 0 or x2 < 0 or y2 < 0:
                    continue

                # 🔹 Increase Pi symbol size by 1.4x
                eye_width = max(int(abs(x2 - x1) * 1.4), 1)  # Ensure width is at least 1 pixel
                eye_height = max(int(eye_width * 0.6), 1)  # Ensure height is at least 1 pixel

                # Resize Pi symbol
                pi_resized = cv2.resize(pi_symbol, (eye_width, eye_height))

                # 🔹 Move Pi symbol **15 pixels up**
                x_offset, y_offset = max(0, x1), max(0, y1 - 15)
                x_end = min(x_offset + eye_width, frame.shape[1])
                y_end = min(y_offset + eye_height, frame.shape[0])

                # Ensure ROI dimensions are valid
                roi_width, roi_height = x_end - x_offset, y_end - y_offset
                if roi_width <= 0 or roi_height <= 0:
                    continue  # Skip if invalid dimensions

                # Resize Pi symbol to fit ROI (avoid OpenCV error)
                if pi_resized.shape[0] != roi_height or pi_resized.shape[1] != roi_width:
                    pi_resized = cv2.resize(pi_resized, (roi_width, roi_height))

                # Extract ROI safely
                roi = frame[y_offset:y_end, x_offset:x_end]

                # Handle alpha channel for transparency
                if pi_resized.shape[2] == 4:  # RGBA image
                    alpha_channel = pi_resized[:, :, 3] / 255.0
                    pi_resized = cv2.cvtColor(pi_resized, cv2.COLOR_BGRA2BGR)
                else:
                    alpha_channel = np.ones((roi_height, roi_width), dtype=np.float32)

                # Blend the Pi symbol onto the ROI
                for c in range(3):  # Loop over color channels (BGR)
                    roi[:, :, c] = (alpha_channel * pi_resized[:, :, c] +
                                    (1 - alpha_channel) * roi[:, :, c])

    # Show result
    cv2.imshow("PiMoji", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()