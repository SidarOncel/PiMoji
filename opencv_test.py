import cv2
import numpy as np
# Load Haar cascade classifiers for face and eyes
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_eye.xml")

# Load the Pi symbol (ensure it's a transparent PNG)
pi_symbol = cv2.imread("pi_symbol.png", cv2.IMREAD_UNCHANGED)

# Open webcam
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # Convert to grayscale

    faces = face_cascade.detectMultiScale(gray, 1.3, 5)  # Detect faces
    for (x, y, w, h) in faces:  # Loop through detected faces
        roi_gray = gray[y:y + h, x:x + w]  # Region of interest (face)
        roi_color = frame[y:y + h, x:x + w]  # Color version of the face

        eyes = eye_cascade.detectMultiScale(roi_gray)  # Detect eyes

        for (ex, ey, ew, eh) in eyes[:2]:  # Limit to two eyes
            eye_x, eye_y = x + ex, y + ey  # Get eye position in original frame

            # Resize Pi symbol to fit the eyes
            pi_resized = cv2.resize(pi_symbol, (ew, eh))

            # Overlay Pi symbol on the eyes (only for non-transparent pixels)
            for i in range(eh):
                for j in range(ew):
                    if pi_resized.shape[2] == 4 and pi_resized[i, j][3] > 0:  # Check alpha channel
                        frame[eye_y + i, eye_x + j] = pi_resized[i, j][:3]  # Copy only RGB

    cv2.imshow("Pi Symbol on Eyes", frame)    #Display the Webcam feed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()          # Release webcam and close window
cv2.destroyAllWindows()
