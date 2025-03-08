import cv2
import mediapipe as mp
import numpy as np
import pygame
import sys 
pygame.init()

# Set up display
window_size = (800, 600)
screen = pygame.display.set_mode(window_size)
pygame.display.set_caption("Choose an Emoji")

# Load images and resize them
button_size = (80, 80)
pi_image = pygame.image.load("pi_symbol_blue.png").convert_alpha()
pi_image = pygame.transform.scale(pi_image, button_size)
star_image = pygame.image.load("star_symbol.png").convert_alpha()
star_image = pygame.transform.scale(star_image, button_size)
laugh_image = pygame.image.load("laugh_symbol.png").convert_alpha()
laugh_image = pygame.transform.scale(laugh_image, button_size)

# Define button positions (centered in a row)
window_width, window_height = window_size
button_y = window_height // 2 - button_size[1] // 2
pi_button_rect = pygame.Rect(window_width // 4 - button_size[0] // 2, button_y, *button_size)
star_button_rect = pygame.Rect(window_width // 2 - button_size[0] // 2, button_y, *button_size)
laugh_button_rect = pygame.Rect(3 * window_width // 4 - button_size[0] // 2, button_y, *button_size)

# Draw buttons
def draw_buttons():
    screen.fill((255, 255, 255))
    screen.blit(pi_image, pi_button_rect)
    screen.blit(star_image, star_button_rect)
    screen.blit(laugh_image, laugh_button_rect)
    pygame.display.flip()

# Main loop for choosing the emoji
running = True
chosen_symbol = None
while running:
    draw_buttons()
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            sys.exit()
        elif event.type == pygame.MOUSEBUTTONDOWN:
            if pi_button_rect.collidepoint(event.pos):
                chosen_symbol = cv2.imread("pi_symbol_blue.png", cv2.IMREAD_UNCHANGED)
                running = False
            elif star_button_rect.collidepoint(event.pos):
                chosen_symbol = cv2.imread("star_symbol.png", cv2.IMREAD_UNCHANGED)
                running = False
            elif laugh_button_rect.collidepoint(event.pos):
                chosen_symbol = cv2.imread("laugh_symbol.png", cv2.IMREAD_UNCHANGED)
                running = False

pygame.quit()

# Initialize MediaPipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=10, refine_landmarks=True)


                    


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
                symbol_resized = cv2.resize(chosen_symbol, (eye_width, eye_height))

                # 🔹 Move Pi symbol **15 pixels up**
                x_offset, y_offset = max(0, x1), max(0, y1 - 15)
                x_end = min(x_offset + eye_width, frame.shape[1])
                y_end = min(y_offset + eye_height, frame.shape[0])

                # Ensure ROI dimensions are valid
                roi_width, roi_height = x_end - x_offset, y_end - y_offset
                if roi_width <= 0 or roi_height <= 0:
                    continue  # Skip if invalid dimensions

                # Resize Pi symbol to fit ROI (avoid OpenCV error)
                if symbol_resized.shape[0] != roi_height or symbol_resized.shape[1] != roi_width:
                    symbol_resized = cv2.resize(symbol_resized, (roi_width, roi_height))

                # Extract ROI safely
                roi = frame[y_offset:y_end, x_offset:x_end]

                # Handle alpha channel for transparency
                if symbol_resized.shape[2] == 4:  # RGBA image
                    alpha_channel = symbol_resized[:, :, 3] / 255.0
                    symbol_resized = cv2.cvtColor(symbol_resized, cv2.COLOR_BGRA2BGR)
                else:
                    alpha_channel = np.ones((roi_height, roi_width), dtype=np.float32)

                # Blend the Pi symbol onto the ROI
                for c in range(3):  # Loop over color channels (BGR)
                    roi[:, :, c] = (alpha_channel * symbol_resized[:, :, c] +
                                    (1 - alpha_channel) * roi[:, :, c])

    # Show result
    cv2.imshow("PiMoji", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    if cv2.getWindowProperty("PiMoji", cv2.WND_PROP_VISIBLE) < 1:
        break

cap.release()
cv2.destroyAllWindows()
pygame.quit()
