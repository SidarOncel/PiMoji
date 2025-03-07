PiMoji - Fun Eye Tracking with Pi Symbol
PiMoji is a computer vision project that detects eyes and overlays a π (Pi) symbol on them using OpenCV and MediaPipe.

 - Features
 Real-time Eye Tracking – Detects eyes using OpenCV or MediaPipe
 Pi Symbol Overlay – Places a Pi symbol on detected eyes
 Customizable Detection – Switch between Haar cascades and MediaPipe
 Fun & Interactive – Experiment with different symbols

📌 Usage
The script will open your webcam.
It will detect your eyes and place the Pi symbol on them.
Press 'q' to exit the program.

🔧 Troubleshooting
⚠️ Pi symbol not appearing? Ensure the PNG file is in the correct directory.
⚠️ MediaPipe detects wrong features? Try adjusting landmark indices.
⚠️ ValueError: shape mismatch? Convert RGBA images to RGB using:

pi_resized = cv2.cvtColor(pi_resized, cv2.COLOR_BGRA2BGR)

 - Future Improvements:
🔹 More Symbols & Emojis – Allow users to choose from various symbols/emojis.

🔹 Interactive GUI – Add a simple UI for customizing overlays.
