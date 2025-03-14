PiMoji - Fun Eye Tracking with Pi Symbol
PiMoji is a computer vision project that detects eyes and overlays a π (Pi) symbol on them using OpenCV and MediaPipe.

 - Features
 Real-time Eye Tracking – Detects eyes using OpenCV or MediaPipe
 Pi Symbol Overlay – Places a Pi symbol on detected eyes
 Fun & Interactive – Experiment with different symbols

🛠️ Installation

1️⃣ Clone the Repository

cd PiMoji

2️⃣ Install Dependencies

Ensure you have Python installed, then install the required libraries:
pip install opencv-python
pip install mediapipe
pip install numpy
pip install pygame


📌 Usage
You have to choose an emoji.
The script will open your webcam.
It will detect your eyes and place the Pi symbol on them.
Can be closed with close option. 


🔧 Troubleshooting
⚠️ Pi symbol not appearing? Ensure the PNG file is in the correct directory.
⚠️ MediaPipe detects wrong features? Try adjusting landmark indices.
⚠️ ValueError: shape mismatch? Convert RGBA images to RGB using:

pi_resized = cv2.cvtColor(pi_resized, cv2.COLOR_BGRA2BGR)

📌 Creators & Maintainers:
Parishad Pourrajab 
Sidar Baran Oncel 
