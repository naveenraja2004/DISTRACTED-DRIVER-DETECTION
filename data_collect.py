# import cv2
# import mediapipe as mp
# import numpy as np
# import pandas as pd

# mp_face_mesh = mp.solutions.face_mesh
# mp_drawing = mp.solutions.drawing_utils

# # Classes
# classes = ['Drowsy', 'Safe', 'Yawn', 'Face_tilt']

# # Video capture
# cap = cv2.VideoCapture(0)

# data = []  # To store landmarks + label

# # Choose label for current session (e.g., Yawn)
# label = "Yawn"

# MAX_SAMPLES = 1000  # Stop after collecting 1000 samples

# with mp_face_mesh.FaceMesh(
#     max_num_faces=1,
#     refine_landmarks=True,
#     min_detection_confidence=0.5,
#     min_tracking_confidence=0.5
# ) as face_mesh:
#     while len(data) < MAX_SAMPLES:
#         ret, frame = cap.read()
#         if not ret:
#             break

#         frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#         results = face_mesh.process(frame_rgb)

#         if results.multi_face_landmarks:
#             for face_landmarks in results.multi_face_landmarks:
#                 # Draw landmarks
#                 mp_drawing.draw_landmarks(frame, face_landmarks, mp_face_mesh.FACEMESH_CONTOURS)

#                 # Extract x, y, z coordinates
#                 landmarks = []
#                 for lm in face_landmarks.landmark:
#                     landmarks.extend([lm.x, lm.y, lm.z])

#                 # Append landmarks with label
#                 data.append(landmarks + [label])

#         cv2.imshow("MediaPipe FaceMesh", frame)
#         if cv2.waitKey(1) & 0xFF == 27:  # Optional: ESC to quit early
#             break

# cap.release()
# cv2.destroyAllWindows()

# # Save data to CSV
# df = pd.DataFrame(data)
# df.to_csv("facial_landmarks_eye_open.csv", index=False)
# print(f"Data saved! Total samples collected: {len(data)}")
import cv2
import mediapipe as mp
import numpy as np
import pandas as pd

mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils

# Define your classes
classes = ['Drowsy', 'Safe', 'Yawn', 'Face_tilt']

# Choose which class you are collecting now
label = "Face_tilt"  # Change for each session

# Number of samples per class
MAX_SAMPLES = 3000

# Video capture
cap = cv2.VideoCapture(0)

data = []  # Store landmarks + label

with mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
) as face_mesh:
    while len(data) < MAX_SAMPLES:
        ret, frame = cap.read()
        if not ret:
            break

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(frame_rgb)

        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                # Draw face landmarks
                mp_drawing.draw_landmarks(frame, face_landmarks, mp_face_mesh.FACEMESH_CONTOURS)

                # Extract x, y, z coordinates
                landmarks = []
                for lm in face_landmarks.landmark:
                    landmarks.extend([lm.x, lm.y, lm.z])

                # Append landmarks + label
                data.append(landmarks + [label])

                # Display label and sample count on the frame
                cv2.putText(frame, f"{label} | Count: {len(data)}", (30, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        cv2.imshow("MediaPipe FaceMesh", frame)
        if cv2.waitKey(1) & 0xFF == 27:  # Press ESC to quit early
            break

cap.release()
cv2.destroyAllWindows()

# Save collected data to CSV with 'class' as last column
df = pd.DataFrame(data)
df.rename(columns={df.columns[-1]: "class"}, inplace=True)
df.to_csv(f"facial_landmarks_{label}.csv", index=False)
print(f"Data saved! Total samples collected for {label}: {len(data)}")
