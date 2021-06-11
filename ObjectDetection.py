import cv2
import time
import mediapipe as mp

cap = cv2.VideoCapture('Sample/video1.mp4')

mpObjectron = mp.solutions.objectron
objectron = mpObjectron.Objectron()
mpDraw = mp.solutions.drawing_utils

c_time = 0
p_time = 0
while True:

    success, img = cap.read()
    img_RGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = objectron.process(img_RGB)
    print(results.detected_objects)
    if results.detected_objects:
        for detected_object in results.detected_objects:
            mpDraw.draw_landmarks(img, detected_object.landmarks_2d, mpObjectron.BOX_CONNECTIONS)
            # mpDraw.draw_axis(img, detected_object.rotation, detected_object.translation)

    c_time = time.time()
    fps = 1/(c_time-p_time)
    p_time = c_time

    cv2.putText(img, str(int(fps)), (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 3)

    cv2.imshow("Image",img)
    cv2.waitKey(1)