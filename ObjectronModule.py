import cv2
import time
import mediapipe as mp

class ObjectDetection:

    def __init__(self, mode=True, maxObj = 5, detectionCon = 0.5, modelName = 'Camera'):

        self.mode = mode
        self.maxObj = maxObj
        self.detectionCon = detectionCon
        self.modelName = modelName

        self.mpObjectron = mp.solutions.objectron
        self.objectron = self.mpObjectron.Objectron(static_image_mode = self.mode,
                                                    max_num_objects = self.maxObj,
                                                    min_detection_confidence = self.detectionCon,
                                                    model_name = self.modelName)
        self.mpDraw = mp.solutions.drawing_utils

    def findObject(self, img, draw = True):

            self.img_RGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            self.results = self.objectron.process(self.img_RGB)

            lmList = []

            if self.results.detected_objects:
                for detected_object in self.results.detected_objects:
                    if(draw):
                        self.mpDraw.draw_landmarks(img, detected_object.landmarks_2d, self.mpObjectron.BOX_CONNECTIONS)
            return img



def main():
    cap = cv2.VideoCapture('Sample/video.mp4')

    detector = ObjectDetection()

    p_time = 0
    while True:
        success, img = cap.read()

        img = detector.findObject(img)

        c_time = time.time()
        fps = 1 / (c_time - p_time)
        p_time = c_time

        cv2.putText(img, str(int(fps)), (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 3)

        cv2.imshow("Image", img)
        cv2.waitKey(1)

if __name__ == '__main__':
    main()