import cv2
import mediapipe as mp
import time


def init():
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    previousTime = 0
    currentTime = 0
    mpFaceMesh = mp.solutions.face_mesh
    face = mpFaceMesh.FaceMesh(max_num_faces=2)
    mpDraw = mp.solutions.drawing_utils
    drawSpec = mpDraw.DrawingSpec(thickness=1, circle_radius=2)
    while True:
        success, img = cap.read()
        imageRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = face.process(imageRGB)

        # get the information of each framework only when a hand is detected.
        if results.multi_face_landmarks:
            for faceLandMarks in results.multi_face_landmarks:
                mpDraw.draw_landmarks(img, faceLandMarks, mpFaceMesh.FACE_CONNECTIONS, drawSpec, drawSpec)

        # compute the fps rate (frame per second)
        currentTime = time.time()
        fps = 1 / (currentTime - previousTime)
        previousTime = currentTime
        cv2.putText(img, str(int(fps)), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (100, 200, 250), 3)


        cv2.imshow("Image", img)
        cv2.waitKey(1)
    cv2.destroyAllWindows()


if __name__ == '__main__':
    init()