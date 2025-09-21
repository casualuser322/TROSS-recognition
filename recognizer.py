import cv2

class HaarFaceRecognizer:
    def __init__(self, cascade_path: str):
        self.face_cascade = cv2.CascadeClassifier(cascade_path)

    def detect_faces(self, gray_frame):
        return self.face_cascade.detectMultiScale(
            gray_frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)
        )

    def detection_callback(result):
        print("Face detected:", result.object_class.decode(),
            "x:", result.x, 
            "y:", result.y, 
            "w:", result.width, 
            "h:", result.height
        )

