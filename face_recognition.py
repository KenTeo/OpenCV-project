import cv2
import pandas as pd

font = cv2.FONT_HERSHEY_PLAIN

# Define LBPH face recognizer and apply trained model in train_images.py
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read('faces.yml')
# Define face_detector
detector = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml");

# initiate id counter
id = 0

# Read past user list and map user to index
df = pd.read_csv("index.csv")
names = list(df.name.values.flatten())

# On the camera
cam = cv2.VideoCapture(0)

while True:
    # Pass camera capture into image and change image to grayscale
    dummy, img = cam.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Detect face from the image using the cascade classifier
    faces = detector.detectMultiScale(
        gray,
        scaleFactor=1.2,
        minNeighbors=5,
        minSize=(80, 80),
    )

    for (x, y, w, h) in faces:
        # Add rectangle to outline face
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
        id, confidence = recognizer.predict(gray[y:y + h, x:x + w])
        # When face recognizer is confident of the prediction, show name of person. Else, show blank.
        if (confidence < 100):
            id = names[id]
        else:
            id = " "
        # Place name of person above their face
        cv2.putText(img, str(id), (x + 5, y - 5), font, 1, (255, 255, 255), 2)

    cv2.imshow('Face Recognition', img)

    # Press 'ESC' to exit program
    k = cv2.waitKey(10) & 0xff
    if k == 27:
        break

# End program
print("\n Exiting...")
cam.release()
cv2.destroyAllWindows()
