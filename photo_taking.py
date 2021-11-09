import cv2
import pandas as pd

# On the camera
cam = cv2.VideoCapture(0)

# Apply trained haarcascade model to cascade classifer for face detection
face_detector = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Request name from user
face_name = input('\n Enter your name: ')

# Read database of past entered names and indexes
df = pd.read_csv("index.csv")
# Add current user name into the database and provide the index of the current user and save the database
df = df.append({'name':face_name}, ignore_index=True)
face_id = len(df)-1
df.to_csv('index.csv', index=False)

# Inform current user that 30 images of them will be taken by the camera for training and identification later
print("\n 30 images of face will be taken as reference for identification")
# Initialize individual sampling face count
count = 0

while(True):
    # Pass camera capture into image and change image to grayscale
    dummy, img = cam.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Detect face from the image using the cascade classifier
    faces = face_detector.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5, minSize=(80, 80))

    for (x,y,w,h) in faces:
        # Drawing a rectangle to outline the face
        cv2.rectangle(img, (x,y), (x+w,y+h), (255,0,0), 2)
        # Count number of cropped faces taken
        count += 1
        # Save the captured image into the datasets folder
        cv2.imwrite("images/User." + str(face_id) + '.' + str(count) + ".jpg", gray[y:y+h,x:x+w])
        # Output camera capture image
        cv2.imshow('Photo taking session', img)

    # Press 'ESC' to exit
    k = cv2.waitKey(100) & 0xff
    if k == 27:
        break
    # Take 30 face images before exiting
    elif count >= 30:
         break

# Do a bit of cleanup
print("\n Closing Program")
cam.release()
cv2.destroyAllWindows()