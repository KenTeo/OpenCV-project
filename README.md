
# OpenCV Project
### **Project** - *To develop a Face Recognition tool using OpenCV.*
![image](https://drive.google.com/uc?export=view&id=1xgd_9d20fU0WoffgsxIMlGAWU9pOvdS_)
This project is split into 3 main tasks:
1. Identify face(s) and capture photos for training. 
2. Train the model using the photos taken in 1.
3. Execute a python file to identify the identity of the face(s).
___
#### 1. Identify face(s) from the camera's capture.

![image](https://drive.google.com/uc?export=view&id=1lxrjfZhR-JWHgAMny12AieqSJI_Wdhph)
Trained face model in haarcascade will be used with OpenCV to identify faces.
Training Neutral Networks to identify faces requires a lot of computational power.
Hence, pre-trained models from haarcascade is used.

**Advantages** of using haarcascade
- Proven model in identifying human faces
- Efficient model where faces can be identified very quickly
- Simple to implement

*Import OpenCV*
```python
import cv2
```
*On the Camera(720p), "0" below refers to the index of the cam. Change the value according to your system*
```python
cam = cv2.VideoCapture(0)
```
*Apply pre-trained haarcascade_frontalface model onto OpenCV*
```python
face_detector = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
```
This sets the weights and constants for the neural networks in the CascadeClassifier.
Now the CascadeClassifier is ready to identify frontal faces of humans.

**Note**:As "haarcascade_frontalface_default.xml" is trained on frontal faces, it is not able to identify faces when faces are tilt away from the camera and showing less of the frontal face to the camera. This is acceptable because users of the software are aware that they are required to face straight at the camera. Thus, this will not pose any problem for the the function of the face recognition tool.

*Store image from camera in "img". Then, convert image into grayscale*
```python
dummy, img = cam.read()
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
```
Trained model is in grayscale as colour is not required in identifying faces. Grayscale allows faster computation as it
has less information as compared to RGB images. Therefore, training and prediction is faster using grayscale.

*Using the trained model to identify frontal faces in the grayscale image taken*
```python
faces = face_detector.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5)
```
*Using the coordinates of the 4 corners of the crop faces identified by the trained cascade classifier. 
Save the the crop faces into a folder*
```python
    for (x,y,w,h) in faces:
        cv2.rectangle(img, (x,y), (x+w,y+h), (255,0,0), 2)
        cv2.imwrite("images/User." + str(face_id) + '.' + str(count) + ".jpg", gray[y:y+h,x:x+w])
```
---
#### 2. Trained cropped frontal faces images taken to identify individuals.

![image](https://drive.google.com/uc?export=view&id=1OHY4pqK5Dv2tuebVkt5EzYjaspA9V62c)
There are 3 face recognition algorithm packaged in OpenCV. 
They are:
1.  Eigenfaces
2.  Fisherfaces
3.  Local Binary Patterns Histograms (LBPH)

In this project, we will be using LBPH for face recognition.
```python
recognizer = cv2.face.LBPHFaceRecognizer_create()
```
*Store the cropped images in faces and the identity of the faces in ids.
Then, train the images with LBPH face recognition algorithm as provided by OpenCV*
```python
recognizer.train(faces, np.array(ids))
```
*Save the trained model as yml format*
```python
recognizer.write('faces.yml')
```
---
#### 3. Identify the identity of faces using trained model "faces.yml"

![image](https://drive.google.com/uc?export=view&id=16xRlkoDQtA3hOgEbnY46389gyzfNAMtF)
*Load the trained faces.yml into the LBPH recognizer*
```python
recognizer.read('faces.yml')
```
*Using the recognizer to predict the identity of the face capture by the camera.
The identity and confidence of the prediction by the recognizer will be churn out by the recognizer*
```python
id, confidence = recognizer.predict(gray[y:y + h, x:x + w])
````
---
## Executing the Face recognizer project
Download project file and install the necessary packages.

*To capture facial photos for training. Do the following:*
```python
python photo_taking.py
```
Enter your name when prompted and your capture facial images will be save with your index that is mapped to your name.
You can run this code with multiple individual.

*Trained the facial images*
```python
python train_images.py
```
Run this when new images are added in images folder with "python photo_taking.py" to update the LBPH recognizer trained model.

*Run the face recognition program*
```python
python face_recognition.py
```
Identity of person will be shown above the face when recognizer is able to recognize the trained face.
As shown below:
![image](https://drive.google.com/uc?export=view&id=1tOtphnC2Dr1CcPWLYP0NpdXnepVazNnj)










