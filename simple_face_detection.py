#open notebook: python -m notebook
import cv2

cascade_path = cv2.data.haarcascades+'haarcascade_frontalface_default.xml'

clf = cv2.CascadeClassifier(cascade_path) #read haarcascade_frontalface_default.xml by cv2.CascadeClassifier (a machine learning based approach where a cascade function is trained from a lot of positive and negative images. It is then used to detect objects in other images.)


camera = cv2.VideoCapture(0) #Open Camera

while True:
    _, frame = camera.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  #cv2.cvtColor() method is used to convert an image from one color space to another
    #Create a dictionary detects objects of different sizes in the input image. The detected objects are returned as a list of rectangles.
    faces = clf.detectMultiScale(
        gray,
        scaleFactor=1.1, #Parameter specifying how much the image size is reduced at each image scale.
        minNeighbors=5, #Parameter specifying how many neighbors each candidate rectangle should have to retain it.
        minSize=(30, 30), #Minimum possible object size. Objects smaller than that are ignored.
        flags=cv2.CASCADE_SCALE_IMAGE #Parameter with the same meaning for an old cascade as in the function cvHaarDetectObjects. It is not used for a new cascade.
    )

    for (x, y, width, height) in faces:
        cv2.rectangle(frame, (x, y), (x+width, y+height), (255, 255, 0), 2)  #cv2.rectangle() method is used to draw a rectangle on any image.

    cv2.imshow("Faces", frame) #cv2.imshow() method is used to display an image in a window
    if cv2.waitKey(1) == ord("q"): #waitkey() function of Python OpenCV allows users to display a window for given milliseconds or until any key is pressed
        break


camera.release() #Release camera
cv2.destroyAllWindows() #allows users to destroy or close all windows at any time after exiting the script
