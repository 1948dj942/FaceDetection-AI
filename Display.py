import cv2
import tensorflow as tf
import numpy as np

filename = 'test\group.png'
image = cv2.imread(filename)
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades +'haarcascade_frontalface_default.xml')
save_model = tf.keras.models.load_model("face.h5")
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
fa = face_cascade.detectMultiScale(gray, 1.1, 5)
fontface = cv2.FONT_HERSHEY_SIMPLEX
for (x, y, w, h) in fa:2
    cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)
    roi_gray = gray[y:y+h, x:x+w]
    roi_gray = cv2.resize(src=roi_gray, dsize=(100,100))
    roi_gray = roi_gray.reshape((100,100,1))
    roi_gray = np.array(roi_gray)
    result = save_model.predict(np.array([roi_gray]))
    final = np.argmax(result)
    if final == 0:
        cv2.putText(image, "Hong Anh",(x+10,y+h+ 30), fontface, 1, (0,255,0),2)
    if final == 1:
        cv2.putText(image, "Quoc Trung",(x+10,y+h+ 30), fontface, 1, (0,255,0),2)
    if final == 2:
        cv2.putText(image, "Phuc Toan",(x+10,y+h+ 30), fontface, 1, (0,255,0),2)
    if final == 3:
        cv2.putText(image, "Elon Musk",(x+10,y+h+ 30), fontface, 1, (0,255,0),2)
    if final == 4:
        cv2.putText(image, "Chadwick",(x+10,y+h+ 30), fontface, 1, (0,255,0),2)
cv2.imshow('trainning',image)
print(result)
cv2.waitKey(0)
cv2.destroyAllWindows()
