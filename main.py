""""
    * A PYTHON MACHINE LEARNING PROJECT
    * This is a simple python matchine learning project that dedects
    the cars from a video
"""
from pip._internal import main as install
try:
    import cv2
except ImportError as e:
    install(["install", "opencv-python"])
finally:
    pass
# capturing the frame for the videos
cap = cv2.VideoCapture("cars3.mp4")
# trained XML classifiers describes some features of some objects we want to detect
cars_casicade = cv2.CascadeClassifier("cars.xml")
# loop runs if the capturing has been initialised
while True:
    # reading frame from a video
    ret, frame = cap.read()
    # change the color to gray scale of each frame
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # detect the car from diffrent sizes in the input image
    cars = cars_casicade.detectMultiScale(gray, 1.1, 1)
    # drawing a rectangle for each car
    i =0
    for (x, y, w, h) in cars:
        i+=1
        cv2.putText(frame, 'Car {}'.format(i), (x+ int(w/2) -10, y-5), cv2.FONT_HERSHEY_PLAIN,  1, (0, 0, 255), 2)
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0,255,0), 2)
    # displaying the frame on the window
    cv2.imshow("CAR DETECTOR", frame)
    # wait for evideo2sc key to quit the app
    if cv2.waitKey(33) == 27:
        break
# De-allocate any associated memory usage
cv2.destroyAllWindows()
