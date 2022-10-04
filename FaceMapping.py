from imutils import face_utils
import dlib
import cv2


# The face detector (HOG) is initialized first, and then the reference points are added on the detected face

# p = The reference to the pretrained model, in this case in the same folder as this script
p = "shape_predictor_68_face_landmarks.dat"
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(p)

cap = cv2.VideoCapture(0)
 
while True:
    # The webcam image is obtained
    _, image = cap.read()

    #  Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
    # The faces of the image are obtained in the webcam
    rects = detector(gray, 0)
    
    # For each face detected, the reference points are searched
    for (i, rect) in enumerate(rects):
    
        # The prediction is made and transformed into a numpy array
        shape = predictor(gray, rect)
        shape = face_utils.shape_to_np(shape)
    
        # It is drawn in our image, all the found coordinates (x,y)
        for (x, y) in shape:
            cv2.circle(image, (x, y), 2, (0, 255, 0), -1)
    
    #  A window with the image is displayed
    cv2.imshow("Face Mapping", image)
    
    # The window closes with the Esc key
    k = cv2.waitKey(5) & 0xFF
    if k == 27:
        break

cv2.destroyAllWindows()
cap.release()
