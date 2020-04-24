import cv2

file_path_face = 'haarcascade_frontalface_default.xml'
file_path_eyes = 'haarcascade_eye.xml'
faceCascade = cv2.CascadeClassifier(file_path_face)
eyeCascade = cv2.CascadeClassifier(file_path_eyes)

# parameters
cap_region_x_begin = 0.5  # start point/total width
cap_region_y_end = 0.8  # start point/total width

cap = cv2.VideoCapture(0)
cap.set(10,200)

## Place your head in the blue box to detect faces.
## change the img to frame to detect the faces in the complete frame
while(True):
    ret, frame = cap.read()

    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)

    cv2.rectangle(frame, (int(cap_region_x_begin * frame.shape[1]), 0),
                  (frame.shape[1], int(cap_region_y_end * frame.shape[0])), (255, 0, 0), 2)

    img = frame[0:int(cap_region_y_end * frame.shape[0]),
          int(cap_region_x_begin * frame.shape[1]):frame.shape[1]]

    faces = faceCascade.detectMultiScale(img, 1.3, 5)
    for x,y,w,h in faces:
        cv2.rectangle(img, (x,y), (x+w, y+h), (0,255,0), 6)
        eyes = eyeCascade.detectMultiScale(img)
        for (ex, ey,ew,eh) in eyes:
            cv2.rectangle(img, (ex, ey), (ex + ew, ey + eh), (0, 0, 255), 6)

    frame = cv2.flip(frame, 1)
    # cv2.rectangle(frame, (300, 100), (500, 400), (0,255,0), 6)
    cv2.imshow('frame', frame)
    cv2.imshow('img', img)
    # cv2.imshow('gray', gray)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

