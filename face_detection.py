import cv2

file_path_face = 'haarcascade_frontalface_default.xml'
file_path_eyes = 'haarcascade_eye.xml'
faceCascade = cv2.CascadeClassifier(file_path_face)
eyeCascade = cv2.CascadeClassifier(file_path_eyes)

cap = cv2.VideoCapture(0)

while(True):
    ret, frame = cap.read()

    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)

    faces = faceCascade.detectMultiScale(frame, 1.3, 5)
    for x,y,w,h in faces:
        cv2.rectangle(frame, (x,y), (x+w, y+h), (0,255,0), 6)
        eyes = eyeCascade.detectMultiScale(gray)
        for (ex, ey,ew,eh) in eyes:
            image = cv2.rectangle(frame, (ex, ey), (ex + ew, ey + eh), (255, 0, 0), 6)

    cv2.imshow('frame', frame)
    # cv2.imshow('gray', gray)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

