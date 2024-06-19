import cv2
import face_recognition

biden_image = face_recognition.load_image_file("known_faces/biden.jpg")
obama_image = face_recognition.load_image_file("known_faces/obama.jpg")
mahesh_image = face_recognition.load_image_file("known_faces/mahesh.jpg")

biden_face_encoding = face_recognition.face_encodings(biden_image)[0]
obama_face_encoding = face_recognition.face_encodings(obama_image)[0]
mahesh_face_encoding = face_recognition.face_encodings(mahesh_image)[0]

known_face_encodings = [
    biden_face_encoding,
    obama_face_encoding,
    mahesh_face_encoding
]
known_face_names = [
    "biden",
    "obama",
    "mahesh"
]

video_capture = cv2.VideoCapture(0)

while True:
    ret, frame = video_capture.read()

    face_locations = face_recognition.face_locations(frame)
    face_encodings = face_recognition.face_encodings(frame, face_locations)

    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding)

        name = "Unknown"
        if True in matches:
            first_match_index = matches.index(True)
            name = known_face_names[first_match_index]

        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)

        cv2.rectangle(frame, (left, bottom - 25), (right, bottom), (0, 255, 0), cv2.FILLED)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(frame, name, (left + 6, bottom - 6), font, 0.5, (0, 0, 0), 1)

    cv2.imshow('Video', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video_capture.release()
cv2.destroyAllWindows()
