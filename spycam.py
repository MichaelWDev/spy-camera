# pip3 install opencv-python

import cv2
import time
import datetime

cap = cv2.VideoCapture(0) # Multiple Cameras: 1, 2, etc..

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
body_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_fullbody_default.xml")

detection              = False
detection_stopped_time = None
timer_started          = False
SECONDS_TO_RECORD_AFTER_DETECTION = 5

frame_size = (int(cap.get(3)), int(cap.get(4)))
fourcc     = cv2.VideoWriter_fourcc(*"mp4v") # * separates into m p v 4, since fourcc takes 4 arguments.

while True:
	_, frame = cap.read() # Frame = video frames

	gray   = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) # Grayscale image.
	faces  = face_cascade.detectMultiScale(gray, 1.3, 5) # 1.3 is scale factor (Lower = more accurate), 5 is number of detected faces.
	bodies = face_cascade.detectMultiScale(gray, 1.3, 5)

	if len(faces) + len(bodies) > 0: # If body/face detected.
		if detection:
			timer_started = False # Don't stop recording.
		else:
			detection    = True
			current_time = datetime.datetime.now().strftime("%d-%m-%Y-%H-%M-%S") # Get current time, format current time Day, Month, Year, Hour, Minute, Second.
			out          = cv2.VideoWriter(f"{current_time}.mp4", fourcc, 20, frame_size) # Video name, four character code, frame rate, frame size.
			print("Started recording.")
	elif detection:
		if timer_started:
			if time.time() - detection_stopped_time >= SECONDS_TO_RECORD_AFTER_DETECTION:
				detection     = False
				timer_started = False
				out.release()
				print("Stopped recording.")
		else:
			timer_started = True
			detection_stopped_time = time.time() # Current time.

	if detection:
		out.write(frame)
	
	# Draws a rectangle around the detected body part.
	for (x, y, width, height) in faces:
		cv2.rectangle(frame, (x, y), (x + width, y + height), (255, 0, 0), 3) # BGR (Blue, Green, Red)

	cv2.imshow("Camera", frame) # Title of window, shows video frame.

	if cv2.waitKey(1) == ord('q'): # Q key stops program.
		break

out.release() # Saves video.
cap.release() # Release the resources, cleans up.
cv2.destroyAllWindows()