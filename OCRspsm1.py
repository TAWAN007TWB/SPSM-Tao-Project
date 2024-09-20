import cv2 as cv
import easyocr
import numpy as np
import RPi.GPIO as GPIO
import time

Servo_signal = 24
LED_pin = 26

GPIO.setmode(GPIO.BCM)
GPIO.setup(LED_pin,GPIO.OUT)

GPIO.setup(Servo_signal, GPIO.OUT)

servo = GPIO.PWM(Servo_signal, 50)
servo.start(2.5)

servo.ChangeDutyCycle(2.5)


licensePlates = {
    1: "HI9876",
    2: "8TRS777"
}

def map_range(x, in_min, in_max, out_min, out_max):
    return int((x-in_min)*(out_max-out_min)/(in_max-in_min)+out_min)

def process_frame(frame, reader):
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

    bfilter = cv.bilateralFilter(gray, 11, 17, 17)

    edged = cv.Canny(bfilter, 30, 200)

    contours, _ = cv.findContours(edged, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv.contourArea, reverse=True)[:10]

    location = []
    for contour in contours:
        approx = cv.approxPolyDP(contour, 10, True)
        if len(approx) == 4:
            location = approx
            break

    # print("Location: ", location)

    if len(location) == 0:
        return frame, frame

    mask = np.zeros(gray.shape, np.uint8)
    new_image = cv.drawContours(mask, [location], 0, 255, -1)
    new_image = cv.bitwise_and(frame, frame, mask=mask)

    cv.imshow("Mask", new_image)

    (x, y) = np.where(mask == 255)
    (x1, y1) = (np.min(x), np.min(y)) #top left
    (x2, y2) = (np.max(x), np.max(y)) #bottom right
    cropped_image = gray[x1:x2+1, y1:y2+1]

    cv.imshow("Cropped Image", cropped_image)

    results = reader.readtext(cropped_image, detail=0)
    print("----",results,"----")

    if len(results) == 0:
        return frame, frame

    text = [i.strip().upper() for i in results]  # Get the detected text, strip extra spaces and convert to uppercase
    print("========> Text: ", text, "<========")

    res = cv.putText(frame, text=",".join(text), org=(location[0][0][0], location[1][0][1] + 60), fontFace=cv.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(0, 255, 0), thickness=2, lineType=cv.LINE_AA)
    res = cv.rectangle(frame, tuple(location[0][0]), tuple(location[2][0]), (0, 255, 0), 3)
    res = frame

    # Check if the detected text is in the license plates list
    mismatch = 0
    for txt in text:
        if txt in licensePlates.values():
            print("Registered License Plate Found:", txt)
            print("Correct")
            servo.ChangeDutyCycle(7.5)
            time.sleep(5)
            servo.ChangeDutyCycle(2.5)
        else:
            mismatch += 1
            if mismatch >= len(text) :
                GPIO.output(LED_pin,1)
                time.sleep(2)
                GPIO.output(LED_pin,0)
            print(mismatch)


    return frame, res

reader = easyocr.Reader(['en'], gpu=True, model_storage_directory="model/")

cap = cv.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret or frame is None:
        print("Failed to grab frame")
        break

    frame = cv.resize(frame, None, fx=0.5, fy=0.5, interpolation=cv.INTER_LANCZOS4)

    processed_frame, results = process_frame(frame, reader)

    cv.imshow('License Plate Detection', processed_frame)

    if cv.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv.destroyAllWindows()