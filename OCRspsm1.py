import cv2 as cv
import easyocr
import numpy as np
import matplotlib.pyplot as plt

licensePlates = {
    1: "HI9876",
}

def showImage(data, convert=False, gray=True):
    plt.figure(figsize=(10, 6))
    if convert:
        data = data[:, :, ::-1]  # BGR -> RGB
    plt.imshow(data, cmap='gray' if gray else None)
    plt.axis('off')
    plt.show()

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
        
    print("Location: ", location)

    if len(location) == 0:
        return frame, frame

    mask = np.zeros(gray.shape, np.uint8)
    new_image = cv.drawContours(mask, [location], 0,255, -1)
    new_image = cv.bitwise_and(frame, frame, mask=mask)

    cv.imshow("Mask", new_image)

    (x,y) = np.where(mask==255)
    (x1, y1) = (np.min(x), np.min(y)) #top left
    (x2, y2) = (np.max(x), np.max(y)) #bottom right
    cropped_image = gray[x1:x2+1, y1:y2+1]

    cv.imshow("Cropped Image", cropped_image)

    # blur = cv.GaussianBlur(gray, (5, 5), 0)
    # adaptive_thresh = cv.adaptiveThreshold(
    #     blur, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, 11, 2)

    results = reader.readtext(cropped_image)

    if len(results) == 0:
        return frame, frame

    text = results[0][-2]
    print("========> Text: ", text, "<========")
    res = cv.putText(frame, text=text, org=(approx[0][0][0], approx[1][0][1]+60), fontFace=cv.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(0,255,0), thickness=2, lineType=cv.LINE_AA)
    res = cv.rectangle(frame, tuple(approx[0][0]), tuple(approx[2][0]), (0,255,0),3)
    res = frame

    if text in licensePlates.values():
        print("Registered License Plate Found :", text)
        # GPIO.output(led, GPIO.HIGH)

    return frame, res


reader = easyocr.Reader(['th', 'en'], gpu=True, model_storage_directory="model/")


cap = cv.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    cv.resize(frame, fx=0.2, fy=0.2, dsize=None, interpolation=cv.INTER_LANCZOS4)

    processed_frame, results = process_frame(frame, reader)


    cv.imshow('License Plate Detection', processed_frame)

    if cv.waitKey(1) & 0xFF == ord('q'):
        break

#hello

cap.release()
cv.destroyAllWindows()
