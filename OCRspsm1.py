import cv2 as cv
import easyocr
import matplotlib.pyplot as plt

def showImage(data, convert=False, gray=True):
    plt.figure(figsize=(10, 6))
    if convert:
        data = data[:, :, ::-1]  # BGR -> RGB
    plt.imshow(data, cmap='gray' if gray else None)
    plt.axis('off')
    plt.show()

def process_frame(frame, reader):

    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

    blur = cv.GaussianBlur(gray, (5, 5), 0)

    adaptive_thresh = cv.adaptiveThreshold(
        blur, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, 11, 2)
    
  
    results = reader.readtext(adaptive_thresh)

   
    for (bbox, text, prob) in results:
        (tl, tr, br, bl) = bbox
        
        cv.rectangle(frame, (int(tl[0]), int(tl[1])), (int(br[0]), int(br[1])), (0, 255, 0), 2)
        
        cv.putText(frame, text, (int(tl[0]), int(tl[1]) - 10), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    return frame, results


reader = easyocr.Reader(['en', 'th'], gpu=True)


cap = cv.VideoCapture(0)  

while True:
    ret, frame = cap.read()  
    if not ret:
        break

    processed_frame, results = process_frame(frame, reader)


    cv.imshow('License Plate Detection', processed_frame)

    if cv.waitKey(1) & 0xFF == ord('q'):
        break


cap.release()
cv.destroyAllWindows()
