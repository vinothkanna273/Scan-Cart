from ultralytics import YOLO
import cv2
import cvzone
import math
from sort import *

# Set up the video capture
cap = cv2.VideoCapture(0)
cap.set(3, 1440)
cap.set(4, 826)
# Load the YOLO model
model = YOLO('lays.pt')
# Define the class names
classNames = ['cream_and_onion', 'classicsalted', 'cream_and_onion', 'hot_and_sweet', 'indian_magic_masala',
              'max_chili', 'maxx_sizzling_barbeque', 'tangy_tomato']
# Create the object tracker
tracker = Sort(max_age=20, min_hits=3, iou_threshold=0.3)
# Initialize the lists to store detected objects
cream_and_onion = []
max_chili = []
indian_magic_masala = []
while True:
    success, img = cap.read()
    # imgGraphics = cv2.imread("graphics.png", cv2.IMREAD_UNCHANGED)
    # img = cvzone.overlayPNG(img, imgGraphics, (0, 0))
    results = model(img, stream=True)
    detections = np.empty((0, 5))
    for r in results:
        boxes = r.boxes
        for box in boxes:
            # Bounding box
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            w, h = x2 - x1, y2 - y1
            cvzone.cornerRect(img, (x1, y1, w, h))
            # Confidence
            conf = math.ceil((box.conf[0] * 100)) / 100
            # Class name
            cls = int(box.cls[0])
            currentClass = classNames[cls]
            if currentClass == "cream_and_onion" or currentClass == "max_chili" or currentClass == "indian_magic_masala":
                currentArray = np.array([x1, y1, x2, y2, conf])
                detections = np.vstack((detections, currentArray))
    resultsTracker = tracker.update(detections)
    for result in resultsTracker:
        x1, y1, x2, y2, id = result
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        w, h = x2 - x1, y2 - y1
        cvzone.cornerRect(img, (x1, y1, w, h), l=9, rt=2, colorR=(255, 0, 255))
        cvzone.putTextRect(img, f'{int(id)}', (max(0, x1), max(35, y1)), scale=2, thickness=3, offset=10)
        cx, cy = x1 + w // 2, y1 + h // 2
        cv2.circle(img, (cx, cy), 5, (255, 0, 255), cv2.FILLED)
        cvzone.putTextRect(img, f'{currentClass} {conf}', (max(0, x1), max(35, y1)), scale=1, thickness=1)

        if currentClass == "cream_and_onion" and cream_and_onion.count(id) == 0:
            cream_and_onion.append(id)
        elif currentClass == "max_chili" and max_chili.count(id) == 0:
            max_chili.append(id)
        elif currentClass == "indian_magic_masala" and indian_magic_masala.count(id) == 0:
            indian_magic_masala.append(id)
    # quantity of the lays
    qty1 = len(cream_and_onion)
    qty2 = len(max_chili)
    qty3 = len(indian_magic_masala)
    # price of individual lays
    lay1 = len(cream_and_onion) * 20
    lay2 = len(max_chili) * 10
    lay3 = len(indian_magic_masala) * 20
    # total price of the lays
    totalcount = (len(cream_and_onion) * 20) + (len(max_chili) * 10) + \
                 (len(indian_magic_masala) * 20)
    print("cream_and_onoion", len(cream_and_onion))
    print("max_chili", len(max_chili))
    # print(len(cream_and_onion))
    print("indian_magic_masala", len(indian_magic_masala))
    print("total", totalcount)

    # cv2.putText(img, str(totalcount), (225, 100), cv2.FONT_HERSHEY_PLAIN, 5, (58, 58, 255), 8)
    cv2.imshow("Image", img)

    if cv2.waitKey(1) == ord('q'):

