import cv2
from ultralytics import YOLO
import cvzone

# YOLOv8 মডেল লোড করা (Pre-trained model)
model = YOLO("yolov8n.pt")  # 'n' মানে nano, যা অনেক ফাস্ট কাজ করে

# ভিডিও ফাইল ওপেন করা
cap = cv2.VideoCapture("video1.mp4") # আপনার ভিডিও ফাইলের নাম দিন

while True:
    success, img = cap.read()
    if not success:
        break

    # অবজেক্ট ডিটেক্ট করা
    results = model(img, stream=True)

    person_count = 0

    for r in results:
        boxes = r.boxes
        for box in boxes:
            # ক্লাস আইডি চেক করা (YOLO-তে মানুষের আইডি হলো 0)
            cls = int(box.cls[0])
            if cls == 0:
                person_count += 1
                
                # বাউন্ডিং বক্স আঁকা
                x1, y1, x2, y2 = box.xyxy[0]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 255), 3)
                cvzone.putTextRect(img, "Person", (max(0, x1), max(35, y1)), scale=1, thickness=1)

    # স্ক্রিনে কতজন আছে তা দেখানো
    cvzone.putTextRect(img, f'Count: {person_count}', (50, 50), scale=2, thickness=2, colorR=(0, 255, 0))

    cv2.imshow("Image", img)
    if cv2.waitKey(1) & 0xFF == ord('q'): # 'q' চাপলে বন্ধ হবে
        break

cap.release()
cv2.destroyAllWindows()