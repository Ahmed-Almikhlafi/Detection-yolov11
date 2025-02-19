import cv2
from ultralytics import YOLO

# تحميل نموذج YOLO المدرب مسبقًا
model = YOLO("yolo11n.pt")  # استخدم نموذج YOLO المناسب

# فتح الفيديو
video_path = "small.mp4"  # استبدل بمسار الفيديو
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # تنفيذ الكشف عن الكائنات
    results = model(frame)

    # معالجة كل كشف داخل النتائج
    for result in results:
        boxes = result.boxes  # الحصول على الصناديق المحيطية
        for box in boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])  # إحداثيات الصندوق
            conf = box.conf[0].item()  # نسبة الثقة
            cls = int(box.cls[0].item())  # فئة الكائن
            label = f"{model.names[cls]}: {conf:.2f}"  # اسم الكائن + نسبة الثقة
            color = (0, 255, 0)  # لون المربع الأخضر

            # رسم المربع حول الكائن
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            # وضع النص بجانب المربع
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, color, 2)

    # عرض الإطار مع النتائج
    cv2.imshow("Object Detection", frame)

    # للخروج من الفيديو عند الضغط على 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# إغلاق الفيديو
cap.release()
cv2.destroyAllWindows()
