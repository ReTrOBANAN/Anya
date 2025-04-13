from ultralytics import YOLO

# Загрузка обученной модели
model = YOLO("best1.pt")


def video():
    results = model("video.mp4", save=True, iou=0.4, conf=0.4, imgsz=(1920,1080), max_det=1, augment=True)

def photo():
    results = model("IMG_2576.PNG", save=True, iou=0.4, conf=0.4, imgsz=(1920,1080), max_det=1, augment=True)

    results[0].show()

