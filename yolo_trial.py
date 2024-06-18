from ultralytics import YOLO
model=YOLO("yolov8x")
save_dir="C:\\Users\\win10\\Desktop\\Project\\AI tennis analysis"
result=model.predict('input_data/tennis_img.jpg',save=True,save_dir="output")
print(result)

for box in result[0].boxes:
    print(box)