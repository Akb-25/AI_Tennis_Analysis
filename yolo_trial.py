from ultralytics import YOLO
model=YOLO("models\\best.pt")
save_dir="C:\\Users\\win10\\Desktop\\Project\\AI tennis analysis"
result=model.predict('input_data\\input_video.mp4',save=True,conf=0.1)
print(result)

for box in result[0].boxes:
    print(box)