from ultralytics import YOLO

#running pretrained models with yolo
# model = YOLO('yolov8x')

#fine tuning with best models
model = YOLO('models/best.pt')

results = model.predict('input_videos/08fd33_4.mp4',save = True)

print(results[0])
print("============================================================")
for box in results[0].boxes:
    print(box)