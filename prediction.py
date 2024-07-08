from ultralytics import YOLO 

model = YOLO('models/last.pt')

results = model.predict('input_videos/test.mp4', save=True)
print(results[0])
print('###########################################')
for box in results[0].boxes:
    print(box)