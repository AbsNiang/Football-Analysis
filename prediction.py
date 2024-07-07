from ultralytics import YOLO 
from dotenv import load_dotenv
import os

def configure():
    load_dotenv()

def main():
    configure()
    #os.getenv('api_key')
    model = YOLO('yolov8x')

    results = model.track('input_videos/test.mp4', save=True)
    print(results[0])
    print('###########################################')
    for box in results[0].boxes:
        print(box)
        #5762

main()