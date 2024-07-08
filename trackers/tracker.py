from ultralytics import YOLO
import supervision as sv

# Using a tracker rather than model.track because goal keeper not well detected since small dataset
class Tracker:
    def __init__(self, model_path):
        self.model = YOLO(model_path)
        self.tracker = sv.ByteTrack()

    def detect_frames(self, frames):
        batch_size = 20

        detections = []
        for i in range(0, len(frames), batch_size):
            detections_batch = self.model.predict(frames[i:i+batch_size], conf=0.1)
            detections += detections_batch

        return detections
    
    def get_object_tracks(self, frames):
        detections = self.detect_frames(frames)

        for frame_num, detection in enumerate(detections):
            cls_names = detection.names # {0: person, 1: goal etc..}
            cls_names_inv = {v:k for k,v in cls_names.items()} # flips class names to be {person:0, goal:1}

            # Convert to sv detection format
            detection_supervision = sv.Detections.from_ultralytics(detection)

            
