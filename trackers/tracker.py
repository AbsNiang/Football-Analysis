from ultralytics import YOLO
import supervision as sv
import pickle
import os

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
    
    def get_object_tracks(self, frames, read_from_stub=False, stub_path=None):

        # If stub exists then read rather than re-doing tracking and detection
        if read_from_stub and stub_path is not None and os.path.exists(stub_path):
            with open(stub_path, 'rb') as f:
                tracks = pickle.load(f)
            return tracks

        # If tracking hasn't been done yet
        detections = self.detect_frames(frames)

        tracks = {
            "players":[],
            "referees":[],
            "ball":[]
        }

        for frame_num, detection in enumerate(detections):
            cls_names = detection.names # {0: person, 1: goal etc..}
            cls_names_inv = {v:k for k,v in cls_names.items()} # flips class names to be {person:0, goal:1}

            # Convert to sv detection format
            detection_supervision = sv.Detections.from_ultralytics(detection)

            # Convert goalkeeper into player object
            for object_index, class_id in enumerate(detection_supervision.class_id):
                if cls_names[class_id] == "goalkeeper":
                    detection_supervision.class_id[object_index] = cls_names_inv["player"]

            # Track objects
            detection_with_tracks = self.tracker.update_with_detections(detection_supervision, frame_num)

            # Each object will have a dictionary containing the id and the bounding box
            tracks["players"].append({})
            tracks["referees"].append({})
            tracks["ball"].append({})

            for frame_detection in detection_with_tracks:
                bbox = frame_detection[0].toList()
                class_id = frame_detection[3]
                track_id = frame_detection[4]

                if class_id == cls_names_inv["player"]:
                    tracks["players"][frame_num][track_id] = {"bbox":bbox}
                
                if class_id == cls_names_inv["referee"]:
                    tracks["referee"][frame_num][track_id] = {"bbox":bbox}

            # Only 1 ball so make the track to act on one ball object
            for frame_detection in detection_supervision:
                bbox = frame_detection[0].toList()
                class_id = frame_detection[3]

                if class_id == cls_names_inv["ball"]:
                    tracks["ball"][frame_num][1] = {"bbox":bbox}
        
        # if no stub path then save as pickle
        if stub_path is not None:
            with open(stub_path, 'wb') as f:
                pickle.dump(tracks, f)

        return tracks