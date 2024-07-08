from ultralytics import YOLO
import supervision as sv
import pickle
import os
import sys
import cv2
import numpy as np
sys.path.append('../')
from utils import get_bbox_center, get_bbox_width

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
            detection_with_tracks = self.tracker.update_with_detections(detection_supervision)

            # Each object will have a dictionary containing the id and the bounding box
            tracks["players"].append({})
            tracks["referees"].append({})
            tracks["ball"].append({})

            for frame_detection in detection_with_tracks:
                bbox = frame_detection[0].tolist()
                class_id = frame_detection[3]
                track_id = frame_detection[4]

                if class_id == cls_names_inv["player"]:
                    tracks["players"][frame_num][track_id] = {"bbox":bbox}
                
                if class_id == cls_names_inv["referee"]:
                    tracks["referees"][frame_num][track_id] = {"bbox":bbox}

            # Only 1 ball so make the track to act on one ball object
            for frame_detection in detection_supervision:
                bbox = frame_detection[0].tolist()
                class_id = frame_detection[3]

                if class_id == cls_names_inv["ball"]:
                    tracks["ball"][frame_num][1] = {"bbox":bbox}
        
        # if no stub path then save as pickle
        if stub_path is not None:
            with open(stub_path, 'wb') as f:
                pickle.dump(tracks, f)

        return tracks
    
    # Draws the ellipse underneath a player for their bounding box
    def draw_ellipse(self, frame, bbox, colour, track_id=None):
        y2 = int(bbox[3])
        
        x_center, _ = get_bbox_center(bbox)
        width = get_bbox_width(bbox)

        # Draw ellipse
        cv2.ellipse(
            frame,
            center=(x_center, y2),
            axes=(int(width), int(0.35*width)),
            angle=0.0,
            startAngle=-45,
            endAngle=235,
            color=colour,
            thickness=2,
            lineType=cv2.LINE_4
        )

        # Draw id rectangle
        rectangle_width = 40
        rectangle_height = 20
        x1_rect = x_center - (rectangle_width / 2)
        x2_rect = x_center + (rectangle_width / 2)
        y1_rect = (y2 - (rectangle_height / 2)) + 15 # add padding
        y2_rect = (y2 + (rectangle_height / 2)) + 15 # add padding

        if track_id is not None:
            cv2.rectangle(
                frame,
                (int(x1_rect), int(y1_rect)),
                (int(x2_rect), int(y2_rect)),
                color=colour,
                thickness=cv2.FILLED
                )
            
            # Neaten up the visuals by centering text
            x1_text = x1_rect + 12
            if track_id > 99:
                x1_text -= 10 # 3 digits so shift it left to center it
            
            cv2.putText(
                frame,
                f"{track_id}",
                (int(x1_text), int(y1_rect + 15)),
                cv2.FONT_HERSHEY_COMPLEX,
                0.6,
                (0, 0, 0),
                2
            )

        return frame

    # Draws pointer for the ball
    def draw_pointer(self, frame, bbox, colour):
        y = int(bbox[1])
        x, _ = get_bbox_center(bbox)

        triangle_points = np.array([
            [x, y],
            [x - 10, y - 20],
            [x + 10, y - 20]
            ])

        cv2.drawContours(frame, [triangle_points], 0, colour, thickness=cv2.FILLED)
        cv2.drawContours(frame, [triangle_points], 0, 0, thickness=2)

        return frame


    def draw_annotations(self, video_frames, tracks):
        output_video_frames = []
        for frame_num, frame in enumerate(video_frames):
            frame = frame.copy()
            
            player_dict = tracks["players"][frame_num]
            referee_dict = tracks["referees"][frame_num]
            ball_dict = tracks["ball"][frame_num]

            # Draw player ellipses
            for track_id, player in player_dict.items():
                frame = self.draw_ellipse(frame, player["bbox"], (0, 0, 255), track_id) # red

            # Draw referee ellipses
            for _, ref in referee_dict.items():
                frame = self.draw_ellipse(frame, ref["bbox"], (0, 255, 255)) # yellow
            
            # Draw ball pointer triangle
            for _, ball in ball_dict.items():
                frame = self.draw_pointer(frame, ball["bbox"], (0, 255, 0)) # green

            output_video_frames.append(frame)
        
        return output_video_frames
        