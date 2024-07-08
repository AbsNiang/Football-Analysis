from utils import read_video, save_video
from trackers import Tracker
import cv2

def main():
    # Read video
    video_frames = read_video('input_videos/test.mp4')

    # Initialise tracker
    tracker = Tracker("models/last.pt")
    
    tracks = tracker.get_object_tracks(video_frames, read_from_stub=True, stub_path="stubs/track_stubs.pkl")

    # Save cropped image of a player
    for _, player in tracks['players'][0].items():
        bbox = player["bbox"]
        frame = video_frames[0]

        # Crop bbox from frame
        cropped_image = frame[int(bbox[1]):int(bbox[3]), int(bbox[0]):int(bbox[2])]

        # Save the cropped image
        cv2.imwrite(f"output_videos/cropped_img.jpg", cropped_image)

    # Draw output & draw object tracks
    output_video_frames = tracker.draw_annotations(video_frames, tracks)

    # Save video
    save_video(output_video_frames, 'output_videos/output_video.avi')

if __name__ == '__main__':
    main()