import cv2

def save_cropped_player_img(video_frames, tracks):
    # Save cropped image of a player
    for _, player in tracks['players'][1].items():
        bbox = player["bbox"]
        frame = video_frames[0]

        # Crop bbox from frame
        cropped_image = frame[int(bbox[1]):int(bbox[3]), int(bbox[0]):int(bbox[2])]

        # Save the cropped image
        cv2.imwrite(f"output_videos/cropped_img.jpg", cropped_image)

        print("reached")
