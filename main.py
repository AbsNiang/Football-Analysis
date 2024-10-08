from utils import read_video, save_video, save_cropped_player_img
from trackers import Tracker
import cv2
import numpy as np
from team_assigner import TeamAssigner
from player_ball_assigner import PlayerBallAssigner

def main():
    # Read video
    video_frames = read_video('input_videos/test.mp4')

    # Initialise tracker
    tracker = Tracker("models/last.pt")
    
    tracks = tracker.get_object_tracks(video_frames, read_from_stub=True, stub_path="stubs/track_stubs.pkl")

    # Interpolate ball positions
    tracks["ball"] = tracker.interpolate_ball_positions(tracks["ball"])

    # Assign player teams
    team_assigner = TeamAssigner()
    team_assigner.assign_team_colour(video_frames[0], tracks["players"][0])

    for frame_num, player_track in enumerate(tracks["players"]):
        for player_id, track in player_track.items():
            team = team_assigner.get_player_team(video_frames[frame_num], track["bbox"], player_id)

            tracks["players"][frame_num][player_id]["team"] = team # introduce new dict value for team
            tracks["players"][frame_num][player_id]["team_colour"] = team_assigner.team_colours[team]

    # Assign ball aquisition
    player_assigner = PlayerBallAssigner()
    team_ball_control = []
    for frame_num, player_track in enumerate(tracks["players"]):
        ball_bbox = tracks["ball"][frame_num][1]["bbox"]
        assigned_player = player_assigner.assign_ball_to_player(player_track, ball_bbox)

        if assigned_player != -1:
            tracks["players"][frame_num][assigned_player]["has_ball"] = True # introduce new dict value for ball posession
            team_ball_control.append(tracks["players"][frame_num][assigned_player]["team"])
        else:
            team_ball_control.append(team_ball_control[-1])
        
    team_ball_control = np.array(team_ball_control)

    # Draw output & draw object tracks
    output_video_frames = tracker.draw_annotations(video_frames, tracks, team_ball_control)

    # Save video
    save_video(output_video_frames, 'output_videos/output_video.avi')

if __name__ == '__main__':
    main()

#2:41:23