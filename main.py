from utils import read_video,save_video,measure_distance_bw,convert_pixel_dist_to_meter,draw_player_stats
from trackers import BallTracker, PlayerTracker
from court_detector import CourtDetector
from small_court import SmallCourt
import cv2
import pandas as pd
import constants
from copy import deepcopy
def main():
    ball_tracker=BallTracker("models\\last.pt")
    player_tracker=PlayerTracker('yolov8x')

    input_video="input_data\\input_video.mp4"
    video_frames=read_video(input_video)

    player_detections=player_tracker.detect_frames(video_frames,read_from_stub=True,stub_path="tracker_stubs\\player_detections.pkl")
    ball_detections=ball_tracker.detect_frames(video_frames,read_from_stub=True,stub_path="tracker_stubs\\ball_detections.pkl")
    ball_detections=ball_tracker.interpolate_ball_position(ball_detections)
    
    court_model_path="models\\keypoints_model.pth"
    court_line_detector=CourtDetector(court_model_path)
    court_keypoints=court_line_detector.predict(video_frames[0])

    player_detections=player_tracker.choose_and_filter_players(court_keypoints,player_detections)
    small_court=SmallCourt(video_frames[0])
    
    ball_hit_frames=ball_tracker.get_ball_hit_frames(ball_detections)
    # print(court_keypoints)
    # print(player_detections)
    player_mini_court_detections,ball_mini_court_detections=small_court.convert_bounding_boxes_to_mini_court(player_detections,
                                                                                                             ball_detections,
                                                                                                             court_keypoints)

    player_stats_data=[{
        'frame_num':0,
        'player_1_number_of_hits':0,
        'player_1_total_hit_speed':0,
        'player_1_last_hit_speed':0,
        'player_1_total_player_speed':0,
        'player_1_last_player_speed':0,

        'player_2_number_of_hits':0,
        'player_2_total_hit_speed':0,
        'player_2_last_hit_speed':0,
        'player_2_total_player_speed':0,
        'player_2_last_player_speed':0,
    }]

    for ball_hit_ind in range(len(ball_hit_frames)-1):
        start_frame=ball_hit_frames[ball_hit_ind]
        end_frame=ball_hit_frames[ball_hit_ind+1]
        ball_hit_time_in_seconds=(end_frame-start_frame)/24

        # print(ball_mini_court_detections[start_frame])
        distance_covered_by_ball_pixels=measure_distance_bw(ball_mini_court_detections[start_frame][1],
                                                            ball_mini_court_detections[end_frame][1])
        

        distance_covered_by_ball_meters=convert_pixel_dist_to_meter(distance_covered_by_ball_pixels,
                                                                    constants.DOUBLE_LINE_WIDTH,
                                                                    small_court.get_width_of_mini_court())

        speed_of_ball_hit=distance_covered_by_ball_meters/ball_hit_time_in_seconds*3.6

        player_positions=player_mini_court_detections[start_frame]
        player_hit_ball=min(player_positions.keys(),key=lambda player_id:measure_distance_bw(player_positions[player_id],
                                                                                             ball_mini_court_detections[start_frame][1]))
        opponent_player_id = 1 if player_hit_ball ==2 else 2 
        distance_covered_by_opponent_pixels=measure_distance_bw(player_mini_court_detections[start_frame][opponent_player_id],
                                                                player_mini_court_detections[end_frame][opponent_player_id])
        
        distance_covered_by_opponent_meters=convert_pixel_dist_to_meter(distance_covered_by_opponent_pixels,
                                                                        constants.DOUBLE_LINE_WIDTH,
                                                                        small_court.get_width_of_mini_court() )
        speed_of_opponent=distance_covered_by_opponent_meters/ball_hit_time_in_seconds*3.6

        current_player_stats=deepcopy(player_stats_data[-1])
        current_player_stats['frame_num']=start_frame
        current_player_stats[f'player_{player_hit_ball}_number_of_hits']+=1
        current_player_stats[f'player_{player_hit_ball}_total_hit_speed']+=speed_of_ball_hit
        current_player_stats[f'player_{player_hit_ball}_last_hit_speed']=speed_of_ball_hit
        current_player_stats[f'player_{player_hit_ball}_total_player_speed']+=speed_of_opponent
        current_player_stats[f'player_{player_hit_ball}_last_player_speed']=speed_of_opponent

        player_stats_data.append(current_player_stats)
    
    player_stats_data_df=pd.DataFrame(player_stats_data)
    frames_df=pd.DataFrame({'frame_num':list(range(len(video_frames)))})
    player_stats_data_df=pd.merge(frames_df,player_stats_data_df,on='frame_num',how='left')
    player_stats_data_df=player_stats_data_df.ffill()

    player_stats_data_df['player_1_average_hit_speed']=player_stats_data_df['player_1_total_hit_speed']/player_stats_data_df['player_1_number_of_hits']
    player_stats_data_df['player_2_average_hit_speed']=player_stats_data_df['player_2_total_hit_speed']/player_stats_data_df['player_1_number_of_hits']
    player_stats_data_df['player_1_average_player_speed']=player_stats_data_df['player_1_total_player_speed']/player_stats_data_df['player_1_number_of_hits']
    player_stats_data_df['player_2_average_player_speed']=player_stats_data_df['player_2_total_player_speed']/player_stats_data_df['player_2_number_of_hits']


    output_video_frames=player_tracker.draw_bbox(video_frames,player_detections)
    output_video_frames=ball_tracker.draw_boxes(output_video_frames,ball_detections)
    output_video_frames=court_line_detector.draw_keypoints_on_video(output_video_frames,court_keypoints)

    output_video_frames=small_court.draw_mini_court(output_video_frames)
    output_video_frames=small_court.draw_points_on_mini_court(output_video_frames,player_mini_court_detections)
    output_video_frames=small_court.draw_points_on_mini_court(output_video_frames,ball_mini_court_detections,color=(255,0,0))

    output_video_frames=draw_player_stats(output_video_frames,player_stats_data_df)
    for i,frame in enumerate(output_video_frames):
        cv2.putText(frame,f"Frame: {i}",(10,30),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2)
    
    save_video(output_video_frames,"output_videos\\output_video.avi")
if __name__ == "__main__":
    main()