from ultralytics import YOLO
import supervision as sv
import pickle
import os
import numpy as np
import pandas as pd
import cv2
import sys
sys.path.append('../')
from utils import get_center_of_bbox, get_bbox_width, get_foot_position

class Tracker:
    def __init__(self, model_path):
        self.model = YOLO(model_path) # A variable that holds YOLO model
        self.tracker = sv.ByteTrack() # A variable that holds ByteTrack.
        
    def get_object_tracks(self,frames, read_from_stub=False, stub_path=None):
        
        if read_from_stub and stub_path is not None and os.path.exists(stub_path):
            # Checks if reading from a stub file is requested.
            # Verifies that a valid stub path is provided and the file exists.
            with open(stub_path,'rb') as f: # Opens the stub file in binary read mode
                tracks = pickle.load(f) # Deserializes the contents of the file (Converts data back to original format)
            return tracks
        
        # Detection and Initialisation
        detections = self.detect_frames(frames)
        
        # A dictionary, 'tracks', is initialised with keys each associated with an empty list.
        tracks = {
            "players":[],
            "referees":[],
            "ball":[]
        }
        
        ## Iterates over detections
        # frame_num - index for current iteration number 
        # detection - current detection object 
        for frame_num, detection in enumerate(detections):
            # store names from detection
            cls_names = detection.names 
            # Swap keys and values 
            cls_names_inv = {v:k for k,v in cls_names.items()} 
            
            # 'detection' converted to a supervision format
            detection_supervision = sv.Detections.from_ultralytics(detection)
            
            ## Convert Goalkeeper to Player 
            # iterate over the class_id list in detection_supervision
            # object_ind is the index of the detection in the list
            # class_id is the ID corresponding to the detected class.
            for object_ind, class_id in enumerate(detection_supervision.class_id):
                # if the cls_name corresponding to class_id is "goalkeeper"
                if cls_names[class_id] == "goalkeeper":
                    # Change its class ID to "player"
                    detection_supervision.class_id[object_ind] = cls_names_inv["player"]
            
            ## Track Objects
            # Updates the tracker with the current detections
            detection_with_tracks = self.tracker.update_with_detections(detection_supervision)
            
            # Initialises empty dictionaries for each type of object for current frame
            tracks["players"].append({})
            tracks["referees"].append({})
            tracks["ball"].append({})
            
            # Iterates over the frames in tracked detections. 
            # Each detection includes:
            # a bounding box (bbox), class ID (cls_id), and tracking ID (track_id).
            for frame_detection in detection_with_tracks:
                bbox = frame_detection[0].tolist() # Bounding box converted to a list
                cls_id = frame_detection[3] # Holds the class id, type of object
                track_id = frame_detection[4] # Holds tracking id 
                
                # If the class ID corresponds to a player
                if cls_id == cls_names_inv['player']: 
                    # The key is the track_id, and the value is a dictionary containing the bounding box
                    tracks["players"][frame_num][track_id] = {"bbox":bbox}
                if cls_id == cls_names_inv['referee']:
                    tracks["referees"][frame_num][track_id] = {"bbox":bbox}
                    
            for frame_detection in detection_supervision:
                bbox = frame_detection[0].tolist()
                cls_id = frame_detection[3]
                
                if cls_id == cls_names_inv['ball']:
                    # A 1, and not track_id as there is only 1 ball. No need for track_id
                    tracks["ball"][frame_num][1] = {"bbox":bbox} 
        
        # Saving the computed tracking data to a file and then returning this data    
        if stub_path is not None:
            with open(stub_path,'wb') as f: # Opens file in binary write mode
                pickle.dump(tracks,f) # Serialise the tracks dictionary and write it to the file opened as f.
                # Serialisation converts the tracks dictionary into a byte stream that can be written to a file
                # This allows for the dictionary to be saved and later restored (deserialised) from the file.
        return tracks
    
     # tracks: A dictionary containing tracking information for objects. 
     # The structure is {object_type: {frame_number: {track_id: {'position': (x, y)}}}} 
        
    def add_position_to_tracks(self,tracks):
        # This line begins a loop over the tracks dictionary
        # object will be the key of each entry (e.g., 'ball', 'player'), object_tracks is value corresponding key
        for object, object_tracks in tracks.items():
            # This line begins a second loop that iterates over object_tracks, 
            # where frame_num is the index of the current frame
            # and track is the data for that frame.
            for frame_num, track in enumerate(object_tracks):
                # This line starts another loop where 
                # track_id is the ID for the track within the current frame
                # and track_info contains the details (e.g., bounding box) of that track.
                for track_id, track_info in track.items():
                    # Extract the bounding box (bbox) from the track_info. 
                    bbox = track_info['bbox']
                    if object == 'ball':
                        position = get_center_of_bbox(bbox)
                    else:
                        position = get_foot_position(bbox)
                    # Updates the tracks dictionary by adding the calculated position to the track_info
                    tracks[object][frame_num][track_id]['position'] = position
    
    def interpolate_ball_positions(self, ball_positions):
        # Retrieve the value with key 1. If not found, return empty dictionary {}
        # From dictionary, attempts to get the value associated with key 'bbox'. If key not present, it returns empty list [].
        # List of bounding boxes for each frame in ball_positions.
        ball_positions = [x.get(1,{}).get('bbox',[]) for x in ball_positions] 
        # Converts the ball_positions list into a Pandas DataFrame
        df_ball_positions = pd.DataFrame(ball_positions, columns=['x1','y1','x2','y2'])
        
        # Interpolate missing values
        df_ball_positions = df_ball_positions.interpolate() # Linear interpolation on DataFrame to estimate missing values
        df_ball_positions = df_ball_positions.bfill() # Fill any remaining missing values by backfilling
        
        # Transforms the DataFrame back into the original format expected for 'ball_positions'
        # df_ball_positions.to_numpy().tolist(): Converts the DataFrame to a NumPy array and then to a list of lists
        # For each bounding box coordinate list x, it creates a dictionary where {k: {k:v}}
        ball_positions = [{1: {"bbox":x}} for x in df_ball_positions.to_numpy().tolist()]
        #[
            # {1: {"bbox": [100, 150, 200, 250]}},
            # {1: {"bbox": [110, 160, 210, 260]}}
        # ]
        return ball_positions
        
    def detect_frames(self, frames):
        batch_size = 20 # Each frame is processed in chunks of 20 
        detections = [] # Initialise empty array
        for i in range(0, len(frames),batch_size): # Iterates over frames in steps of batch sizes 
            # Extract batch of frames and predicts the method of 'self.model' 
            detections_batch = self.model.predict(frames[i:i+batch_size],conf=0.1) 
            # Results stored in variable detections with concatenation 
            detections += detections_batch
        return detections
            
    def draw_ellipse(self,frame,bbox,color,track_id=None):
        y2 = int(bbox[3])
        
        x_center, _ = get_center_of_bbox(bbox)
        width = get_bbox_width(bbox)
        
        cv2.ellipse(
            frame,
            center=(x_center,y2),
            axes=(int(width),int(0.35*width)),
            angle = 0.0,
            startAngle = -45,
            endAngle = 235,
            color = color,
            thickness= 2,
            lineType=cv2.LINE_4
        )
        
        rectangle_width = 40
        rectangle_height = 20
        x1_rect = x_center - rectangle_width//2
        x2_rect = x_center + rectangle_width//2
        y1_rect = (y2 - rectangle_height//2) + 15
        y2_rect = (y2 + rectangle_height//2) + 15
        
        if track_id is not None:
            cv2.rectangle(frame,
                         (int(x1_rect),int(y1_rect)),
                         (int(x2_rect),int(y2_rect)),
                         color,
                         cv2.FILLED)
            
            x1_text = x1_rect+12
            if track_id > 99:
                x1_text -=10
                
            cv2.putText(
                frame,
                f"{track_id}",
                (int(x1_text),int(y1_rect+15)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0,0,0),
                2
            )
        
        return frame
    
    def draw_triangle(self,frame,bbox,color):
        y = int(bbox[1])
        x,_ = get_center_of_bbox(bbox)
        
        triangle_points = np.array([
            [x,y],
            [x-10,y-20],
            [x+10,y-20]
        ])
        cv2.drawContours(frame,[triangle_points],0,color,cv2.FILLED)
        cv2.drawContours(frame,[triangle_points],0,(0,0,0),2)
        
        return frame
            
    def draw_team_ball_control(self,frame,frame_num,team_ball_control):
        # Draw semi-transparent rectangle
        overlay = frame.copy()
        cv2.rectangle(overlay,(1350,850),(1900,970),(255,255,255),cv2.FILLED)
        alpha = 0.4 # 40%
        cv2.addWeighted(overlay,alpha,frame, 1-alpha,0,frame)
        
        team_ball_control_till_frame = team_ball_control[:frame_num+1]
        # Get the number of times each team had ball 
        team_1_num_frames = team_ball_control_till_frame[team_ball_control_till_frame==1].shape[0]
        team_2_num_frames = team_ball_control_till_frame[team_ball_control_till_frame==2].shape[0]
        
        team_1 = team_1_num_frames/(team_1_num_frames+team_2_num_frames)
        team_2 = team_2_num_frames/(team_1_num_frames+team_2_num_frames)
        
        cv2.putText(frame,f"Team 1 Ball Control: {team_1*100:.2f}%",(1400,900),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,0),3)
        cv2.putText(frame,f"Team 2 Ball Control: {team_2*100:.2f}%",(1400,950),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,0),3)
        
        return frame
        
    def draw_annotations(self,video_frames, tracks,team_ball_control):
        output_video_frames = []            
        for frame_num, frame in enumerate(video_frames):
            frame = frame.copy()
            
            player_dict = tracks["players"][frame_num]
            ball_dict = tracks["ball"][frame_num]
            referee_dict = tracks["referees"][frame_num]
            
            # Draw players
            for track_id, player in player_dict.items():
                color = player.get("team_color",(0,0,255))
                frame = self.draw_ellipse(frame, player["bbox"],color, track_id)
                
                if player.get('has_ball', False):
                    frame = self.draw_triangle(frame,player["bbox"],(0,0,255))
            
            # Draw Referee
            for _, referee in referee_dict.items():
                frame = self.draw_ellipse(frame, referee["bbox"],(0,255,255))
                
            # Draw ball
            for track_id, ball in ball_dict.items():
                frame = self.draw_triangle(frame, ball["bbox"], (0,255,0))
            
            # Draw Team Ball Control
            frame = self.draw_team_ball_control(frame,frame_num,team_ball_control)
            
            output_video_frames.append(frame)
            
        return output_video_frames
            