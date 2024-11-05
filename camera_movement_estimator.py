import pickle
import cv2
import numpy as np
import os
import sys
sys.path.append('../')
from utils import measure_distance, measure_xy_distance

class CameraMovementEstimator():
    def __init__(self,frame):
        self.minimum_distance = 5
        
        self.lk_params = dict(
            winSize = (15,15),
            maxLevel = 2,
            criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,10,0.03)
        )
        
        first_frame_grayscale = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
        mask_features = np.zeros_like(first_frame_grayscale)
        mask_features[:,0:20] = 1
        mask_features[:,900:1050] = 1
        
        self.features = dict(
            maxCorners = 100,
            qualityLevel = 0.3,
            minDistance =3,
            blockSize = 7,
            mask = mask_features
        )
        
    def get_camera_movement(self,frames,read_from_stub=False,stub_path=None):
        # Read the stub
        if read_from_stub and stub_path is not None and os.path.exists(stub_path):
            with open(stub_path,'rb') as f:
                return pickle.load(f) 
                
        # Creates a list of [0, 0] repeated len(frames) times.
        camera_movement = [[0,0]]*len(frames) 
        
        # Converts the first frame of the video from color (BGR) to grayscale using OpenCV
        old_gray = cv2.cvtColor(frames[0],cv2.COLOR_BGR2GRAY)
        # Find good features to track in the grayscale image
        old_features = cv2.goodFeaturesToTrack(old_gray,**self.features)
        
        # Loop that iterates over the frames from the second frame to the last
        for frame_num in range(1,len(frames)):
            # Converts the current frame to grayscale
            frame_gray = cv2.cvtColor(frames[frame_num],cv2.COLOR_BGR2GRAY)
            # Track the features detected in old_gray as they move to frame_gray.
            new_features, _,_ = cv2.calcOpticalFlowPyrLK(old_gray,frame_gray,old_features,None,**self.lk_params)
            
            max_distance = 0 # Tracks the maximum distance moved by any feature point between frames
            camera_movement_x, camera_movement_y = 0,0 # Initialized to zero and will store the overall camera movement
            
            # Iterates over pairs of new and old feature points using zip and enumerate.
            # 'new' and 'old' are points in the current and previous frames.
            for i, (new,old) in enumerate(zip(new_features, old_features)):
                
                # Converts feature points from 2D arrays to 1D arrays 
                new_features_point = new.ravel()
                old_features_point = old.ravel()
                
                # Calculates the distance between the new and old feature points
                distance = measure_distance(new_features_point,old_features_point)
                
                # If the distance between a feature point pair is greater
                if distance > max_distance:
                    # Update max_distance
                    max_distance = distance 
                    # Measure camera distance
                    camera_movement_x, camera_movement_y = measure_xy_distance(old_features_point,new_features_point)
            
            # If the max_distance is greater than threshold self.minimum_distance
            if max_distance > self.minimum_distance:
                # update the camera_movement for the current frame  
                camera_movement[frame_num] = [camera_movement_x,camera_movement_y]
                # Detect new features in the current frame, 'frame_gray', for use in the next iteration
                old_features = cv2.goodFeaturesToTrack(frame_gray,**self.features)
                
            # Copy frame_gray to old_gray for next iteration     
            old_gray = frame_gray.copy()
        
        if stub_path is not None:
            with open(stub_path,'wb') as f:
                pickle.dump(camera_movement,f) 
        
        return camera_movement # A list
    
    def adjust_positions_to_tracks(self, tracks,camera_movement_per_frame):
        # Iterate Over Objects: This line starts a loop over the tracks dictionary. 
        # object: Represents the type of object being tracked.
        # object_tracks: Contains the tracking data for the current object type, organized by frame number.
        for object, object_tracks in tracks.items():
            # This line starts a loop over each frame’s track data for the current object type.
            # frame_num: The index or number of the current frame.
            # track: Contains the tracking information for the current frame, organized by track ID.
            for frame_num, track in enumerate(object_tracks):
                # This line starts another loop where 
                # track_id is the ID for the track within the current frame
                # track_info contains the details (e.g., bounding box) of that track.
                for track_id, track_info in track.items():
                    # Extracts the position of the tracked object
                    position = track_info['position']
                    # Retrieves the camera movement for the current frame
                    camera_movement = camera_movement_per_frame[frame_num]
                    # Computes the adjusted position by subtracting the camera movement from the original position
                    position_adjusted = (position[0] - camera_movement[0], position[1] - camera_movement[1])
                    # Updates the tracks dictionary by adding a new key 'position_adjusted' to store the adjusted position 
                    # for the current object, frame, and track ID.
                    tracks[object][frame_num][track_id]['position_adjusted'] = position_adjusted
    
    def draw_camera_movement(self,frames,camera_movement_per_frame):
        output_frames = []
        
        for frame_num, frame in enumerate(frames):
            frame = frame.copy()
            overlay = frame.copy()
            cv2.rectangle(overlay,(0,0),(500,100),(255,255,255),-1)
            alpha = 0.6
            cv2.addWeighted(overlay,alpha,frame,1-alpha,0,frame)
            
            x_movement, y_movement = camera_movement_per_frame[frame_num]
            frame = cv2.putText(frame,f"Camera Movement X: {x_movement:.2f}",(10,30),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,0),3)
            frame = cv2.putText(frame,f"Camera Movement Y: {y_movement:.2f}",(10,60),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,0),3)
            
            output_frames.append(frame)
            
        return output_frames