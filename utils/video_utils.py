import cv2

def read_video(video_path):
    cap = cv2.VideoCapture(video_path)
    frames = []
    while True:
        ret, frame = cap.read() 
        # ret is boolean whether frame is read successfully and end of video is reached, false
        # frame is actual frame data that was read 
        if not ret: # when video is finished
            break # Exit loop
        frames.append(frame) # frame appended to the 'frames' list in the order they're read 
    return frames

def save_video(output_video_frames,output_video_path):
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(output_video_path, fourcc, 24, (output_video_frames[0].shape[1],output_video_frames[0].shape[0]))
    for frame in output_video_frames:
        out.write(frame)
    out.release()

