from ultralytics import YOLO
import cv2
import streamlit as st
import numpy as np
import pandas as pd
from datetime import datetime
import time

# Load the YOLO model
model = YOLO("yolov8n.pt")

def click_event(event, x, y, flags, params):
    if event == cv2.EVENT_LBUTTONDOWN and len(params['selected_points']) < 4:
        params['selected_points'].append([x, y])
        cv2.circle(params['frame'], (x, y), 5, (0, 0, 255), -1)
        if len(params['selected_points']) == 4:
            cv2.polylines(params['frame'], [np.array(params['selected_points'])], isClosed=True, color=(0, 255, 0), thickness=3)
        cv2.imshow(params['window_name'], params['frame'])

def select_roi_points(frame, window_name):
    selected_points = []
    cv2.imshow(window_name, frame)
    cv2.setMouseCallback(window_name, click_event, {'selected_points': selected_points, 'frame': frame, 'window_name': window_name})
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return np.array(selected_points)

def point_in_roi(point, roi):
    pt = (int(point[0]), int(point[1]))
    return cv2.pointPolygonTest(roi, pt, False) >= 0

def box_in_roi(box, roi):
    top_mid = ((box[0] + box[2]) / 2, box[1]) 
    bottom_mid = ((box[0] + box[2]) / 2, box[3])
    return point_in_roi(top_mid, roi) or point_in_roi(bottom_mid, roi)

def inference(image, roi_points_list):
    results = model.predict(source=image, classes=[0])
    boxes = results[0].boxes.xyxy.cpu()
    
    person_cnt = [0 for _ in range(len(roi_points_list))]
    for roi_index, roi_points in enumerate(roi_points_list):
        cv2.polylines(image, [np.array(roi_points)], isClosed=True, color=(255, 0, 0), thickness=3)
        for box in boxes:
            if box_in_roi((box[0], box[1], box[2], box[3]), np.array(roi_points)):
                cv2.rectangle(image, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (0, 255, 0), 2)
                person_cnt[roi_index] += 1
        cv2.putText(image, f'PERSON COUNT {roi_index + 1}: {person_cnt[roi_index]}', (10, 30 * (roi_index + 1)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    return image, person_cnt

def process_videos(video_paths, output_paths, roi_points_list, image_placeholders, chart_placeholder):
    assert len(video_paths) == len(output_paths) == len(image_placeholders), "Mismatch in the number of video paths, output paths, and image placeholders."
    assert all(len(roi_points) == 2 for roi_points in roi_points_list), "Each video must have exactly two lists of ROI points."
    
    video_captures = [cv2.VideoCapture(path) for path in video_paths]
    assert all(cap.isOpened() for cap in video_captures), "Failed to open one or more video files."
    
    video_props = [(int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                    int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
                    int(cap.get(cv2.CAP_PROP_FPS)))
                   for cap in video_captures]
    
    video_writers = [cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'MP4V'), fps, (w, h))
                     for (w, h, fps), output_path in zip(video_props, output_paths)]
    
    all_person_count_lists = [[] for _ in range(sum(len(roi_points) for roi_points in roi_points_list))]
    time_list = []
    start_times = [time.time() for _ in video_captures]

    while all(cap.isOpened() for cap in video_captures):
        frames_processed = 0
        for i, (cap, roi_points, image_placeholder) in enumerate(zip(video_captures, roi_points_list, image_placeholders)):
            ret, frame = cap.read()
            if not ret:
                continue
            
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame, person_counts = inference(frame, roi_points)
            # FPS calculation for the current video
            end_time = time.time()
            fps = int(1 / (end_time - start_times[i]))
            start_times[i] = end_time  
            cv2.putText(frame, f'FPS: {fps:.2f}', (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
            image_placeholder.image(frame)
            
            for j, count in enumerate(person_counts):
                all_person_count_lists[i*len(roi_points) + j].append(count)
            
            video_writers[i].write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
            frames_processed += 1
        
        if frames_processed == len(video_captures):
            time_list.append(datetime.now().strftime("%H:%M:%S"))
            
            for person_count_list in all_person_count_lists:
                while len(person_count_list) < len(time_list):
                    person_count_list.append(0)  
            
            chart_data = pd.DataFrame({"Time": time_list})
            for i, person_count_list in enumerate(all_person_count_lists):
                chart_data[f"ROI {i+1} Person Count"] = person_count_list
            
            chart_placeholder.line_chart(chart_data.set_index("Time"))

    for cap in video_captures:
        cap.release()
    for writer in video_writers:
        writer.release()

video_paths = ["C:/Users/ADMIN/Downloads/YOLO-Realtime-Human-Detection/Detection_Person/test.mp4", 
               "C:/Users/ADMIN/Downloads/YOLO-Realtime-Human-Detection/Detection_Person/test2.mp4"]
output_paths = ["C:/Users/ADMIN/Downloads/YOLO-Realtime-Human-Detection/Detection_Person/output1.mp4", 
                "C:/Users/ADMIN/Downloads/YOLO-Realtime-Human-Detection/Detection_Person/output2.mp4"]

# Select ROI for video 1
cap1 = cv2.VideoCapture(video_paths[0])
assert cap1.isOpened(), f"Error reading video file {video_paths[0]}"
success1, first_frame1 = cap1.read()
assert success1, "Failed to read the first frame"
first_frame1 = cv2.cvtColor(first_frame1, cv2.COLOR_BGR2RGB)
roi_points1 = select_roi_points(first_frame1, "Select ROI Points 1")
roi_points2 = select_roi_points(first_frame1, "Select ROI Points 2")

# Select ROI for video 2
cap2 = cv2.VideoCapture(video_paths[1])
assert cap2.isOpened(), f"Error reading video file {video_paths[1]}"
success2, first_frame2 = cap2.read()
assert success2, "Failed to read the first frame"
first_frame2 = cv2.cvtColor(first_frame2, cv2.COLOR_BGR2RGB)
roi_points3 = select_roi_points(first_frame2, "Select ROI Points 2")
roi_points4 = select_roi_points(first_frame2, "Select ROI Points 2")

col1, col2 = st.columns([0.5, 0.5])
with col1:
    image_placeholder1 = st.empty()
    image_placeholder2 = st.empty()
with col2:
    chart_placeholder = st.empty()
process_videos(
    video_paths, 
    output_paths, 
    [(roi_points1, roi_points2), (roi_points3, roi_points4)], 
    [image_placeholder1, image_placeholder2], 
    chart_placeholder
)
