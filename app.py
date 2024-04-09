import streamlit as st
from roi_selector import ROISelector
from video_processor import VideoProcessor
import cv2

def main():
    st.title("YOLO Real-time Human Detection")

    col1, col2 = st.columns(2)
    with col1:
        image_placeholder1 = st.empty()
        image_placeholder2 = st.empty()
    with col2:
        chart_placeholder = st.empty()

    model_path = "yolov8n.pt"

    video_processor = VideoProcessor(model_path)

    video_paths = ["videos/test.mp4", 
               "videos/test2.mp4"]
    output_paths = ["videos/output1.mp4", 
                "videos/output2.mp4"]

    roi_points_list = []

    # Select ROI for each video
    for idx, video_path in enumerate(video_paths):
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            st.error(f"Failed to open video file: {video_path}")
            return
        success, frame = cap.read()
        if not success:
            st.error("Failed to read the first frame of the video")
            return
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        #st.image(frame, caption=f"First frame of video {idx+1}", channels="RGB")

        roi_points1 = ROISelector.select_roi_points(frame, f"Select ROI Points 1 for Video {idx+1}")
        roi_points2 = ROISelector.select_roi_points(frame, f"Select ROI Points 2 for Video {idx+1}")
        roi_points_list.append((roi_points1, roi_points2))

        cap.release()

    # Check if ROIs have been selected for all videos
    if not all(len(roi_points) == 2 for roi_points in roi_points_list):
        st.error("ROI selection not completed for all videos.")
        return

    # Process the videos
    video_processor.process_videos(
        video_paths, 
        output_paths, 
        roi_points_list, 
        [image_placeholder1, image_placeholder2], 
        chart_placeholder
    )

if __name__ == "__main__":
    main()