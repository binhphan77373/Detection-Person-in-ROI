import cv2
import numpy as np
from datetime import datetime
import time
import pandas as pd
from yolo_model import YOLOModel
from concurrent.futures import ThreadPoolExecutor
import threading

class VideoProcessor:
    def __init__(self, model_path):
        self.yolo_model = YOLOModel(model_path)
    
    @staticmethod
    def point_in_roi(point, roi):
        """
        Determines if a given point is inside a region of interest (ROI) or not.

        Parameters:
        - point (tuple): The (x, y) coordinates of the point to be tested.
        - roi (array): An array of points that define the vertices of the ROI polygon. 
                    The ROI is expected to be a sequence of (x, y) tuples.

        Returns:
        - bool: True if the point is inside the ROI, False otherwise.
        """
        pt = (int(point[0]), int(point[1]))
        return cv2.pointPolygonTest(roi, pt, False) >= 0

    @staticmethod
    def box_in_roi(box, roi):
        """
        Checks if the top middle or bottom middle point of a bounding box is inside a region of interest (ROI).

        Parameters:
        - box (tuple): A tuple of four values (x1, y1, x2, y2) representing the top-left (x1, y1) and bottom-right (x2, y2)
                    corners of the bounding box.
        - roi (array): An array of points that define the vertices of the ROI polygon. The ROI is expected to be a sequence
                    of (x, y) tuples, defining a closed polygon.

        Returns:
        - bool: True if either the top middle or bottom middle point of the box is inside the ROI, False otherwise.
        """
        top_mid = ((box[0] + box[2]) / 2, box[1])
        bottom_mid = ((box[0] + box[2]) / 2, box[3])
        return VideoProcessor.point_in_roi(top_mid, roi) or VideoProcessor.point_in_roi(bottom_mid, roi)

    def inference(self, image, roi_points_list):
        """
        Performs object detection on an image, counts how many detected objects are within each specified ROI,
        and annotates the image with the results.

        Parameters:
        - image (array): The image on which object detection and ROI marking are performed. This is typically
                        a numpy array as used in OpenCV.
        - roi_points_list (list): A list of lists, where each sublist contains points (tuples or lists of two integers)
                                that define a polygonal ROI in the image.

        Returns:
        - tuple: A tuple containing the annotated image and a list of integers, where each integer represents
                the number of detected objects within the corresponding ROI.
        """
        boxes = self.yolo_model.predict(image)
        person_cnt = [0 for _ in range(len(roi_points_list))]
        for roi_index, roi_points in enumerate(roi_points_list):
            cv2.polylines(image, [np.array(roi_points)], isClosed=True, color=(255, 0, 0), thickness=3)
            for box in boxes:
                if self.box_in_roi((box[0], box[1], box[2], box[3]), np.array(roi_points)):
                    cv2.rectangle(image, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (0, 255, 0), 2)
                    person_cnt[roi_index] += 1
            cv2.putText(image, f'PERSON COUNT {roi_index + 1}: {person_cnt[roi_index]}', (10, 30 * (roi_index + 1)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        return image, person_cnt

    def process_videos(self, video_paths, output_paths, roi_points_list, image_placeholders, chart_placeholder):
        """
        Processes multiple videos to detect objects (specifically persons) within predefined regions of interest (ROIs),
        annotates these videos with detection results and FPS, and updates UI placeholders with the processed frames.
        Additionally, this function visualizes the count of detected objects over time in a specified chart placeholder.

        Parameters:
        - video_paths (list of str): Paths to the input video files to be processed.
        - output_paths (list of str): Paths where the annotated output videos should be saved.
        - roi_points_list (list of list of tuples): Each element is a list containing two lists of (x, y) tuples,
        which define the polygonal ROIs for the corresponding video in `video_paths`.
        - image_placeholders (list): UI elements (placeholders) where the processed frames of each video will be displayed.
        - chart_placeholder: A UI element (placeholder) where the line chart visualizing the count of detected objects
        within each ROI over time will be displayed.

        For each video, the method:
        - Opens the video and asserts successful opening.
        - Retrieves video properties such as frame width, height, and FPS to configure the output video writer.
        - Initializes a video writer for saving the annotated output video.
        - Processes each frame to detect objects and annotate the frame with the detection results and the current FPS.
        - Updates the respective image placeholder with the processed frame.
        - Writes the processed frame to the output video file.
        - Compiles a list of object counts per ROI over time and visualizes this data in the specified chart placeholder.

        """
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
                frame, person_counts = self.inference(frame, roi_points)
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

    


