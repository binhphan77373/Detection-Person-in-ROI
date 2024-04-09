import cv2
import numpy as np

class ROISelector:
    @staticmethod
    def click_event(event, x, y, flags, params):
        """
        Handles mouse click events on an OpenCV window to select points and draw a polygon.
        
        Parameters:
        - event: The type of mouse event (e.g., left button click, right button click, mouse move, etc.).
        - x (int): The x-coordinate of the mouse position when the event occurred.
        - y (int): The y-coordinate of the mouse position when the event occurred.
        - flags: Any specific flags passed by OpenCV for this event. Not used in this function.
        - params (dict): A dictionary containing parameters for the callback. Expected keys are:
            - 'selected_points' (list): A list to store the coordinates of the points selected by the user.
            - 'frame' (numpy array): The current image/frame being displayed in the OpenCV window.
            - 'window_name' (str): The name of the OpenCV window where the image is displayed.
        """
        if event == cv2.EVENT_LBUTTONDOWN and len(params['selected_points']) < 4:
            params['selected_points'].append([x, y])
            cv2.circle(params['frame'], (x, y), 5, (0, 0, 255), -1)
            if len(params['selected_points']) == 4:
                cv2.polylines(params['frame'], [np.array(params['selected_points'])], isClosed=True, color=(0, 255, 0), thickness=3)
            cv2.imshow(params['window_name'], params['frame'])

    @staticmethod
    def select_roi_points(frame, window_name):
        """
        Allows a user to select points on an image displayed in an OpenCV window by clicking.

        Parameters:
        - frame (numpy array): The image on which the user will select points. This should be in the format suitable
                            for OpenCV, typically an array read from `cv2.imread` or captured from a video stream.
        - window_name (str): The name of the OpenCV window where the image will be displayed. This name is used to
                            identify the window and must be unique or match the name of an existing window.

        Returns:
        - numpy array: An array containing the (x, y) coordinates of the points selected by the user. The array shape
                    will be (N, 2) where N is the number of points selected (up to 4).
        """
        selected_points = []
        cv2.imshow(window_name, frame)
        cv2.setMouseCallback(window_name, ROISelector.click_event, {'selected_points': selected_points, 'frame': frame, 'window_name': window_name})
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        return np.array(selected_points)
