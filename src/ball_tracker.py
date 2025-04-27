# src/ball_tracker.py

import cv2
import numpy as np
import torch
from typing import List, Dict, Tuple, Any, Optional
import time


class BallTracker:
    """
    Detects and tracks the soccer ball in video frames.
    """

    def __init__(self, model_path: str = None, device: str = 'cuda',
                 conf_threshold: float = 0.5, temporal_smoothing: bool = True):
        """
        Initialize the ball tracker.

        Args:
            model_path: Path to ball detection model (if None, uses traditional CV methods)
            device: Device to run inference on ('cuda' or 'cpu')
            conf_threshold: Confidence threshold for ball detections
            temporal_smoothing: Whether to apply temporal smoothing to ball tracking
        """
        self.device = device
        self.conf_threshold = conf_threshold
        self.temporal_smoothing = temporal_smoothing

        # Initialize ball detection model if specified
        self.use_ml_model = model_path is not None
        if self.use_ml_model:
            try:
                from ultralytics import YOLO
                self.model = YOLO(model_path)
            except ImportError:
                print("Ultralytics YOLO not found, using traditional methods")
                self.use_ml_model = False

        # Ball tracking state
        self.prev_ball_positions = []  # List of previous ball positions
        self.max_history = 10  # Number of previous positions to keep
        self.ball_trajectory = []  # For longer trajectory analysis
        self.last_pass_time = 0  # For pass detection
        self.pass_cooldown = 0.5  # Minimum time between passes (seconds)
        self.velocity_threshold = 15  # Minimum velocity change to register as a pass
        self.prev_velocity = None

        # Kalman filter for ball tracking
        # State: [x, y, dx, dy], Measurement: [x, y]
        self.kalman = cv2.KalmanFilter(4, 2)
        self.kalman.transitionMatrix = np.array([
            [1, 0, 1, 0],
            [0, 1, 0, 1],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ], dtype=np.float32)
        self.kalman.measurementMatrix = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0]
        ], dtype=np.float32)
        self.kalman.processNoiseCov = np.eye(4, dtype=np.float32) * 0.03
        self.kalman.measurementNoiseCov = np.eye(2, dtype=np.float32) * 0.1

        self.kalman_initialized = False

    def detect_circles(self, frame: np.ndarray) -> List[Tuple[int, int, int]]:
        """
        Detect circular objects in the frame using HoughCircles.

        Args:
            frame: Input frame

        Returns:
            List of circles as (x, y, radius)
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (9, 9), 2)

        # Detect circles
        circles = cv2.HoughCircles(
            blurred,
            cv2.HOUGH_GRADIENT,
            dp=1.2,
            minDist=20,
            param1=50,
            param2=30,
            minRadius=5,
            maxRadius=20
        )

        # Return detected circles
        if circles is not None:
            return [(int(x), int(y), int(r)) for x, y, r in circles[0, :]]
        else:
            return []

    def filter_ball_candidates(self, frame: np.ndarray, circles: List[Tuple[int, int, int]]) -> Optional[Tuple[int, int, int]]:
        """
        Filter circle candidates to find the most likely ball.

        Args:
            frame: Input frame
            circles: List of circles as (x, y, radius)

        Returns:
            Best ball candidate as (x, y, radius) or None
        """
        if not circles:
            return None

        ball_scores = []

        for x, y, r in circles:
            # Skip if circle is out of bounds
            h, w = frame.shape[:2]
            if x - r < 0 or y - r < 0 or x + r >= w or y + r >= h:
                continue

            # Extract circle region
            circle_mask = np.zeros(frame.shape[:2], dtype=np.uint8)
            cv2.circle(circle_mask, (x, y), r, 255, -1)

            # Calculate color features
            circle_region = cv2.bitwise_and(frame, frame, mask=circle_mask)
            mask_pixels = circle_mask > 0
            if not np.any(mask_pixels):
                continue

            # Calculate white percentage (soccer ball is often white)
            hsv = cv2.cvtColor(circle_region, cv2.COLOR_BGR2HSV)
            s_channel = hsv[:, :, 1]
            v_channel = hsv[:, :, 2]
            white_pixels = np.logical_and(s_channel < 30, v_channel > 200)
            white_percentage = np.sum(white_pixels) / np.sum(mask_pixels)

            # Calculate roundness (using contour analysis)
            gray_circle = cv2.cvtColor(circle_region, cv2.COLOR_BGR2GRAY)
            ret, thresh = cv2.threshold(
                gray_circle, 127, 255, cv2.THRESH_BINARY)
            contours, _ = cv2.findContours(
                thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            roundness = 0
            if contours:
                largest_contour = max(contours, key=cv2.contourArea)
                area = cv2.contourArea(largest_contour)
                perimeter = cv2.arcLength(largest_contour, True)
                if perimeter > 0:
                    roundness = 4 * np.pi * area / \
                        (perimeter * perimeter)  # 1 for perfect circle

            # Score based on color and shape
            score = white_percentage * 0.7 + roundness * 0.3
            ball_scores.append((x, y, r, score))

        # Return highest scoring ball
        if ball_scores:
            ball_scores.sort(key=lambda x: x[3], reverse=True)
            x, y, r, _ = ball_scores[0]
            return (x, y, r)

        return None

    def track_with_model(self, frame: np.ndarray) -> Optional[Tuple[int, int, int]]:
        """
        Track ball using ML model.

        Args:
            frame: Input frame

        Returns:
            Ball position as (x, y, radius) or None
        """
        results = self.model(frame, conf=self.conf_threshold, classes=[
                             32])  # 32 is sports ball in COCO

        ball_detections = []
        for result in results:
            if hasattr(result, 'boxes'):
                boxes = result.boxes
                for box in boxes:
                    if box.cls.cpu().numpy()[0] == 32:  # sports ball
                        bbox = box.xyxy.cpu().numpy()[0]  # (x1, y1, x2, y2)
                        confidence = box.conf.cpu().numpy()[0]

                        # Convert bbox to circle format
                        x1, y1, x2, y2 = bbox
                        center_x = int((x1 + x2) / 2)
                        center_y = int((y1 + y2) / 2)
                        radius = int(max(x2 - x1, y2 - y1) / 2)

                        ball_detections.append(
                            (center_x, center_y, radius, confidence))

        # Return highest confidence detection
        if ball_detections:
            ball_detections.sort(key=lambda x: x[3], reverse=True)
            x, y, r, _ = ball_detections[0]
            return (x, y, r)

        return None

    def update_kalman(self, ball_pos: Optional[Tuple[int, int, int]]) -> Optional[Tuple[int, int, int]]:
        """
        Update Kalman filter with new measurement.

        Args:
            ball_pos: Detected ball position (x, y, radius) or None

        Returns:
            Updated ball position (x, y, radius) or None
        """
        if ball_pos is None:
            # Prediction only if no detection
            if self.kalman_initialized:
                prediction = self.kalman.predict()
                x, y = prediction[0, 0], prediction[1, 0]

                # Use last known radius
                if self.prev_ball_positions:
                    radius = self.prev_ball_positions[-1][2]
                else:
                    radius = 10  # Default radius

                return (int(x), int(y), radius)
            return None

        x, y, radius = ball_pos

        if not self.kalman_initialized:
            # Initialize Kalman filter with first detection
            self.kalman.statePre = np.array(
                [[x], [y], [0], [0]], dtype=np.float32)
            self.kalman_initialized = True

        # Update with measurement
        measurement = np.array([[x], [y]], dtype=np.float32)
        self.kalman.correct(measurement)

        # Predict next position
        prediction = self.kalman.predict()
        x_pred, y_pred = prediction[0, 0], prediction[1, 0]

        return (int(x_pred), int(y_pred), radius)

    def track(self, frame: np.ndarray) -> Optional[Tuple[int, int, int]]:
        """
        Track the ball in the current frame.

        Args:
            frame: Input frame

        Returns:
            Ball position as (x, y, radius) or None if not detected
        """
        # Detect ball
        ball_pos = None

        if self.use_ml_model:
            ball_pos = self.track_with_model(frame)
        else:
            circles = self.detect_circles(frame)
            ball_pos = self.filter_ball_candidates(frame, circles)

        # Apply Kalman filtering
        if self.temporal_smoothing:
            ball_pos = self.update_kalman(ball_pos)

        # Update tracking history
        if ball_pos is not None:
            self.prev_ball_positions.append(ball_pos)
            if len(self.prev_ball_positions) > self.max_history:
                self.prev_ball_positions.pop(0)

            # Update ball trajectory for pass detection
            self.ball_trajectory.append(
                (ball_pos[0], ball_pos[1], time.time()))
            if len(self.ball_trajectory) > 30:  # Keep last second or so
                self.ball_trajectory.pop(0)

        return ball_pos

    def detect_pass(self) -> bool:
        """
        Detect when the ball is passed based on velocity change.

        Returns:
            Boolean indicating if a pass was just detected
        """
        if len(self.ball_trajectory) < 3:
            return False

        # Calculate current velocity
        recent_positions = self.ball_trajectory[-3:]
        (x1, y1, t1) = recent_positions[0]
        (x2, y2, t2) = recent_positions[-1]

        if t2 - t1 < 0.001:  # Avoid division by zero
            return False

        dx = x2 - x1
        dy = y2 - y1
        dt = t2 - t1

        current_velocity = np.sqrt(dx*dx + dy*dy) / dt

        # Check for sudden velocity change
        is_pass = False
        if self.prev_velocity is not None:
            velocity_change = current_velocity - self.prev_velocity
            current_time = time.time()

            if (velocity_change > self.velocity_threshold and
                    current_time - self.last_pass_time > self.pass_cooldown):
                is_pass = True
                self.last_pass_time = current_time

        self.prev_velocity = current_velocity
        return is_pass

    def get_ball_direction(self) -> Optional[Tuple[float, float]]:
        """
        Get the current direction of the ball movement.

        Returns:
            Normalized direction vector (dx, dy) or None
        """
        if len(self.prev_ball_positions) < 2:
            return None

        # Get last two positions
        (x1, y1, _) = self.prev_ball_positions[-2]
        (x2, y2, _) = self.prev_ball_positions[-1]

        # Calculate direction vector
        dx, dy = x2 - x1, y2 - y1

        # Normalize
        magnitude = np.sqrt(dx*dx + dy*dy)
        if magnitude > 0:
            return (dx / magnitude, dy / magnitude)

        return (0, 0)
