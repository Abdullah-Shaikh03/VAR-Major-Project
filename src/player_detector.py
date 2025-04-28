# src/player_detector.py

import cv2
import numpy as np
import torch
from typing import List, Dict, Tuple, Any, Optional
import os

# Import YOLOv8 from Ultralytics
try:
    from ultralytics import YOLO
except ImportError:
    raise ImportError(
        "Ultralytics YOLO is required. Install with 'pip install ultralytics'")


class PlayerDetection:
    """Class representing a detected player."""

    def __init__(self, bbox: Tuple[float, float, float, float],
                 confidence: float,
                 player_id: Optional[int] = None,
                 segmentation_mask: Optional[np.ndarray] = None):
        """
        Initialize a player detection.

        Args:
            bbox: Bounding box coordinates (x1, y1, x2, y2)
            confidence: Detection confidence score
            player_id: Optional player tracking ID
            segmentation_mask: Optional segmentation mask
        """
        self.bbox = bbox  # (x1, y1, x2, y2)
        self.confidence = confidence
        self.player_id = player_id
        self.segmentation_mask = segmentation_mask
        self.team = None  # Will be assigned by TeamClassifier
        self.position_2d = None  # Will be assigned by PerspectiveTransformer
        self.is_goalkeeper = False
        self.jersey_number = None
        self.furthest_forward_point = None  # Critical for offside determination

    @property
    def center(self) -> Tuple[float, float]:
        """Return the center point of the bounding box."""
        x1, y1, x2, y2 = self.bbox
        return ((x1 + x2) / 2, (x1 + x2) / 2)

    @property
    def bottom_center(self) -> Tuple[float, float]:
        """Return the bottom center point of the bounding box (player's feet)."""
        x1, y1, x2, y2 = self.bbox
        return ((x1 + x2) / 2, y2)

    @property
    def width(self) -> float:
        """Return the width of the bounding box."""
        return self.bbox[2] - self.bbox[0]

    @property
    def height(self) -> float:
        """Return the height of the bounding box."""
        return self.bbox[3] - self.bbox[1]

    @property
    def area(self) -> float:
        """Return the area of the bounding box."""
        return self.width * self.height

    def compute_furthest_forward_point(self, direction: str = 'right') -> None:
        """
        Compute the furthest forward point of the player (based on segmentation if available).
        This is crucial for accurate offside detection as the rules consider any playable body part.

        Args:
            direction: Direction of attack ('right' or 'left')
        """
        if self.segmentation_mask is not None:
            # Find contour of the segmentation mask
            contours, _ = cv2.findContours(self.segmentation_mask.astype(np.uint8),
                                           cv2.RETR_EXTERNAL,
                                           cv2.CHAIN_APPROX_NONE)

            if contours:
                # Flatten contour points
                points = np.vstack(contours).squeeze()

                # Find the furthest point based on direction
                if direction == 'right':
                    idx = np.argmax(points[:, 0])
                    self.furthest_forward_point = (
                        points[idx, 0], points[idx, 1])
                else:  # left
                    idx = np.argmin(points[:, 0])
                    self.furthest_forward_point = (
                        points[idx, 0], points[idx, 1])
        else:
            # If no segmentation, use bounding box
            x1, y1, x2, y2 = self.bbox
            if direction == 'right':
                self.furthest_forward_point = (x2, (y1 + y2) / 2)
            else:  # left
                self.furthest_forward_point = (x1, (y1 + y2) / 2)


class PlayerDetector:
    """
    Detects and tracks players in video frames using YOLOv8.
    """

    def __init__(self, model_path: str = 'yolo11n.pt', device: str = 'cuda',
                 conf_threshold: float = 0.25, track: bool = True):
        """
        Initialize the player detector.

        Args:
            model_path: Path to YOLOv8 model weights
            device: Device to run inference on ('cuda' or 'cpu')
            conf_threshold: Confidence threshold for detections
            track: Whether to track players across frames
        """
        # Check if model exists, download if not
        if not os.path.exists(model_path):
            print(f"Model not found at {model_path}, downloading YOLOv8...")
            self.model = YOLO('yolo11n.pt')
        else:
            self.model = YOLO(model_path)

        self.device = device
        self.conf_threshold = conf_threshold
        self.track = track
        self.class_names = self.model.names

        # Person class ID in COCO dataset (used by YOLOv8)
        self.person_class_id = 0

    def detect(self, frame: np.ndarray) -> List[PlayerDetection]:
        """
        Detect players in a frame.

        Args:
            frame: Input frame as numpy array

        Returns:
            List of PlayerDetection objects
        """
        # Run YOLOv8 inference
        if self.track:
            results = self.model.track(frame, conf=self.conf_threshold, classes=[
                                       self.person_class_id], device=self.device)
        else:
            results = self.model(frame, conf=self.conf_threshold, classes=[
                                 self.person_class_id], device=self.device)

        # Extract detections
        player_detections = []

        for result in results:
            # Handle results based on version
            if hasattr(result, 'boxes'):
                # New YOLO version
                boxes = result.boxes
                if boxes.id is not None:
                    track_ids = boxes.id.int().cpu().tolist()
                else:
                    track_ids = [None] * len(boxes)

                for i, box in enumerate(boxes):
                    if box.cls.cpu().numpy()[0] == self.person_class_id:
                        bbox = box.xyxy.cpu().numpy()[0]  # (x1, y1, x2, y2)
                        confidence = box.conf.cpu().numpy()[0]

                        # Get segmentation mask if available
                        mask = None
                        if hasattr(result, 'masks') and result.masks is not None:
                            if i < len(result.masks):
                                mask = result.masks[i].data[0].cpu().numpy()

                        # Create PlayerDetection object
                        player_detection = PlayerDetection(
                            bbox=tuple(bbox),
                            confidence=confidence,
                            player_id=int(
                                track_ids[i]) if track_ids[i] is not None else None,
                            segmentation_mask=mask
                        )

                        # Compute furthest forward point (critical for offside)
                        player_detection.compute_furthest_forward_point()

                        player_detections.append(player_detection)
            else:
                # Legacy handling for older YOLO versions
                if result.pred is not None and len(result.pred) > 0:
                    for *box, conf, cls_id in result.pred[0].cpu().numpy():
                        if int(cls_id) == self.person_class_id and conf > self.conf_threshold:
                            player_detection = PlayerDetection(
                                bbox=tuple(box),
                                confidence=conf,
                                player_id=None  # No tracking ID in this mode
                            )
                            player_detection.compute_furthest_forward_point()
                            player_detections.append(player_detection)

        return player_detections

    def detect_with_pose(self, frame: np.ndarray) -> List[PlayerDetection]:
        """
        Detect players with pose estimation to better identify body parts.
        Useful for determining the furthest forward body part for offside.

        Args:
            frame: Input frame as numpy array

        Returns:
            List of PlayerDetection objects with pose information
        """
        # This is a placeholder for pose estimation implementation
        # For a real implementation, you would integrate a pose estimation model
        # such as OpenPose, HRNet, or YOLOv8-pose

        # For now, we'll just use the standard detection
        return self.detect(frame)
