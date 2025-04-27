# src/video_processor.py

import cv2
import numpy as np
import os
from typing import Tuple, Optional


class VideoProcessor:
    """
    Handles video loading, frame extraction, and preprocessing.
    """

    def __init__(self, video_path: str, resize_dim: Optional[Tuple[int, int]] = None):
        """
        Initialize the video processor.

        Args:
            video_path: Path to the video file
            resize_dim: Optional dimensions to resize frames to (width, height)
        """
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video file not found: {video_path}")

        self.video_path = video_path
        self.cap = cv2.VideoCapture(video_path)
        self.resize_dim = resize_dim

        # Get video properties
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # Frame buffer for temporal analysis
        self.frame_buffer = []
        self.buffer_size = 5  # Number of frames to keep in buffer

    def get_dimensions(self) -> Tuple[int, int]:
        """Return current video dimensions (width, height)."""
        return (self.width, self.height)

    def get_fps(self) -> float:
        """Return video FPS."""
        return self.fps

    def get_total_frames(self) -> int:
        """Return total number of frames in the video."""
        return self.total_frames

    def read_frame(self) -> Tuple[bool, Optional[np.ndarray]]:
        """
        Read the next frame from the video.

        Returns:
            Tuple of (success, frame) where success is a boolean indicating if frame was read successfully
            and frame is the numpy array containing the frame data or None if reading failed.
        """
        success, frame = self.cap.read()
        if not success:
            return False, None

        # Add to buffer and maintain buffer size
        self.frame_buffer.append(frame.copy())
        if len(self.frame_buffer) > self.buffer_size:
            self.frame_buffer.pop(0)

        # Resize if specified
        if self.resize_dim:
            frame = cv2.resize(frame, self.resize_dim)

        return True, frame

    def get_buffer(self) -> list:
        """Return the current frame buffer for temporal analysis."""
        return self.frame_buffer

    def seek_to_frame(self, frame_number: int) -> bool:
        """
        Seek to a specific frame in the video.

        Args:
            frame_number: Frame number to seek to (0-indexed)

        Returns:
            Boolean indicating if seeking was successful
        """
        if frame_number < 0 or frame_number >= self.total_frames:
            return False

        self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        self.frame_buffer = []  # Clear buffer after seeking
        return True

    def frame_to_timestamp(self, frame_number: int) -> str:
        """
        Convert frame number to timestamp string (MM:SS.ms).

        Args:
            frame_number: Frame number to convert

        Returns:
            Timestamp string in MM:SS.ms format
        """
        seconds = frame_number / self.fps
        minutes = int(seconds // 60)
        seconds = seconds % 60
        return f"{minutes:02d}:{seconds:06.3f}"

    def extract_key_frames(self, interval: int = 5) -> list:
        """
        Extract frames at regular intervals.

        Args:
            interval: Extract every nth frame

        Returns:
            List of extracted frames
        """
        current_pos = int(self.cap.get(cv2.CAP_PROP_POS_FRAMES))
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

        frames = []
        count = 0
        while True:
            success, frame = self.cap.read()
            if not success:
                break

            if count % interval == 0:
                frames.append(frame)

            count += 1

        # Restore original position
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, current_pos)
        return frames

    def detect_scene_changes(self, threshold: float = 30.0) -> list:
        """
        Detect significant scene changes in the video.

        Args:
            threshold: Threshold for scene change detection

        Returns:
            List of frame numbers where scene changes occur
        """
        current_pos = int(self.cap.get(cv2.CAP_PROP_POS_FRAMES))
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

        prev_frame = None
        scene_changes = []
        count = 0

        while True:
            success, frame = self.cap.read()
            if not success:
                break

            # Convert to grayscale for comparison
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            if prev_frame is not None:
                # Calculate mean absolute difference
                diff = cv2.absdiff(gray, prev_frame)
                mean_diff = cv2.mean(diff)[0]

                if mean_diff > threshold:
                    scene_changes.append(count)

            prev_frame = gray
            count += 1

        # Restore original position
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, current_pos)
        return scene_changes

    def reset(self) -> None:
        """Reset the video to the beginning and clear the buffer."""
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        self.frame_buffer = []

    def release(self) -> None:
        """Release the video capture resource."""
        self.cap.release()
