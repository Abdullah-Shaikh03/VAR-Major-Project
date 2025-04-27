# src/team_classifier.py

import cv2
import numpy as np
from typing import List, Dict, Tuple, Any
from sklearn.cluster import KMeans


class TeamClassifier:
    """
    Classifies detected players into teams based on jersey colors.
    """

    def __init__(self, n_clusters: int = 3, samples_per_player: int = 10):
        """
        Initialize the team classifier.

        Args:
            n_clusters: Number of color clusters to identify (typically 3: team1, team2, referee)
            samples_per_player: Number of color samples to take from each player detection
        """
        self.n_clusters = n_clusters
        self.samples_per_player = samples_per_player
        self.team_colors = None  # Will store the representative colors for each team
        self.initialized = False
        self.kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)

        # Team labels
        self.TEAM_A = 0
        self.TEAM_B = 1
        self.REFEREE = 2  # Or could be goalkeeper, etc.
        self.UNKNOWN = -1

    def calibrate(self, frame: np.ndarray, player_detections: List) -> None:
        """
        Calibrate team colors based on detected players.

        Args:
            frame: Current video frame
            player_detections: List of PlayerDetection objects
        """
        if len(player_detections) < self.n_clusters:
            return  # Not enough players to determine teams

        # Extract color features from players
        color_samples = []

        for detection in player_detections:
            x1, y1, x2, y2 = detection.bbox
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

            # Crop player region
            player_region = frame[y1:y2, x1:x2]
            if player_region.size == 0:  # Skip invalid regions
                continue

            # Extract color samples from center of player (avoid including background)
            h, w = player_region.shape[:2]
            center_x, center_y = w // 2, h // 2
            sample_region = player_region[
                max(0, center_y - h//4):min(h, center_y + h//4),
                max(0, center_x - w//4):min(w, center_x + w//4)
            ]

            # Convert to HSV for better color clustering
            sample_hsv = cv2.cvtColor(sample_region, cv2.COLOR_BGR2HSV)

            # Reshape to get all pixels and sample randomly
            pixels = sample_hsv.reshape(-1, 3)
            if len(pixels) > self.samples_per_player:
                indices = np.random.choice(
                    len(pixels), self.samples_per_player, replace=False)
                samples = pixels[indices]
                color_samples.extend(samples)

        # Perform clustering to find team colors
        if len(color_samples) >= self.n_clusters:
            color_samples = np.array(color_samples)
            self.kmeans.fit(color_samples)
            self.team_colors = self.kmeans.cluster_centers_

            # Sort colors by hue to make team assignment more consistent
            hue_values = self.team_colors[:, 0]
            sorted_indices = np.argsort(hue_values)
            self.team_colors = self.team_colors[sorted_indices]

            self.initialized = True

    def classify(self, frame: np.ndarray, player_detections: List) -> Dict[int, int]:
        """
        Classify detected players into teams.

        Args:
            frame: Current video frame
            player_detections: List of PlayerDetection objects

        Returns:
            Dictionary mapping player indices to team labels
        """
        if not self.initialized or len(player_detections) == 0:
            return {}

        team_assignments = {}

        for i, detection in enumerate(player_detections):
            x1, y1, x2, y2 = map(int, detection.bbox)

            # Crop player region
            player_region = frame[y1:y2, x1:x2]
            if player_region.size == 0:
                team_assignments[i] = self.UNKNOWN
                continue

            # Extract color samples from torso area
            h, w = player_region.shape[:2]
            torso_region = player_region[
                max(0, h//3):min(h, 2*h//3),  # Middle third vertically (torso)
                max(0, w//4):min(w, 3*w//4)   # Middle half horizontally
            ]

            if torso_region.size == 0:
                team_assignments[i] = self.UNKNOWN
                continue

            # Convert to HSV
            torso_hsv = cv2.cvtColor(torso_region, cv2.COLOR_BGR2HSV)

            # Calculate average color
            avg_color = np.mean(torso_hsv.reshape(-1, 3),
                                axis=0).reshape(1, -1)

            # Find closest team color
            distances = np.sqrt(
                np.sum((self.team_colors - avg_color)**2, axis=1))
            closest_team = np.argmin(distances)

            team_assignments[i] = closest_team

            # Assign to player detection object as well
            detection.team = closest_team

            # Check if likely a goalkeeper based on distinct color
            if len(distances) > 1:
                sorted_distances = np.sort(distances)
                # Significantly different from other teams
                if sorted_distances[1] / sorted_distances[0] > 1.5:
                    detection.is_goalkeeper = True

        return team_assignments

    def detect_jersey_numbers(self, frame: np.ndarray, player_detections: List) -> None:
        """
        Attempt to detect jersey numbers for players.
        This is a more advanced feature that would require OCR integration.

        Args:
            frame: Current video frame
            player_detections: List of PlayerDetection objects
        """
        # This would be implemented with OCR or a specialized jersey number detection model
        # For now, this is a placeholder
        pass

    def get_team_colors(self) -> Dict[int, Tuple[int, int, int]]:
        """
        Get the BGR values for each team's jersey color.

        Returns:
            Dictionary mapping team IDs to BGR color values
        """
        if not self.initialized:
            return {}

        team_bgr_colors = {}
        for i, hsv_color in enumerate(self.team_colors):
            # Convert HSV cluster centers back to BGR
            hsv_color_uint8 = np.array(
                [[hsv_color[0], 255, 255]], dtype=np.uint8)
            bgr_color = cv2.cvtColor(hsv_color_uint8.reshape(
                1, 1, 3), cv2.COLOR_HSV2BGR)[0, 0]
            team_bgr_colors[i] = tuple(map(int, bgr_color))

        return team_bgr_colors
