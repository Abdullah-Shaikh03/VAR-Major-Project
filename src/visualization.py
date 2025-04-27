# src/visualization.py

import cv2
import numpy as np
from typing import List, Dict, Tuple, Any, Optional
from src.player_detector import PlayerDetection
from src.team_classifier import TeamClassifier

class Visualizer:
    """Handles visualization of detection and analysis results."""
    
    def __init__(self, pitch_color: Tuple[int, int, int] = (0, 128, 0)):
        """
        Initialize the visualizer.
        
        Args:
            pitch_color: RGB color for the pitch in top-down view
        """
        self.pitch_color = pitch_color
        self.team_colors = {
            0: (0, 0, 255),    # Team A - Red
            1: (255, 0, 0),    # Team B - Blue
            2: (0, 255, 255),  # Referee - Yellow
            -1: (255, 255, 255) # Unknown - White
        }
        
    def draw_results(self, frame: np.ndarray,
                    player_detections: List[PlayerDetection],
                    team_assignments: Dict[int, int],
                    ball_position: Optional[Tuple[int, int, int]],
                    homography_matrix: Optional[np.ndarray],
                    offside_result: Optional[Dict]) -> np.ndarray:
        """
        Draw detection and analysis results on the frame.
        
        Args:
            frame: Original video frame
            player_detections: List of detected players
            team_assignments: Dictionary mapping player indices to teams
            ball_position: Ball position (x, y, radius) or None
            homography_matrix: Homography matrix for perspective transform
            offside_result: Offside analysis result or None
            
        Returns:
            Frame with visualizations overlaid
        """
        # Draw player bounding boxes
        for i, player in enumerate(player_detections):
            team = team_assignments.get(i, -1)
            color = self.team_colors.get(team, (255, 255, 255))
            
            # Draw bounding box
            x1, y1, x2, y2 = map(int, player.bbox)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            
            # Draw player ID and team
            label = f"ID: {player.player_id}" if player.player_id is not None else f"P{i}"
            if player.is_goalkeeper:
                label += " (GK)"
            cv2.putText(frame, label, (x1, y1 - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
            
            # Draw furthest forward point if available
            if player.furthest_forward_point:
                fx, fy = map(int, player.furthest_forward_point)
                cv2.circle(frame, (fx, fy), 3, (0, 255, 0), -1)
        
        # Draw ball
        if ball_position:
            x, y, r = map(int, ball_position)
            cv2.circle(frame, (x, y), r, (0, 0, 255), -1)
            cv2.circle(frame, (x, y), r, (255, 255, 255), 2)
        
        # Draw offside result if available
        if offside_result and offside_result['is_offside']:
            player = offside_result['offside_player']
            x1, y1, x2, y2 = map(int, player['detection'].bbox)
            
            # Highlight offside player
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 3)
            cv2.putText(frame, "OFFSIDE", (x1, y1 - 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
            # Draw distance to offside line
            if 'offside_distance' in offside_result:
                dist = offside_result['offside_distance']
                cv2.putText(frame, f"{dist:.2f}m", (x1, y1 - 60),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        
        return frame
    
    def draw_top_down_view(self, frame: np.ndarray,
                         perspective_transformer,
                         player_detections: List[PlayerDetection],
                         ball_position: Optional[Tuple[int, int, int]]) -> np.ndarray:
        """
        Generate a top-down view of the pitch with player and ball positions.
        
        Args:
            frame: Original frame (for dimensions)
            perspective_transformer: PerspectiveTransformer instance
            player_detections: List of detected players
            ball_position: Ball position (x, y, radius) or None
            
        Returns:
            Top-down view image
        """
        return perspective_transformer.get_top_down_view(frame, player_detections, ball_position)