# src/perspective_transformer.py

import cv2
import numpy as np
from typing import List, Dict, Tuple, Any, Optional
import matplotlib.pyplot as plt


class PerspectiveTransformer:
    """
    Transforms player and ball positions from camera view to 2D pitch coordinates.
    """

    def __init__(self, calibration_method: str = 'auto', pitch_dims: Tuple[float, float] = (105.0, 68.0)):
        """
        Initialize the perspective transformer.

        Args:
            calibration_method: Method for calibration ('auto', 'manual', or 'preset')
            pitch_dims: Real-world dimensions of the pitch in meters (length, width)
        """
        self.calibration_method = calibration_method
        self.pitch_length = pitch_dims[0]  # standard soccer pitch is 105m long
        self.pitch_width = pitch_dims[1]   # standard soccer pitch is 68m wide
        
        # Pixel coordinates of the corners of the field in the image (source)
        self.src_points = None
        
        # Real-world coordinates in meters (destination)
        self.dst_points = np.array([
            [0, 0],                               # Top-left corner
            [self.pitch_length, 0],               # Top-right corner
            [self.pitch_length, self.pitch_width], # Bottom-right corner
            [0, self.pitch_width]                 # Bottom-left corner
        ], dtype=np.float32)
        
        # Homography matrix for transformation
        self.homography_matrix = None
        
        # For calibration - store detected lines for refinement
        self.detected_lines = []
        
        # Standard field markings in meters for reference
        self.field_markings = {
            'center_circle_radius': 9.15,
            'penalty_area_length': 16.5,
            'penalty_area_width': 40.32,
            'goal_area_length': 5.5,
            'goal_area_width': 18.32,
            'penalty_mark_dist': 11.0,
            'goal_width': 7.32
        }
        
        # Transformation quality metric
        self.calibration_quality = 0.0
    
    def detect_field(self, frame: np.ndarray) -> np.ndarray:
        """
        Detect the soccer field in the image.
        
        Args:
            frame: Input frame
            
        Returns:
            Binary mask of the detected field
        """
        # Convert to HSV for better green detection
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # Define range for green color in HSV
        lower_green = np.array([35, 50, 50])
        upper_green = np.array([90, 255, 255])
        
        # Create mask for green color
        mask = cv2.inRange(hsv, lower_green, upper_green)
        
        # Apply morphological operations to clean up the mask
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        
        # Find contours of the field
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # If no contours found, return the original mask
        if not contours:
            return mask
        
        # Find the largest contour (should be the field)
        largest_contour = max(contours, key=cv2.contourArea)
        
        # Create a new mask with only the largest contour
        field_mask = np.zeros_like(mask)
        cv2.drawContours(field_mask, [largest_contour], 0, 255, -1)
        
        return field_mask
    
    def detect_field_lines(self, frame: np.ndarray, field_mask: Optional[np.ndarray] = None) -> List[np.ndarray]:
        """
        Detect white lines on the soccer field.
        
        Args:
            frame: Input frame
            field_mask: Optional mask of the detected field
            
        Returns:
            List of detected line segments
        """
        # Create grayscale image
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Apply field mask if provided
        if field_mask is not None:
            gray = cv2.bitwise_and(gray, gray, mask=field_mask)
        
        # Enhance white lines
        _, thresh = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)
        
        # Use morphological operations to clean up
        kernel = np.ones((3, 3), np.uint8)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
        
        # Detect lines using HoughLinesP
        lines = cv2.HoughLinesP(
            thresh,
            rho=1,
            theta=np.pi/180,
            threshold=50,
            minLineLength=30,
            maxLineGap=10
        )
        
        if lines is None:
            return []
        
        # Filter and clean up lines
        filtered_lines = []
        for line in lines:
            x1, y1, x2, y2 = line[0]
            # Calculate line length
            length = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
            
            # Filter out very short lines
            if length > 30:
                filtered_lines.append(line[0])
        
        return filtered_lines
    
    def find_field_corners(self, lines: List[np.ndarray], frame_shape: Tuple[int, int]) -> np.ndarray:
        """
        Estimate the corners of the field from detected lines.
        
        Args:
            lines: List of detected line segments
            frame_shape: Shape of the frame (height, width)
            
        Returns:
            Array of the four corners of the field
        """
        h, w = frame_shape[:2]
        
        # Group lines by slope (vertical or horizontal)
        vertical_lines = []
        horizontal_lines = []
        
        for line in lines:
            x1, y1, x2, y2 = line
            
            # Check if line is more vertical or horizontal
            dx, dy = x2 - x1, y2 - y1
            
            if abs(dx) > abs(dy):  # Horizontal-ish line
                horizontal_lines.append(line)
            else:  # Vertical-ish line
                vertical_lines.append(line)
        
        # If we don't have enough lines, make an educated guess
        if len(horizontal_lines) < 2 or len(vertical_lines) < 2:
            # Default to the frame boundaries with some margins
            margin = min(w, h) // 10
            default_corners = np.array([
                [margin, margin],                   # Top-left
                [w - margin, margin],               # Top-right
                [w - margin, h - margin],           # Bottom-right
                [margin, h - margin]                # Bottom-left
            ], dtype=np.float32)
            return default_corners
        
        # Find the boundaries of the field lines
        min_x, max_x = w, 0
        min_y, max_y = h, 0
        
        for line in vertical_lines + horizontal_lines:
            x1, y1, x2, y2 = line
            min_x = min(min_x, x1, x2)
            max_x = max(max_x, x1, x2)
            min_y = min(min_y, y1, y2)
            max_y = max(max_y, y1, y2)
        
        # Create corners with some margin
        margin = 10
        corners = np.array([
            [max(min_x - margin, 0), max(min_y - margin, 0)],               # Top-left
            [min(max_x + margin, w), max(min_y - margin, 0)],               # Top-right
            [min(max_x + margin, w), min(max_y + margin, h)],               # Bottom-right
            [max(min_x - margin, 0), min(max_y + margin, h)]                # Bottom-left
        ], dtype=np.float32)
        
        return corners
    
    def initialize(self, frame: np.ndarray) -> bool:
        """
        Initialize the perspective transformer from a frame.
        
        Args:
            frame: Input frame to use for calibration
            
        Returns:
            Boolean indicating if initialization was successful
        """
        if self.calibration_method == 'manual':
            # Manual calibration would require user input
            # For this implementation, we'll fall back to auto
            pass
        
        # Auto calibration
        # Detect field and field lines
        field_mask = self.detect_field(frame)
        self.detected_lines = self.detect_field_lines(frame, field_mask)
        
        # Find corners of the field
        h, w = frame.shape[:2]
        self.src_points = self.find_field_corners(self.detected_lines, (h, w))
        
        # Compute homography matrix
        self.homography_matrix = cv2.getPerspectiveTransform(self.src_points, self.dst_points)
        
        # Calculate calibration quality based on reprojection error
        self.calculate_calibration_quality()
        
        # Return whether calibration was successful (threshold of 0.7)
        return self.calibration_quality > 0.7
    
    def calculate_calibration_quality(self) -> float:
        """
        Calculate the quality of the calibration using reprojection error.
        
        Returns:
            Quality metric between 0.0 and 1.0
        """
        if self.homography_matrix is None or self.src_points is None:
            self.calibration_quality = 0.0
            return self.calibration_quality
        
        # Project source points to destination and back
        src_projected = cv2.perspectiveTransform(
            self.src_points.reshape(-1, 1, 2),
            self.homography_matrix
        ).reshape(-1, 2)
        
        inverse_matrix = np.linalg.inv(self.homography_matrix)
        reprojects = cv2.perspectiveTransform(
            src_projected.reshape(-1, 1, 2),
            inverse_matrix
        ).reshape(-1, 2)
        
        # Calculate mean reprojection error
        error = np.mean(np.sqrt(np.sum((self.src_points - reprojects)**2, axis=1)))
        
        # Convert to quality metric (0.0 to 1.0)
        # Lower error means higher quality
        max_error = np.sqrt(np.sum(np.array([1920, 1080])**2))  # Diagonal of HD frame
        quality = max(0.0, min(1.0, 1.0 - (error / max_error)))
        
        self.calibration_quality = quality
        return quality
    
    def update_calibration(self, frame: np.ndarray) -> bool:
        """
        Update calibration based on new frame (for moving camera).
        
        Args:
            frame: New video frame
            
        Returns:
            Boolean indicating if update was successful
        """
        # For static camera, we might skip recalibration
        # For moving camera, we'd need to track features from previous frames
        # This is a simplified implementation that just recalibrates
        
        return self.initialize(frame)
    
    def transform_point(self, point: Tuple[float, float]) -> Optional[Tuple[float, float]]:
        """
        Transform a single point from image coordinates to pitch coordinates.
        
        Args:
            point: Point in image coordinates (x, y)
            
        Returns:
            Point in pitch coordinates (x, y) in meters or None if invalid
        """
        if self.homography_matrix is None:
            return None
        
        # Convert point to proper format
        point_np = np.array([[point]], dtype=np.float32)
        
        # Transform point
        transformed = cv2.perspectiveTransform(point_np, self.homography_matrix)
        
        # Extract coordinates
        x, y = transformed[0][0]
        
        # Check if point is within pitch boundaries
        if 0 <= x <= self.pitch_length and 0 <= y <= self.pitch_width:
            return (x, y)
        else:
            return None
    
    def transform_players(self, player_detections: List, team_assignments: Dict[int, int]) -> Dict[int, Dict]:
        """
        Transform player positions from image to pitch coordinates.
        
        Args:
            player_detections: List of PlayerDetection objects
            team_assignments: Dictionary mapping player indices to team labels
            
        Returns:
            Dictionary mapping team to lists of player positions
        """
        pitch_positions = {
            'team_a': [],
            'team_b': [],
            'other': []
        }
        
        for i, player in enumerate(player_detections):
            if i in team_assignments:
                team = team_assignments[i]
                
                # Get player's foot position (bottom center of bounding box)
                x1, y1, x2, y2 = player.bbox
                foot_point = ((x1 + x2) / 2, y2)
                
                # Transform to pitch coordinates
                pitch_pos = self.transform_point(foot_point)
                
                if pitch_pos:
                    # Store player information
                    player_info = {
                        'position': pitch_pos,
                        'is_goalkeeper': player.is_goalkeeper,
                        'player_id': player.player_id,
                        'detection': player
                    }
                    
                    # Add to appropriate team
                    if team == 0:  # Team A
                        pitch_positions['team_a'].append(player_info)
                    elif team == 1:  # Team B
                        pitch_positions['team_b'].append(player_info)
                    else:  # Referee or other
                        pitch_positions['other'].append(player_info)
                    
                    # Store the 2D position in the player detection object too
                    player.position_2d = pitch_pos
        
        return pitch_positions
    
    def transform_ball(self, ball_position: Optional[Tuple[int, int, int]]) -> Optional[Tuple[float, float]]:
        """
        Transform ball position from image to pitch coordinates.
        
        Args:
            ball_position: Ball position as (x, y, radius) or None
            
        Returns:
            Ball position in pitch coordinates (x, y) in meters or None
        """
        if ball_position is None or self.homography_matrix is None:
            return None
        
        x, y, _ = ball_position
        return self.transform_point((x, y))
    
    def visualize_transformation(self, frame: np.ndarray, overlay: bool = True) -> np.ndarray:
        """
        Visualize the perspective transformation.
        
        Args:
            frame: Input frame
            overlay: Whether to overlay visualization on original frame
            
        Returns:
            Visualization image
        """
        if self.homography_matrix is None:
            return frame
        
        h, w = frame.shape[:2]
        vis_img = frame.copy() if overlay else np.zeros((h, w, 3), dtype=np.uint8)
        
        # Draw the detected field corners
        if self.src_points is not None:
            for i in range(4):
                pt1 = tuple(map(int, self.src_points[i]))
                pt2 = tuple(map(int, self.src_points[(i + 1) % 4]))
                cv2.line(vis_img, pt1, pt2, (0, 255, 0), 2)
                cv2.circle(vis_img, pt1, 5, (0, 0, 255), -1)
        
        # Draw grid lines on the field
        grid_step = 10  # meters
        for x in range(0, int(self.pitch_length) + 1, grid_step):
            for y in range(0, int(self.pitch_width) + 1, grid_step):
                pitch_pt = np.array([[[x, y]]], dtype=np.float32)
                
                # Transform from pitch to image
                inv_matrix = np.linalg.inv(self.homography_matrix)
                img_pt = cv2.perspectiveTransform(pitch_pt, inv_matrix)
                
                img_x, img_y = map(int, img_pt[0][0])
                if 0 <= img_x < w and 0 <= img_y < h:
                    cv2.circle(vis_img, (img_x, img_y), 3, (255, 0, 0), -1)
        
        # Draw quality metric
        quality_text = f"Calibration Quality: {self.calibration_quality:.2f}"
        cv2.putText(vis_img, quality_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                   1, (0, 255, 255), 2, cv2.LINE_AA)
        
        return vis_img
    
    def get_homography_matrix(self) -> Optional[np.ndarray]:
        """Get the homography matrix."""
        return self.homography_matrix
    
    def get_top_down_view(self, frame: np.ndarray, player_detections: List = None, 
                         ball_position: Tuple = None, width: int = 600, height: int = 400) -> np.ndarray:
        """
        Generate a top-down view of the pitch with players and ball.
        
        Args:
            frame: Original frame
            player_detections: List of PlayerDetection objects
            ball_position: Ball position as (x, y, radius) or None
            width: Width of output image in pixels
            height: Height of output image in pixels
            
        Returns:
            Top-down view image
        """
        # Create empty top-down view
        pitch_img = np.ones((height, width, 3), dtype=np.uint8) * 50  # Dark gray
        
        # Draw pitch
        # Field is green
        cv2.rectangle(pitch_img, (0, 0), (width, height), (0, 128, 0), -1)
        
        # Scale factors
        scale_x = width / self.pitch_length
        scale_y = height / self.pitch_width
        
        # Draw pitch markings
        # Center line
        cv2.line(pitch_img, 
                (int(width/2), 0), 
                (int(width/2), height), 
                (255, 255, 255), 2)
        
        # Center circle
        center_radius = int(self.field_markings['center_circle_radius'] * scale_x)
        cv2.circle(pitch_img, 
                  (int(width/2), int(height/2)), 
                  center_radius, 
                  (255, 255, 255), 2)
        
        # Draw penalty areas
        pen_area_x = int(self.field_markings['penalty_area_length'] * scale_x)
        pen_area_y = int(self.field_markings['penalty_area_width'] * scale_y)
        pen_area_start_y = int((height - pen_area_y) / 2)
        
        # Left penalty area
        cv2.rectangle(pitch_img, 
                     (0, pen_area_start_y), 
                     (pen_area_x, pen_area_start_y + pen_area_y), 
                     (255, 255, 255), 2)
        
        # Right penalty area
        cv2.rectangle(pitch_img, 
                     (width - pen_area_x, pen_area_start_y), 
                     (width, pen_area_start_y + pen_area_y), 
                     (255, 255, 255), 2)
        
        # Draw players if provided
        if player_detections:
            for player in player_detections:
                if hasattr(player, 'position_2d') and player.position_2d:
                    x, y = player.position_2d
                    x_px = int(x * scale_x)
                    y_px = int(y * scale_y)
                    
                    # Draw player as circle with team color
                    color = (0, 0, 255)  # Default red
                    if hasattr(player, 'team'):
                        if player.team == 0:
                            color = (0, 0, 255)  # Team A: Red
                        elif player.team == 1:
                            color = (255, 0, 0)  # Team B: Blue
                        else:
                            color = (0, 255, 255)  # Other: Yellow
                    
                    # Draw player
                    cv2.circle(pitch_img, (x_px, y_px), 5, color, -1)
                    
                    # Draw player ID or jersey number if available
                    if player.player_id is not None:
                        cv2.putText(pitch_img, str(player.player_id), 
                                   (x_px + 5, y_px - 5), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Draw ball if provided
        if ball_position and self.homography_matrix is not None:
            ball_pitch_pos = self.transform_ball(ball_position)
            if ball_pitch_pos:
                x, y = ball_pitch_pos
                x_px = int(x * scale_x)
                y_px = int(y * scale_y)
                
                # Draw ball
                cv2.circle(pitch_img, (x_px, y_px), 4, (255, 255, 255), -1)
        
        return pitch_img