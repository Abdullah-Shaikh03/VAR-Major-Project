# src/offside_detector.py

import numpy as np
import cv2
from typing import List, Dict, Tuple, Any, Optional
import time


class OffsideDetector:
    """
    Detects offside situations in soccer by analyzing player positions relative to defenders.
    """

    def __init__(self, attack_direction: str = 'auto', offside_threshold: float = 0.2):
        """
        Initialize the offside detector.

        Args:
            attack_direction: Direction of attack ('left', 'right', or 'auto')
            offside_threshold: Buffer distance in meters to account for margin of error
        """
        self.attack_direction = attack_direction
        self.offside_threshold = offside_threshold
        self.team_directions = {
            'team_a': None,  # Will be determined based on positions
            'team_b': None
        }
        self.last_pass_time = 0
        self.offside_line_positions = {
            'team_a': None,
            'team_b': None
        }
        self.currently_offside = []
        
        # For tracking possession
        self.possession_team = None
        self.possession_confidence = 0.0
        
        # For tracking play state
        self.play_state = "active"  # "active", "goal_kick", "free_kick", "throw_in", etc.
        
        # For tracking the ball's state at moment of pass
        self.ball_position_at_pass = None
        self.players_positions_at_pass = None
        self.pass_direction = None

    def determine_attack_direction(self, player_positions: Dict) -> Dict[str, str]:
        """
        Determine which direction each team is attacking.

        Args:
            player_positions: Dictionary of player positions by team

        Returns:
            Dictionary mapping team to direction ('left' or 'right')
        """
        team_directions = {}
        
        # Calculate average x-coordinate for each team
        for team, players in player_positions.items():
            if team in ['team_a', 'team_b'] and players:
                x_coords = [player['position'][0] for player in players if player['position']]
                if x_coords:
                    avg_x = sum(x_coords) / len(x_coords)
                    # Determine direction based on which half they're in
                    if avg_x < 52.5:  # Left half of pitch
                        team_directions[team] = 'right'  # They're attacking right
                    else:
                        team_directions[team] = 'left'  # They're attacking left
        
        # Store for later use
        if 'team_a' in team_directions:
            self.team_directions['team_a'] = team_directions['team_a']
        if 'team_b' in team_directions:
            self.team_directions['team_b'] = team_directions['team_b']
        
        return team_directions

    def calculate_offside_line(self, team: str, player_positions: Dict) -> Optional[float]:
        """
        Calculate the offside line for a given team (position of second-last defender).

        Args:
            team: Team identifier ('team_a' or 'team_b')
            player_positions: Dictionary of player positions by team

        Returns:
            x-coordinate of the offside line or None if cannot be determined
        """
        defending_team = 'team_b' if team == 'team_a' else 'team_a'
        
        if defending_team not in player_positions:
            return None
        
        defending_players = player_positions[defending_team]
        if not defending_players or len(defending_players) < 2:
            return None
            
        # Get attacking direction
        if self.team_directions[team] is None:
            self.determine_attack_direction(player_positions)
        
        attacking_direction = self.team_directions[team]
        if attacking_direction is None:
            return None
            
        # Get x-coordinates of defending players
        x_coords = [player['position'][0] for player in defending_players if player['position']]
        if not x_coords or len(x_coords) < 2:
            return None
            
        # Sort coordinates based on attacking direction
        if attacking_direction == 'right':
            # Attacking right, defenders closest to left goal
            x_coords.sort(reverse=True)  # Descending order
        else:
            # Attacking left, defenders closest to right goal
            x_coords.sort()  # Ascending order
        
        # Second last defender position (or last if goalkeeper not detected)
        offside_line = x_coords[1] if len(x_coords) > 2 else x_coords[0]
        
        # Store the offside line position
        self.offside_line_positions[team] = offside_line
        
        return offside_line

    def detect_possession(self, ball_position: Optional[Tuple[float, float]], 
                         player_positions: Dict,
                         ball_direction: Optional[Tuple[float, float]] = None) -> str:
        """
        Determine which team has possession based on proximity to the ball.

        Args:
            ball_position: Position of the ball in pitch coordinates
            player_positions: Dictionary of player positions by team
            ball_direction: Optional direction vector of the ball

        Returns:
            Team identifier ('team_a', 'team_b', or 'unknown')
        """
        if ball_position is None:
            return 'unknown'
            
        ball_x, ball_y = ball_position
        closest_distance = float('inf')
        closest_team = 'unknown'

        for team, players in player_positions.items():
            if team in ['team_a', 'team_b']:
                for player in players:
                    if player['position']:
                        player_x, player_y = player['position']
                        distance = np.sqrt((player_x - ball_x)**2 + (player_y - ball_y)**2)
                        
                        # Adjust distance based on ball direction (players in direction of ball movement more likely to get possession)
                        if ball_direction:
                            dx, dy = ball_direction
                            player_dir_x, player_dir_y = player_x - ball_x, player_y - ball_y
                            alignment = dx * player_dir_x + dy * player_dir_y
                            if alignment > 0:  # Ball is moving toward this player
                                distance *= 0.8  # Reduce distance to increase likelihood
                        
                        if distance < closest_distance:
                            closest_distance = distance
                            closest_team = team
        
        # Update possession only if confident (player is close to ball)
        if closest_distance < 3.0:  # Within 3 meters
            self.possession_team = closest_team
            self.possession_confidence = 1.0 - (closest_distance / 5.0)  # Higher confidence if closer
        else:
            # Decay confidence over time if no close player
            self.possession_confidence *= 0.9
            
        return self.possession_team if self.possession_confidence > 0.3 else 'unknown'

    def detect_pass(self, ball_position: Optional[Tuple[float, float]], 
                   is_pass: bool, 
                   player_positions: Dict,
                   ball_direction: Optional[Tuple[float, float]] = None) -> Tuple[bool, List]:
        """
        Detect if a pass was made and check for offside.

        Args:
            ball_position: Position of the ball in pitch coordinates
            is_pass: Flag indicating if a pass was detected
            player_positions: Dictionary of player positions by team
            ball_direction: Optional direction vector of the ball

        Returns:
            Tuple of (pass_detected, offside_players)
        """
        current_time = time.time()
        offside_players = []
        
        # Check for a pass
        if is_pass and current_time - self.last_pass_time > 0.5:  # Minimum time between passes
            self.last_pass_time = current_time
            
            # Determine possession team
            possession_team = self.detect_possession(ball_position, player_positions, ball_direction)
            
            if possession_team in ['team_a', 'team_b']:
                # Store ball position at moment of pass
                self.ball_position_at_pass = ball_position
                self.pass_direction = ball_direction
                self.players_positions_at_pass = {
                    'team_a': [player.copy() for player in player_positions.get('team_a', [])],
                    'team_b': [player.copy() for player in player_positions.get('team_b', [])],
                }
                
                # Calculate offside line
                offside_line = self.calculate_offside_line(possession_team, player_positions)
                
                if offside_line is not None and self.team_directions[possession_team] is not None:
                    # Check attacking players for offside
                    attacking_team_players = player_positions[possession_team]
                    attacking_direction = self.team_directions[possession_team]
                    
                    for player in attacking_team_players:
                        if player['position']:
                            player_x = player['position'][0]
                            
                            # Check if player is ahead of offside line
                            is_offside = False
                            if attacking_direction == 'right':
                                is_offside = player_x > offside_line + self.offside_threshold
                            else:  # attacking left
                                is_offside = player_x < offside_line - self.offside_threshold
                                
                            # Check if player is in offside position
                            if is_offside:
                                # Check if player is involved in play
                                # This is a simplified check - in a real system this would be more complex
                                if ball_direction:
                                    player_pos = np.array(player['position'])
                                    ball_pos = np.array(ball_position)
                                    direction_vec = np.array(ball_direction)
                                    
                                    # Check if ball is moving toward this player
                                    player_vec = player_pos - ball_pos
                                    player_dist = np.linalg.norm(player_vec)
                                    
                                    if player_dist > 0:
                                        player_dir = player_vec / player_dist
                                        dot_product = np.dot(player_dir, direction_vec)
                                        
                                        # If ball is moving toward player and they're in offside position
                                        if dot_product > 0.7:  # Threshold for alignment
                                            offside_players.append(player)
                
                # Update currently offside list
                self.currently_offside = offside_players
                
            return True, offside_players
            
        return False, self.currently_offside

    def check_offside_exceptions(self, player_positions: Dict) -> None:
        """
        Check for situations where offside rule doesn't apply.
        
        Args:
            player_positions: Dictionary of player positions by team
        """
        # In a real system, we'd implement checks for:
        # 1. Player in own half
        # 2. Goal kick
        # 3. Throw-in
        # 4. Corner kick
        # These would require additional game state tracking
        pass

    def get_offside_line(self, team: str) -> Optional[float]:
        """
        Get the current offside line for a team.
        
        Args:
            team: Team identifier ('team_a' or 'team_b')
            
        Returns:
            x-coordinate of the offside line or None
        """
        return self.offside_line_positions.get(team)

    def get_attack_direction(self, team: str) -> Optional[str]:
        """
        Get the current attack direction for a team.
        
        Args:
            team: Team identifier ('team_a' or 'team_b')
            
        Returns:
            Direction ('left' or 'right') or None
        """
        return self.team_directions.get(team)

    def visualize_offside_line(self, top_down_view: np.ndarray, team: str, 
                              pitch_dims: Tuple[int, int]) -> np.ndarray:
        """
        Visualize the offside line on a top-down view.
        
        Args:
            top_down_view: Top-down view image
            team: Team identifier ('team_a' or 'team_b')
            pitch_dims: Dimensions of the pitch visualization (width, height)
            
        Returns:
            Top-down view with offside line drawn
        """
        offside_line = self.offside_line_positions.get(team)
        if offside_line is None:
            return top_down_view
            
        # Calculate pixel position based on pitch dimensions
        pitch_width, pitch_height = 105.0, 68.0  # Standard pitch size in meters
        view_width, view_height = pitch_dims
        
        x_pixels = int(offside_line * view_width / pitch_width)
        
        # Draw vertical line
        color = (0, 0, 255) if team == 'team_a' else (255, 0, 0)
        cv2.line(top_down_view, (x_pixels, 0), (x_pixels, view_height), color, 2)
        
        # Label the line
        label = f"{team} Offside Line"
        cv2.putText(top_down_view, label, (x_pixels + 5, 20), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
                   
        return top_down_view