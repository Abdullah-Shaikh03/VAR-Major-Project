# src/offside_analyzer.py

from typing import List, Dict, Tuple, Any, Optional
import numpy as np
from src.offside_detector import OffsideDetector

class OffsideAnalyzer:
    """Analyzes offside situations based on player and ball positions."""
    
    def __init__(self, attack_direction: str = 'auto'):
        """
        Initialize the offside analyzer.
        
        Args:
            attack_direction: Direction of attack ('left', 'right', or 'auto')
        """
        self.offside_detector = OffsideDetector(attack_direction=attack_direction)
        self.last_offside_event = None
        
    def analyze(self, player_positions: Dict, 
               ball_position: Tuple[float, float],
               ball_direction: Optional[Tuple[float, float]] = None) -> Dict:
        """
        Analyze current frame for offside situations.
        
        Args:
            player_positions: Dictionary of player positions by team
            ball_position: Current ball position in pitch coordinates
            ball_direction: Optional direction vector of the ball
            
        Returns:
            Dictionary containing offside analysis results
        """
        # Detect possession
        possession_team = self.offside_detector.detect_possession(
            ball_position, player_positions, ball_direction)
        
        # Check for pass and offside
        is_pass = True  # Assume pass was detected by ball tracker
        pass_detected, offside_players = self.offside_detector.detect_pass(
            ball_position, is_pass, player_positions, ball_direction)
        
        result = {
            'is_offside': False,
            'offside_player': None,
            'offside_distance': 0,
            'offside_line': None,
            'possession_team': possession_team
        }
        
        if pass_detected and offside_players:
            # Get the most offside player (furthest beyond the line)
            offside_player = max(offside_players, key=lambda p: abs(
                p['position'][0] - self.offside_detector.get_offside_line(possession_team)))
            
            result['is_offside'] = True
            result['offside_player'] = offside_player
            result['offside_line'] = self.offside_detector.get_offside_line(possession_team)
            
            # Calculate distance to offside line
            if result['offside_line'] is not None:
                player_x = offside_player['position'][0]
                attack_dir = self.offside_detector.get_attack_direction(possession_team)
                
                if attack_dir == 'right':
                    result['offside_distance'] = player_x - result['offside_line']
                else:
                    result['offside_distance'] = result['offside_line'] - player_x
                
            self.last_offside_event = result
        
        return result
    
    def get_last_offside_event(self) -> Optional[Dict]:
        """Get the last detected offside event."""
        return self.last_offside_event