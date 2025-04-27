# src/utils/metrics.py

from typing import List, Dict, Any
import json

class OffsideMetricsEvaluator:
    """Evaluates system performance against ground truth data."""
    
    def __init__(self, ground_truth_path: str):
        """
        Initialize the metrics evaluator.
        
        Args:
            ground_truth_path: Path to JSON file containing ground truth annotations
        """
        with open(ground_truth_path) as f:
            self.ground_truth = json.load(f)
        
        self.true_positives = 0
        self.false_positives = 0
        self.false_negatives = 0
        self.total_frames = 0
        
    def update(self, frame_number: int, system_result: Dict):
        """
        Update metrics with system's offside determination for a frame.
        
        Args:
            frame_number: Current frame number
            system_result: System's offside analysis result
        """
        self.total_frames += 1
        
        # Find ground truth for this frame
        gt_for_frame = next(
            (gt for gt in self.ground_truth if gt['frame'] == frame_number), None)
        
        if gt_for_frame:
            gt_is_offside = gt_for_frame['is_offside']
            system_is_offside = system_result['is_offside']
            
            if gt_is_offside and system_is_offside:
                self.true_positives += 1
            elif gt_is_offside and not system_is_offside:
                self.false_negatives += 1
            elif not gt_is_offside and system_is_offside:
                self.false_positives += 1
    
    def compute_metrics(self) -> Dict[str, float]:
        """
        Compute evaluation metrics.
        
        Returns:
            Dictionary containing precision, recall, and F1 score
        """
        precision = self.true_positives / (self.true_positives + self.false_positives + 1e-6)
        recall = self.true_positives / (self.true_positives + self.false_negatives + 1e-6)
        f1_score = 2 * (precision * recall) / (precision + recall + 1e-6)
        
        return {
            'precision': precision,
            'recall': recall,
            'f1_score': f1_score,
            'true_positives': self.true_positives,
            'false_positives': self.false_positives,
            'false_negatives': self.false_negatives,
            'total_frames': self.total_frames
        }