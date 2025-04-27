# main.py - System entrypoint

import argparse
import cv2
import numpy as np
import torch
import os
from datetime import datetime

from src.video_processor import VideoProcessor
from src.player_detector import PlayerDetector
from src.team_classifier import TeamClassifier
from src.ball_tracker import BallTracker
from src.perspective_transformer import PerspectiveTransformer
from src.offside_analyzer import OffsideAnalyzer
from src.visualization import Visualizer
from src.utils.matrics import OffsideMetricsEvaluator
from src.utils.logger import setup_logger

def parse_args():
    parser = argparse.ArgumentParser(description='Soccer Offside Detection System')
    parser.add_argument('--video_path', type=str, required=True, help='Path to input video file')
    parser.add_argument('--output_dir', type=str, default='outputs', help='Directory to save outputs')
    parser.add_argument('--model_path', type=str, default='models/yolov8x.pt', help='Path to YOLOv8 model')
    parser.add_argument('--ground_truth', type=str, default=None, help='Path to ground truth annotations (if available)')
    parser.add_argument('--visualize', action='store_true', help='Enable visualization')
    parser.add_argument('--save_video', action='store_true', help='Save output video')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu', 
                        help='Device to run inference on: cuda or cpu')
    
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Setup logging
    logger = setup_logger(__name__)
    logger.info(f"Starting offside detection system on {args.device}")
    
    # Create output directory if it doesn't exist
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(args.output_dir, f"run_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize components
    video_processor = VideoProcessor(args.video_path)
    player_detector = PlayerDetector(model_path=args.model_path, device=args.device)
    team_classifier = TeamClassifier()
    ball_tracker = BallTracker(device=args.device)
    perspective_transformer = PerspectiveTransformer()
    offside_analyzer = OffsideAnalyzer()
    
    # Initialize visualization if enabled
    visualizer = None
    output_video = None
    if args.visualize or args.save_video:
        visualizer = Visualizer()
        if args.save_video:
            output_path = os.path.join(output_dir, 'output_video.mp4')
            frame_width, frame_height = video_processor.get_dimensions()
            output_video = cv2.VideoWriter(output_path, 
                                          cv2.VideoWriter_fourcc(*'mp4v'),
                                          video_processor.get_fps(),
                                          (frame_width, frame_height))
    
    # Initialize metrics evaluator if ground truth is provided
    metrics_evaluator = None
    if args.ground_truth:
        metrics_evaluator = OffsideMetricsEvaluator(args.ground_truth)
    
    # Process video
    frame_count = 0
    offside_events = []
    
    # Process first frame to initialize perspective transformer
    success, first_frame = video_processor.read_frame()
    if not success:
        logger.error("Failed to read first frame")
        return
    
    # Initialize perspective transformer with field lines from first frame
    perspective_transformer.initialize(first_frame)
    
    # Reset video to beginning
    video_processor.reset()
    
    logger.info("Starting frame-by-frame processing")
    while True:
        # Read frame
        success, frame = video_processor.read_frame()
        if not success:
            break
            
        frame_count += 1
        if frame_count % 100 == 0:
            logger.info(f"Processing frame {frame_count}")
        
        # Process frame
        try:
            # Detect players
            player_detections = player_detector.detect(frame)
            
            # Classify teams
            if frame_count == 1 or frame_count % 30 == 0:  # Re-calibrate team colors periodically
                team_classifier.calibrate(frame, player_detections)
            
            team_assignments = team_classifier.classify(frame, player_detections)
            
            # Track ball
            ball_position = ball_tracker.track(frame)
            
            # Check if ball was just passed
            ball_passed = ball_tracker.detect_pass()
            
            # Convert to pitch coordinates
            player_pitch_positions = perspective_transformer.transform_players(player_detections, team_assignments)
            ball_pitch_position = perspective_transformer.transform_ball(ball_position) if ball_position else None
            
            # Analyze offside when ball is passed
            offside_result = None
            if ball_passed and ball_pitch_position:
                offside_result = offside_analyzer.analyze(player_pitch_positions, ball_pitch_position)
                if offside_result['is_offside']:
                    offside_events.append({
                        'frame_number': frame_count,
                        'timestamp': video_processor.frame_to_timestamp(frame_count),
                        'offside_player': offside_result['offside_player'],
                        'offside_distance': offside_result['offside_distance']
                    })
                    logger.info(f"Offside detected at frame {frame_count}")
            
            # Visualize results
            if visualizer:
                visualization = visualizer.draw_results(
                    frame.copy(),
                    player_detections,
                    team_assignments,
                    ball_position,
                    perspective_transformer.get_homography_matrix(),
                    offside_result
                )
                
                if args.visualize:
                    cv2.imshow('Offside Detection', visualization)
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('q'):
                        break
                
                if args.save_video and output_video:
                    output_video.write(visualization)
            
            # Evaluate metrics if ground truth is available
            if metrics_evaluator and offside_result:
                metrics_evaluator.update(frame_count, offside_result)
                
        except Exception as e:
            logger.error(f"Error processing frame {frame_count}: {e}")
            continue
    
    # Cleanup
    video_processor.release()
    if args.visualize:
        cv2.destroyAllWindows()
    if output_video:
        output_video.release()
    
    # Save offside events to file
    if offside_events:
        import json
        with open(os.path.join(output_dir, 'offside_events.json'), 'w') as f:
            json.dump(offside_events, f, indent=4)
    
    # Calculate and report metrics
    if metrics_evaluator:
        metrics = metrics_evaluator.compute_metrics()
        logger.info(f"Evaluation metrics: {metrics}")
        with open(os.path.join(output_dir, 'metrics.json'), 'w') as f:
            json.dump(metrics, f, indent=4)
    
    logger.info(f"Processing complete. Results saved to {output_dir}")

if __name__ == "__main__":
    main()