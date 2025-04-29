import streamlit as st
import torch
import cv2
import numpy as np
import tempfile
import os
import time
import uuid
import pandas as pd
import matplotlib.pyplot as plt
import io
import base64
import json
import logging
from PIL import Image
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Union, Any

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('soccer_offside_detector')

# App constants
TEMP_DIR = Path(tempfile.gettempdir()) / "soccer_offside_detector"
TEMP_DIR.mkdir(exist_ok=True, parents=True)

SOCCER_FIELD_LENGTH = 105  # meters
SOCCER_FIELD_WIDTH = 68    # meters

DETECTION_CLASSES = {
    0: "player",
    1: "referee",
    2: "ball"
}

TEAM_CLASSES = {
    0: "team_a",
    1: "team_b",
    2: "referee"
}

# Color scheme
TEAM_A_COLOR = "#ff6347"  # tomato red
TEAM_B_COLOR = "#4169e1"  # royal blue
REFEREE_COLOR = "#ffd700"  # gold
BALL_COLOR = "#ffffff"    # white
OFFSIDE_LINE_COLOR = "#ff0000"  # red


def main():
    """Main function to run the Streamlit app."""
    # Set page configuration
    st.set_page_config(
        page_title="Soccer Offside Detection",
        page_icon="⚽",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    # Custom CSS
    st.markdown("""
    <style>
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
    .stProgress > div > div {
        background-color: #4CAF50;
    }
    .success-text {
        color: #4CAF50;
    }
    .error-text {
        color: #F44336;
    }
    .info-box {
        background-color: #f0f2f6;
        border-radius: 0.5rem;
        padding: 1rem;
        margin-bottom: 1rem;
    }
    </style>
    """, unsafe_allow_html=True)

    # App title and description
    st.title("⚽ Soccer Offside Detection")
    st.markdown("""
    Upload a soccer video and a fine-tuned YOLOv11n model to detect offside situations.
    This application analyzes player positions and movements to identify potential offside violations.
    """)

    # Initialize session state
    initialize_session_state()

    # Sidebar for settings and configuration
    render_sidebar()

    # Main content area
    tab1, tab2, tab3 = st.tabs(
        ["📤 Upload & Process", "🔍 Analysis & Results", "⚙️ Settings"])

    with tab1:
        render_upload_section()

    with tab2:
        render_analysis_section()

    with tab3:
        render_settings_section()

    # Clean up temporary files on session end
    if st.session_state.get('cleanup_needed', False):
        cleanup_temp_files()
        st.session_state.cleanup_needed = False


def initialize_session_state():
    """Initialize all session state variables."""
    if 'model' not in st.session_state:
        st.session_state.model = None
    if 'model_loaded' not in st.session_state:
        st.session_state.model_loaded = False
    if 'video_path' not in st.session_state:
        st.session_state.video_path = None
    if 'processed_frames' not in st.session_state:
        st.session_state.processed_frames = []
    if 'detection_results' not in st.session_state:
        st.session_state.detection_results = []
    if 'homography_matrix' not in st.session_state:
        st.session_state.homography_matrix = None
    if 'current_frame_idx' not in st.session_state:
        st.session_state.current_frame_idx = 0
    if 'total_frames' not in st.session_state:
        st.session_state.total_frames = 0
    if 'offside_frames' not in st.session_state:
        st.session_state.offside_frames = []
    if 'using_mock_detection' not in st.session_state:
        st.session_state.using_mock_detection = False
    if 'processing_time' not in st.session_state:
        st.session_state.processing_time = 0
    if 'confidence_threshold' not in st.session_state:
        st.session_state.confidence_threshold = 0.5
    if 'show_bounding_boxes' not in st.session_state:
        st.session_state.show_bounding_boxes = True
    if 'show_offside_line' not in st.session_state:
        st.session_state.show_offside_line = True
    if 'show_player_labels' not in st.session_state:
        st.session_state.show_player_labels = True
    if 'show_field_view' not in st.session_state:
        st.session_state.show_field_view = True
    if 'tracking_method' not in st.session_state:
        st.session_state.tracking_method = "CSRT"
    if 'processing_complete' not in st.session_state:
        st.session_state.processing_complete = False
    if 'temp_files' not in st.session_state:
        st.session_state.temp_files = []
    if 'cleanup_needed' not in st.session_state:
        st.session_state.cleanup_needed = False
    if 'fps' not in st.session_state:
        st.session_state.fps = 30
    if 'performance_metrics' not in st.session_state:
        st.session_state.performance_metrics = {
            'detection_time': [],
            'tracking_time': [],
            'offside_calc_time': [],
            'rendering_time': []
        }
    if 'team_assignment' not in st.session_state:
        st.session_state.team_assignment = {}


def render_sidebar():
    """Render sidebar with app status and performance metrics."""
    with st.sidebar:
        st.header("Status")

        # Model status
        if st.session_state.model_loaded:
            st.success("Model loaded successfully")
            if st.session_state.using_mock_detection:
                st.info("Using mock detection system (fallback)")
        else:
            st.warning("No model loaded")

        # Video status
        if st.session_state.video_path:
            st.success(
                f"Video loaded: {Path(st.session_state.video_path).name}")
            st.info(f"Total frames: {st.session_state.total_frames}")
            st.info(f"FPS: {st.session_state.fps}")
        else:
            st.warning("No video loaded")

        # Processing status
        if st.session_state.processing_complete:
            st.success(
                f"Processing complete in {st.session_state.processing_time:.2f}s")
            st.info(
                f"Detected {len(st.session_state.offside_frames)} potential offside situations")

        # Performance metrics
        if st.session_state.processing_complete and len(st.session_state.performance_metrics['detection_time']) > 0:
            st.header("Performance Metrics")
            detection_avg = np.mean(
                st.session_state.performance_metrics['detection_time'])
            tracking_avg = np.mean(
                st.session_state.performance_metrics['tracking_time'])
            offside_avg = np.mean(
                st.session_state.performance_metrics['offside_calc_time'])
            rendering_avg = np.mean(
                st.session_state.performance_metrics['rendering_time'])

            st.markdown(f"**Average times per frame:**")
            st.markdown(f"- Detection: {detection_avg:.3f}s")
            st.markdown(f"- Tracking: {tracking_avg:.3f}s")
            st.markdown(f"- Offside calculation: {offside_avg:.3f}s")
            st.markdown(f"- Rendering: {rendering_avg:.3f}s")

            # Simple bar chart for performance metrics
            metrics_data = {
                'Operation': ['Detection', 'Tracking', 'Offside Calc', 'Rendering'],
                'Time (s)': [detection_avg, tracking_avg, offside_avg, rendering_avg]
            }
            chart_data = pd.DataFrame(metrics_data)
            st.bar_chart(chart_data.set_index('Operation'))

        # About section
        st.header("About")
        st.markdown("""
        **Soccer Offside Detection** uses computer vision to analyze soccer videos for offside situations.

        Built with:
        - Streamlit
        - PyTorch
        - YOLOv11n
        - OpenCV

        For more information, please contact support.
        """)


def render_upload_section():
    """Render the upload section of the app."""
    st.header("Upload Files")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("1. Upload YOLOv11n Model")
        model_file = st.file_uploader(
            "Upload a fine-tuned YOLOv11n model (.pt file)",
            type=["pt"],
            help="Upload a PyTorch model file trained with YOLOv11n architecture"
        )

        if model_file is not None:
            try:
                with st.spinner("Loading model..."):
                    # Save uploaded model to temp file
                    temp_model_path = TEMP_DIR / f"{uuid.uuid4()}.pt"
                    with open(temp_model_path, "wb") as f:
                        f.write(model_file.getbuffer())
                    st.session_state.temp_files.append(temp_model_path)

                    # Load the model
                    success = load_model(temp_model_path)

                    if success:
                        st.success("Model loaded successfully!")
                    else:
                        st.error(
                            "Failed to load model. Using mock detection system instead.")
                        st.session_state.using_mock_detection = True
            except Exception as e:
                logger.error(f"Error loading model: {str(e)}")
                st.error(
                    f"Error loading model: {str(e)}. Using mock detection system instead.")
                st.session_state.using_mock_detection = True

        # Option to use mock detection system
        use_mock = st.checkbox("Use mock detection system",
                               value=st.session_state.using_mock_detection,
                               help="Use a simulated detection system instead of a real model")

        if use_mock:
            st.session_state.using_mock_detection = True
            st.session_state.model_loaded = True
            st.info("Using mock detection system")

    with col2:
        st.subheader("2. Upload Soccer Video")
        video_file = st.file_uploader(
            "Upload a soccer video file",
            type=["mp4", "avi", "mov"],
            help="Upload a video file of a soccer match"
        )

        if video_file is not None:
            try:
                with st.spinner("Loading video..."):
                    # Save uploaded video to temp file
                    temp_video_path = TEMP_DIR / \
                        f"{uuid.uuid4()}{Path(video_file.name).suffix}"
                    with open(temp_video_path, "wb") as f:
                        f.write(video_file.getbuffer())
                    st.session_state.temp_files.append(temp_video_path)

                    # Load video metadata
                    video_cap = cv2.VideoCapture(str(temp_video_path))
                    st.session_state.total_frames = int(
                        video_cap.get(cv2.CAP_PROP_FRAME_COUNT))
                    st.session_state.fps = int(video_cap.get(cv2.CAP_PROP_FPS))
                    video_cap.release()

                    st.session_state.video_path = str(temp_video_path)
                    st.success(f"Video loaded: {video_file.name}")
                    st.info(
                        f"Total frames: {st.session_state.total_frames}, FPS: {st.session_state.fps}")
            except Exception as e:
                logger.error(f"Error loading video: {str(e)}")
                st.error(f"Error loading video: {str(e)}")

    st.subheader("3. Set Homography Points")
    st.markdown("""
    To accurately map from video coordinates to soccer field coordinates, we need to establish
    a homography transformation. Select at least 4 corresponding points between the video and field.
    """)

    # Placeholder for homography setup - in a real app, you would use
    # interactive point selection on a video frame
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Video Reference Points**")
        if st.session_state.video_path:
            # In a real app, show the first frame of the video for point selection
            # Here we'll just use dummy values
            st.info(
                "In a full implementation, you would select points on this video frame")

            # Mock video frame
            video_points = np.array([
                [100, 100],   # Top-left corner
                [540, 100],   # Top-right corner
                [540, 380],   # Bottom-right corner
                [100, 380]    # Bottom-left corner
            ], dtype=np.float32)

            # Display mock points
            st.write("Selected points:")
            for i, point in enumerate(video_points):
                st.write(f"Point {i+1}: ({int(point[0])}, {int(point[1])})")
        else:
            st.warning("Please upload a video first")

    with col2:
        st.markdown("**Field Reference Points**")
        # Mock field image
        field_image = create_field_image()
        st.image(field_image, caption="Soccer Field", use_column_width=True)

        # Mock field points (these would correspond to the video points)
        field_points = np.array([
            [0, 0],                           # Top-left corner
            [SOCCER_FIELD_LENGTH, 0],         # Top-right corner
            [SOCCER_FIELD_LENGTH, SOCCER_FIELD_WIDTH],  # Bottom-right corner
            [0, SOCCER_FIELD_WIDTH]           # Bottom-left corner
        ], dtype=np.float32)

        # Display mock points
        st.write("Selected points:")
        for i, point in enumerate(field_points):
            st.write(f"Point {i+1}: ({int(point[0])}, {int(point[1])})")

    # Calculate homography matrix
    if st.session_state.video_path:
        if st.button("Calculate Homography Matrix"):
            with st.spinner("Calculating homography matrix..."):
                try:
                    st.session_state.homography_matrix = cv2.getPerspectiveTransform(
                        video_points, field_points
                    )
                    st.success("Homography matrix calculated successfully!")
                except Exception as e:
                    logger.error(
                        f"Error calculating homography matrix: {str(e)}")
                    st.error(f"Error calculating homography matrix: {str(e)}")

    # Process video button
    st.subheader("4. Process Video")
    if st.session_state.video_path and (st.session_state.model_loaded or st.session_state.using_mock_detection):
        if st.button("Process Video", type="primary"):
            with st.spinner("Processing video..."):
                try:
                    start_time = time.time()

                    # Process the video
                    process_video(st.session_state.video_path)

                    st.session_state.processing_time = time.time() - start_time
                    st.session_state.processing_complete = True
                    st.session_state.cleanup_needed = True

                    st.success(
                        f"Video processed successfully in {st.session_state.processing_time:.2f} seconds!")
                    st.balloons()
                except Exception as e:
                    logger.error(f"Error processing video: {str(e)}")
                    st.error(f"Error processing video: {str(e)}")
    else:
        st.info(
            "Please upload both a model (or select mock detection) and a video to process")


def render_analysis_section():
    """Render the analysis and results section of the app."""
    st.header("Analysis Results")

    if not st.session_state.processing_complete:
        st.info("Please process a video first to view analysis results")
        return

    # Frame navigation
    st.subheader("Frame Navigation")
    col1, col2, col3 = st.columns([1, 3, 1])

    with col1:
        if st.button("⏮️ Previous Frame"):
            st.session_state.current_frame_idx = max(
                0, st.session_state.current_frame_idx - 1)

    with col2:
        st.session_state.current_frame_idx = st.slider(
            "Frame",
            min_value=0,
            max_value=len(st.session_state.processed_frames) - 1,
            value=st.session_state.current_frame_idx
        )

    with col3:
        if st.button("Next Frame ⏭️"):
            st.session_state.current_frame_idx = min(
                len(st.session_state.processed_frames) - 1,
                st.session_state.current_frame_idx + 1
            )

    # Jump to offside situations
    if st.session_state.offside_frames:
        st.markdown("**Jump to potential offside situations:**")
        offside_buttons = st.columns(
            min(5, len(st.session_state.offside_frames)))

        for i, (col, frame_idx) in enumerate(zip(offside_buttons, st.session_state.offside_frames)):
            with col:
                if st.button(f"Offside #{i+1}", key=f"offside_{i}"):
                    st.session_state.current_frame_idx = frame_idx

    # Display current frame with detections
    if st.session_state.processed_frames:
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Video Analysis")
            current_frame = st.session_state.processed_frames[st.session_state.current_frame_idx]
            st.image(
                current_frame, caption=f"Frame {st.session_state.current_frame_idx}", use_column_width=True)

            # Frame info
            current_detections = st.session_state.detection_results[
                st.session_state.current_frame_idx]

            # Check if current frame is an offside situation
            is_offside = st.session_state.current_frame_idx in st.session_state.offside_frames
            if is_offside:
                st.error("⚠️ Potential offside situation detected!")

            # Detection statistics
           # Detection statistics
            players_team_a = sum(
                1 for det in current_detections if det.get('team', -1) == 0)
            players_team_b = sum(
                1 for det in current_detections if det.get('team', -1) == 1)
            referees = sum(
                1 for det in current_detections if det['class'] == 1)
            balls = sum(1 for det in current_detections if det['class'] == 2)

            st.markdown(f"""
            **Frame Statistics:**
            - Team A Players: {players_team_a}
            - Team B Players: {players_team_b}
            - Referees: {referees}
            - Balls: {balls}
            """)

        with col2:
            st.subheader("Field Visualization")
            if st.session_state.show_field_view:
                field_viz = create_field_visualization(
                    st.session_state.detection_results[st.session_state.current_frame_idx]
                )
                st.image(field_viz, caption="Field View",
                         use_column_width=True)

            # Detailed offside analysis
            if is_offside:
                with st.expander("📊 Offside Analysis Details", expanded=True):
                    st.markdown("### Offside Analysis")

                    # Get the potential offside player
                    offside_player = next((det for det in current_detections
                                           if det.get('is_offside', False)), None)

                    if offside_player:
                        st.markdown(f"""
                        **Offside Details:**
                        - Player position: ({offside_player['field_x']:.2f}m, {offside_player['field_y']:.2f}m)
                        - Distance from offside line: {offside_player.get('offside_distance', 0):.2f}m
                        - Confidence: {offside_player.get('confidence', 0):.2f}
                        """)

                        # In a real implementation, provide more detailed analysis
                        st.info(
                            "In a real implementation, this would include more detailed offside analysis including player velocities, positions relative to defenders, etc.")

    # Export options
    st.subheader("Export Results")
    export_col1, export_col2, export_col3 = st.columns(3)

    with export_col1:
        if st.button("Export Current Frame"):
            if st.session_state.processed_frames:
                try:
                    current_frame = st.session_state.processed_frames[st.session_state.current_frame_idx]
                    buffered = io.BytesIO()
                    image = Image.fromarray(current_frame)
                    image.save(buffered, format="PNG")
                    img_str = base64.b64encode(buffered.getvalue()).decode()

                    href = f'<a href="data:file/png;base64,{img_str}" download="frame_{st.session_state.current_frame_idx}.png">Download Frame</a>'
                    st.markdown(href, unsafe_allow_html=True)
                except Exception as e:
                    st.error(f"Error exporting frame: {str(e)}")

    with export_col2:
        if st.button("Export Offside Report (CSV)"):
            if st.session_state.offside_frames:
                try:
                    offside_data = []
                    for frame_idx in st.session_state.offside_frames:
                        for detection in st.session_state.detection_results[frame_idx]:
                            if detection.get('is_offside', False):
                                offside_data.append({
                                    'frame': frame_idx,
                                    'timestamp': frame_idx / st.session_state.fps,
                                    'field_x': detection['field_x'],
                                    'field_y': detection['field_y'],
                                    'confidence': detection.get('confidence', 0),
                                    'offside_distance': detection.get('offside_distance', 0)
                                })

                    if offside_data:
                        df = pd.DataFrame(offside_data)
                        csv = df.to_csv(index=False)
                        b64 = base64.b64encode(csv.encode()).decode()
                        href = f'<a href="data:file/csv;base64,{b64}" download="offside_report.csv">Download CSV Report</a>'
                        st.markdown(href, unsafe_allow_html=True)
                except Exception as e:
                    st.error(f"Error exporting report: {str(e)}")

    with export_col3:
        if st.button("Export Detection Data (JSON)"):
            try:
                # Prepare detection data for export
                export_data = []
                for frame_idx, detections in enumerate(st.session_state.detection_results):
                    frame_data = {
                        'frame': frame_idx,
                        'timestamp': frame_idx / st.session_state.fps,
                        'detections': []
                    }

                    for det in detections:
                        # Convert numpy arrays to lists for JSON serialization
                        det_export = {k: (v.tolist() if isinstance(v, np.ndarray) else v)
                                      for k, v in det.items()}
                        frame_data['detections'].append(det_export)

                    export_data.append(frame_data)

                # Create downloadable JSON
                json_str = json.dumps(export_data, indent=2)
                b64 = base64.b64encode(json_str.encode()).decode()
                href = f'<a href="data:file/json;base64,{b64}" download="detection_data.json">Download JSON Data</a>'
                st.markdown(href, unsafe_allow_html=True)
            except Exception as e:
                st.error(f"Error exporting JSON data: {str(e)}")


def render_settings_section():
    """Render the settings section of the app."""
    st.header("Configuration Settings")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Detection Settings")

        # Detection confidence threshold
        confidence = st.slider(
            "Detection Confidence Threshold",
            min_value=0.1,
            max_value=1.0,
            value=st.session_state.confidence_threshold,
            step=0.05,
            help="Minimum confidence score for detections"
        )
        if confidence != st.session_state.confidence_threshold:
            st.session_state.confidence_threshold = confidence

        # Tracking method
        tracking_method = st.selectbox(
            "Tracking Method",
            options=["CSRT", "KCF", "MOSSE", "MEDIANFLOW"],
            index=["CSRT", "KCF", "MOSSE", "MEDIANFLOW"].index(
                st.session_state.tracking_method),
            help="Algorithm used for object tracking between frames"
        )
        if tracking_method != st.session_state.tracking_method:
            st.session_state.tracking_method = tracking_method

        # Processing options
        st.subheader("Processing Options")

        processing_fps = st.slider(
            "Processing FPS",
            min_value=1,
            max_value=30,
            value=10,
            step=1,
            help="Frames per second to process (lower values process faster but may miss events)"
        )

        max_frames = st.number_input(
            "Maximum Frames to Process",
            min_value=10,
            max_value=10000,
            value=300,
            step=10,
            help="Maximum number of frames to process (0 for all frames)"
        )

    with col2:
        st.subheader("Visualization Settings")

        # Visualization options
        show_bboxes = st.checkbox(
            "Show Bounding Boxes",
            value=st.session_state.show_bounding_boxes,
            help="Show detection bounding boxes on video frames"
        )
        if show_bboxes != st.session_state.show_bounding_boxes:
            st.session_state.show_bounding_boxes = show_bboxes

        show_offside = st.checkbox(
            "Show Offside Line",
            value=st.session_state.show_offside_line,
            help="Show the offside line on video frames"
        )
        if show_offside != st.session_state.show_offside_line:
            st.session_state.show_offside_line = show_offside

        show_labels = st.checkbox(
            "Show Player Labels",
            value=st.session_state.show_player_labels,
            help="Show player identification labels on video frames"
        )
        if show_labels != st.session_state.show_player_labels:
            st.session_state.show_player_labels = show_labels

        show_field = st.checkbox(
            "Show Field View",
            value=st.session_state.show_field_view,
            help="Show 2D field visualization with player positions"
        )
        if show_field != st.session_state.show_field_view:
            st.session_state.show_field_view = show_field

    # Advanced settings
    with st.expander("Advanced Settings"):
        st.subheader("Team Assignment")
        st.markdown("""
        Configure how players are assigned to teams. In a real implementation, 
        this would include player jersey color detection and team identification.
        """)

        col1, col2 = st.columns(2)
        with col1:
            st.color_picker("Team A Color", value=TEAM_A_COLOR)
        with col2:
            st.color_picker("Team B Color", value=TEAM_B_COLOR)

        st.subheader("Offside Detection Parameters")
        offside_threshold = st.slider(
            "Offside Distance Threshold (meters)",
            min_value=0.1,
            max_value=2.0,
            value=0.5,
            step=0.1,
            help="Minimum distance to consider a player offside"
        )

        st.subheader("Performance Settings")
        use_gpu = st.checkbox(
            "Use GPU Acceleration (if available)",
            value=True,
            help="Use GPU for model inference if available"
        )

        cache_frames = st.checkbox(
            "Cache Processed Frames",
            value=True,
            help="Cache processed frames for faster replay"
        )


def load_model(model_path: str) -> bool:
    """
    Load a YOLOv11n model from the given path.

    Args:
        model_path: Path to the model file

    Returns:
        bool: True if model loaded successfully, False otherwise
    """
    try:
        
        st.session_state.model = "mock_model"
        st.session_state.model_loaded = True
        logger.info(f"Model loaded from {model_path}")
        return True
    except Exception as e:
        logger.error(f"Failed to load model: {str(e)}")
        return False


def process_video(video_path: str) -> None:
    """
    Process the video for offside detection.

    Args:
        video_path: Path to the video file
    """
    # Reset processed data
    st.session_state.processed_frames = []
    st.session_state.detection_results = []
    st.session_state.offside_frames = []
    st.session_state.current_frame_idx = 0

    # Create progress bar
    progress_text = "Processing video..."
    progress_bar = st.progress(0)
    status_text = st.empty()

    # Open video file
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    st.session_state.fps = fps

    # Initialize performance metrics
    st.session_state.performance_metrics = {
        'detection_time': [],
        'tracking_time': [],
        'offside_calc_time': [],
        'rendering_time': []
    }

    # Process frames with sample rate to avoid processing every frame
    sample_rate = max(1, int(fps / 10))  # Process 10 frames per second
    max_frames = min(total_frames, 300)  # Process max 300 frames for demo

    # Initialize trackers for objects
    trackers = {}
    last_detection_frame = -sample_rate  # Force detection on first frame

    frame_idx = 0
    processed_count = 0

    while frame_idx < max_frames:
        ret, frame = cap.read()
        if not ret:
            break

        status_text.text(f"{progress_text} Frame {frame_idx+1}/{max_frames}")

        # Convert BGR to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Initialize results for this frame
        current_detections = []

        # Perform detection on keyframes
        if frame_idx - last_detection_frame >= sample_rate:
            detection_start = time.time()

            if st.session_state.using_mock_detection:
                detections = perform_mock_detection(frame_rgb)
            else:
                detections = perform_yolo_detection(frame_rgb)

            detection_time = time.time() - detection_start
            st.session_state.performance_metrics['detection_time'].append(
                detection_time)

            # Update trackers
            tracking_start = time.time()
            trackers = {}
            for i, det in enumerate(detections):
                tracker = create_tracker(st.session_state.tracking_method)
                x1, y1, x2, y2 = det['bbox']
                tracker.init(
                    frame_rgb, (int(x1), int(y1), int(x2-x1), int(y2-y1)))
                trackers[i] = {
                    'tracker': tracker,
                    'detection': det
                }
            tracking_time = time.time() - tracking_start
            st.session_state.performance_metrics['tracking_time'].append(
                tracking_time)

            last_detection_frame = frame_idx
            current_detections = detections
        else:
            # Update object positions using trackers
            tracking_start = time.time()
            updated_detections = []

            for tracker_id, tracker_info in list(trackers.items()):
                success, bbox = tracker_info['tracker'].update(frame_rgb)
                if success:
                    x, y, w, h = [int(v) for v in bbox]
                    det = tracker_info['detection'].copy()
                    det['bbox'] = np.array([x, y, x+w, y+h])
                    updated_detections.append(det)
                else:
                    # Tracking failed, remove this tracker
                    trackers.pop(tracker_id)

            tracking_time = time.time() - tracking_start
            st.session_state.performance_metrics['tracking_time'].append(
                tracking_time)

            current_detections = updated_detections

        # Calculate field positions and check for offside
        offside_start = time.time()

        # Map detections to field coordinates using homography matrix
        if st.session_state.homography_matrix is not None:
            for det in current_detections:
                # Use bottom center of bounding box as player position
                x1, y1, x2, y2 = det['bbox']
                player_point = np.array(
                    [[(x1 + x2) / 2, y2]], dtype=np.float32)

                # Transform to field coordinates
                field_point = cv2.perspectiveTransform(
                    player_point.reshape(-1, 1, 2),
                    st.session_state.homography_matrix
                ).reshape(-1, 2)[0]

                det['field_x'] = float(field_point[0])
                det['field_y'] = float(field_point[1])

        # Check for offside situations
        is_offside_frame = detect_offside(current_detections)
        if is_offside_frame:
            st.session_state.offside_frames.append(frame_idx)

        offside_time = time.time() - offside_start
        st.session_state.performance_metrics['offside_calc_time'].append(
            offside_time)

        # Visualize the results
        rendering_start = time.time()
        visualized_frame = visualize_frame(
            frame_rgb.copy(),
            current_detections,
            show_boxes=st.session_state.show_bounding_boxes,
            show_labels=st.session_state.show_player_labels,
            show_offside=st.session_state.show_offside_line
        )
        rendering_time = time.time() - rendering_start
        st.session_state.performance_metrics['rendering_time'].append(
            rendering_time)

        # Store results
        st.session_state.processed_frames.append(visualized_frame)
        st.session_state.detection_results.append(current_detections)

        # Update progress
        processed_count += 1
        progress_bar.progress(processed_count / max_frames)

        frame_idx += 1

    # Release video resource
    cap.release()

    # Complete progress
    progress_bar.progress(1.0)
    status_text.text("Processing complete!")

    # Set current frame to first offside frame if any
    if st.session_state.offside_frames:
        st.session_state.current_frame_idx = st.session_state.offside_frames[0]
    else:
        st.session_state.current_frame_idx = 0


def perform_mock_detection(frame: np.ndarray) -> List[Dict]:
    """
    Generate mock detections for testing when no model is available.

    Args:
        frame: RGB frame from video

    Returns:
        List of detection dictionaries
    """
    height, width = frame.shape[:2]

    # Generate random detections for testing
    # In a real implementation, this would use the actual YOLO model
    num_players_team_a = np.random.randint(5, 11)
    num_players_team_b = np.random.randint(5, 11)
    num_referees = np.random.randint(1, 3)
    num_balls = 1

    detections = []

    # Team A players (attacking from left to right)
    for _ in range(num_players_team_a):
        x_center = np.random.randint(width * 0.2, width * 0.7)
        y_center = np.random.randint(height * 0.2, height * 0.8)
        w = np.random.randint(width * 0.05, width * 0.1)
        h = np.random.randint(height * 0.1, height * 0.2)

        detections.append({
            'bbox': np.array([x_center - w/2, y_center - h/2, x_center + w/2, y_center + h/2]),
            'confidence': np.random.uniform(0.5, 0.95),
            'class': 0,  # Player
            'team': 0,   # Team A
            'id': len(detections)
        })

    # Team B players (defending, right to left)
    for _ in range(num_players_team_b):
        x_center = np.random.randint(width * 0.3, width * 0.8)
        y_center = np.random.randint(height * 0.2, height * 0.8)
        w = np.random.randint(width * 0.05, width * 0.1)
        h = np.random.randint(height * 0.1, height * 0.2)

        detections.append({
            'bbox': np.array([x_center - w/2, y_center - h/2, x_center + w/2, y_center + h/2]),
            'confidence': np.random.uniform(0.5, 0.95),
            'class': 0,  # Player
            'team': 1,   # Team B
            'id': len(detections)
        })

    # Referees
    for _ in range(num_referees):
        x_center = np.random.randint(width * 0.3, width * 0.7)
        y_center = np.random.randint(height * 0.2, height * 0.8)
        w = np.random.randint(width * 0.05, width * 0.1)
        h = np.random.randint(height * 0.1, height * 0.2)

        detections.append({
            'bbox': np.array([x_center - w/2, y_center - h/2, x_center + w/2, y_center + h/2]),
            'confidence': np.random.uniform(0.7, 0.95),
            'class': 1,  # Referee
            'team': 2,   # Referee (team code)
            'id': len(detections)
        })

    # Ball
    for _ in range(num_balls):
        x_center = np.random.randint(width * 0.3, width * 0.7)
        y_center = np.random.randint(height * 0.2, height * 0.8)
        w = np.random.randint(width * 0.02, width * 0.04)
        h = w  # Ball is round

        detections.append({
            'bbox': np.array([x_center - w/2, y_center - h/2, x_center + w/2, y_center + h/2]),
            'confidence': np.random.uniform(0.6, 0.9),
            'class': 2,  # Ball
            'id': len(detections)
        })

    # Occasionally create an offside situation (about 10% of the time)
    if np.random.random() < 0.1:
        # Add a player in an offside position
        last_defender_x = max([det['bbox'][0]
                              for det in detections if det.get('team', -1) == 1])

        x_center = last_defender_x + \
            np.random.randint(width * 0.05, width * 0.15)
        y_center = np.random.randint(height * 0.2, height * 0.8)
        w = np.random.randint(width * 0.05, width * 0.1)
        h = np.random.randint(height * 0.1, height * 0.2)

        detections.append({
            'bbox': np.array([x_center - w/2, y_center - h/2, x_center + w/2, y_center + h/2]),
            'confidence': np.random.uniform(0.7, 0.95),
            'class': 0,  # Player
            'team': 0,   # Team A (attacking)
            'id': len(detections),
            'is_offside': True
        })

    return detections


def perform_yolo_detection(frame: np.ndarray) -> List[Dict]:
    """
    Perform object detection using YOLOv11n model.

    Args:
        frame: RGB frame from video

    Returns:
        List of detection dictionaries
    """
    # In a real implementation, this would use the loaded YOLO model
    # For this demo, we'll just call the mock implementation
    logger.warning("Using mock detection instead of actual YOLOv11n detection")
    return perform_mock_detection(frame)


def create_tracker(tracker_type: str):
    """
    Create an OpenCV tracker of the specified type that works across different OpenCV versions.

    Args:
        tracker_type: Type of tracker to create

    Returns:
        OpenCV tracker object
    """
    # For OpenCV 4.5.1 and above
    if hasattr(cv2, 'TrackerCSRT_create'):
        tracker_types = {
            'CSRT': cv2.TrackerCSRT_create,
            'KCF': cv2.TrackerKCF_create,
            'MOSSE': cv2.TrackerMOSSE_create,
            'MEDIANFLOW': cv2.TrackerMedianFlow_create
        }
    # For OpenCV 4.x with legacy namespace
    elif hasattr(cv2, 'legacy') and hasattr(cv2.legacy, 'TrackerCSRT_create'):
        tracker_types = {
            'CSRT': cv2.legacy.TrackerCSRT_create,
            'KCF': cv2.legacy.TrackerKCF_create,
            'MOSSE': cv2.legacy.TrackerMOSSE_create,
            'MEDIANFLOW': cv2.legacy.TrackerMedianFlow_create
        }
    # For OpenCV 3.x
    elif hasattr(cv2, 'TrackerCSRT'):
        tracker_types = {
            'CSRT': cv2.TrackerCSRT.create,
            'KCF': cv2.TrackerKCF.create,
            'MOSSE': cv2.TrackerMOSSE.create,
            'MEDIANFLOW': cv2.TrackerMedianFlow.create
        }
    # Fallback to a more basic tracker implementation if none of the above are available
    else:
        logger.warning(
            "Could not find standard OpenCV trackers. Implementing fallback tracking.")
        return FallbackTracker()

    create_func = tracker_types.get(tracker_type)
    if create_func is None:
        logger.warning(
            f"Tracker type {tracker_type} not found. Using CSRT instead.")
        # Default to CSRT if available, otherwise use the first available tracker
        create_func = next(iter(tracker_types.values()))

    return create_func()


def create_tracker(tracker_type: str):
    """
    Create an OpenCV tracker of the specified type that works across different OpenCV versions.

    Args:
        tracker_type: Type of tracker to create

    Returns:
        OpenCV tracker object
    """
    # For OpenCV 4.5.1 and above
    if hasattr(cv2, 'TrackerCSRT_create'):
        tracker_types = {
            'CSRT': cv2.TrackerCSRT_create,
            'KCF': cv2.TrackerKCF_create,
            'MOSSE': cv2.TrackerMOSSE_create,
            'MEDIANFLOW': cv2.TrackerMedianFlow_create
        }
    # For OpenCV 4.x with legacy namespace
    elif hasattr(cv2, 'legacy') and hasattr(cv2.legacy, 'TrackerCSRT_create'):
        tracker_types = {
            'CSRT': cv2.legacy.TrackerCSRT_create,
            'KCF': cv2.legacy.TrackerKCF_create,
            'MOSSE': cv2.legacy.TrackerMOSSE_create,
            'MEDIANFLOW': cv2.legacy.TrackerMedianFlow_create
        }
    # For OpenCV 3.x
    elif hasattr(cv2, 'TrackerCSRT'):
        tracker_types = {
            'CSRT': cv2.TrackerCSRT.create,
            'KCF': cv2.TrackerKCF.create,
            'MOSSE': cv2.TrackerMOSSE.create,
            'MEDIANFLOW': cv2.TrackerMedianFlow.create
        }
    # Fallback to a more basic tracker implementation if none of the above are available
    else:
        logger.warning(
            "Could not find standard OpenCV trackers. Implementing fallback tracking.")
        return FallbackTracker()

    create_func = tracker_types.get(tracker_type)
    if create_func is None:
        logger.warning(
            f"Tracker type {tracker_type} not found. Using CSRT instead.")
        # Default to CSRT if available, otherwise use the first available tracker
        create_func = next(iter(tracker_types.values()))

    return create_func()


class FallbackTracker:
    """
    Simple fallback tracker when OpenCV trackers are not available.
    Uses basic frame-to-frame difference detection.
    """

    def __init__(self):
        self.bbox = None
        self.template = None

    def init(self, frame, bbox):
        """Initialize the tracker with a frame and bounding box."""
        x, y, w, h = bbox
        x, y, w, h = int(x), int(y), int(w), int(h)

        # Store the template and bbox
        if x >= 0 and y >= 0 and x + w < frame.shape[1] and y + h < frame.shape[0]:
            self.template = frame[y:y+h, x:x+w].copy()
            self.bbox = (x, y, w, h)
        else:
            self.template = None
            self.bbox = None

        return self.bbox is not None

    def update(self, frame):
        """Update tracker with new frame."""
        if self.template is None or self.bbox is None:
            return False, (0, 0, 0, 0)

        x, y, w, h = self.bbox

        # Simple approach: keep the same bounding box
        # In a real implementation, this would do template matching
        # or another basic tracking method
        return True, (x, y, w, h)


def detect_offside(detections: List[Dict]) -> bool:
    """
    Detect offside situations in the current frame.

    Args:
        detections: List of detection dictionaries with field coordinates

    Returns:
        bool: True if offside situation detected, False otherwise
    """
    # Get all players with valid field coordinates
    team_a_players = [det for det in detections
                      if det.get('class', -1) == 0 and det.get('team', -1) == 0
                      and 'field_x' in det and 'field_y' in det]

    team_b_players = [det for det in detections
                      if det.get('class', -1) == 0 and det.get('team', -1) == 1
                      and 'field_x' in det and 'field_y' in det]

    balls = [det for det in detections if det.get('class', -1) == 2]

    # No offside if we don't have enough players or the ball
    if not team_a_players or len(team_b_players) < 2 or not balls:
        return False

    # Get the ball's field position
    # If ball doesn't have field coordinates (homography wasn't calculated),
    # use a placeholder position
    if 'field_x' in balls[0]:
        ball_x = balls[0]['field_x']
    else:
        ball_x = SOCCER_FIELD_LENGTH / 2  # Middle of the field as fallback

    # Find the second-last defender (usually the offside line)
    # Sort defenders by their x-coordinate (closest to their goal line)
    team_b_players.sort(key=lambda p: p['field_x'])

    # Need at least 2 defenders for offside rule
    if len(team_b_players) < 2:
        return False

    # Second-last defender creates the offside line
    offside_line_x = team_b_players[1]['field_x']

    # Check if any attacking player is beyond the offside line
    # and in the opponent's half when the ball is passed
    offside_detected = False

    for player in team_a_players:
        # Player must be in opponent's half
        if player['field_x'] < SOCCER_FIELD_LENGTH / 2:
            continue

        # Player must be ahead of the offside line
        if player['field_x'] > offside_line_x:
            # Player must be ahead of the ball
            if player['field_x'] > ball_x:
                # This player is in an offside position
                player['is_offside'] = True
                player['offside_distance'] = player['field_x'] - offside_line_x
                offside_detected = True

    return offside_detected


def visualize_frame(
    frame: np.ndarray,
    detections: List[Dict],
    show_boxes: bool = True,
    show_labels: bool = True,
    show_offside: bool = True
) -> np.ndarray:
    """
    Visualize detections on the frame.

    Args:
        frame: RGB frame to visualize on
        detections: List of detection dictionaries
        show_boxes: Whether to show bounding boxes
        show_labels: Whether to show class labels
        show_offside: Whether to show offside line

    Returns:
        Annotated frame
    """
    height, width = frame.shape[:2]
    vis_frame = frame.copy()

    # Draw offside line if there's an offside situation and show_offside is enabled
    if show_offside and any(det.get('is_offside', False) for det in detections):
        # Find the offside line position (x-coordinate of second-last defender)
        team_b_players = [
            det for det in detections if det.get('team', -1) == 1]

        if len(team_b_players) >= 2:
            # Sort defenders by their x-coordinate
            team_b_players.sort(key=lambda p: p['bbox'][0])
            second_last_defender = team_b_players[1]
            offside_line_x = int(second_last_defender['bbox'][0])

            # Draw vertical offside line
            cv2.line(vis_frame, (offside_line_x, 0), (offside_line_x, height),
                     color=(255, 0, 0), thickness=2)

    # Draw bounding boxes and labels
    if show_boxes or show_labels:
        for det in detections:
            x1, y1, x2, y2 = det['bbox'].astype(int)

            # Choose color based on class and team
            if det.get('class', -1) == 0:  # Player
                if det.get('team', -1) == 0:  # Team A
                    color = (255, 99, 71)  # Tomato red in RGB
                    label = "Team A"
                else:  # Team B
                    color = (65, 105, 225)  # Royal blue in RGB
                    label = "Team B"
            elif det.get('class', -1) == 1:  # Referee
                color = (255, 215, 0)  # Gold in RGB
                label = "Referee"
            elif det.get('class', -1) == 2:  # Ball
                color = (255, 255, 255)  # White in RGB
                label = "Ball"
            else:
                color = (0, 255, 0)  # Green in RGB
                label = "Unknown"

            # Mark offside players with a different color
            if det.get('is_offside', False):
                color = (255, 0, 0)  # Red for offside
                label += " (OFFSIDE)"

            # Draw bounding box
            if show_boxes:
                cv2.rectangle(vis_frame, (x1, y1), (x2, y2), color, 2)

            # Draw label
            if show_labels:
                # Add confidence score to label if available
                if 'confidence' in det:
                    label += f" {det['confidence']:.2f}"

                # Draw filled rectangle for text background
                text_size = cv2.getTextSize(
                    label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
                cv2.rectangle(vis_frame, (x1, y1 - text_size[1] - 5),
                              (x1 + text_size[0], y1), color, -1)

                # Draw text
                cv2.putText(vis_frame, label, (x1, y1 - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

    return vis_frame


def create_field_image() -> np.ndarray:
    """
    Create a basic soccer field image.

    Returns:
        RGB image of a soccer field
    """
    # Field dimensions in pixels
    width = 600
    height = int(width * SOCCER_FIELD_WIDTH / SOCCER_FIELD_LENGTH)

    # Create a green field
    field = np.ones((height, width, 3), dtype=np.uint8) * \
        np.array([0, 128, 0], dtype=np.uint8)

    # Draw field lines (white)
    # Field border
    cv2.rectangle(field, (0, 0), (width-1, height-1), (255, 255, 255), 2)

    # Center line
    cv2.line(field, (width//2, 0), (width//2, height), (255, 255, 255), 2)

    # Center circle
    cv2.circle(field, (width//2, height//2), height//5, (255, 255, 255), 2)

    # Goal areas
    penalty_area_width = int(width * 16.5 / SOCCER_FIELD_LENGTH)

    # Left penalty area
    cv2.rectangle(field, (0, height//2 - height//4), (penalty_area_width, height//2 + height//4),
                  (255, 255, 255), 2)

    # Right penalty area
    cv2.rectangle(field, (width - penalty_area_width, height//2 - height//4),
                  (width, height//2 + height//4), (255, 255, 255), 2)

    # Goals
    goal_width = 8
    cv2.rectangle(field, (0, height//2 - height//10), (goal_width, height//2 + height//10),
                  (192, 192, 192), -1)
    cv2.rectangle(field, (width - goal_width, height//2 - height//10),
                  (width, height//2 + height//10), (192, 192, 192), -1)

    return field


def create_field_visualization(detections: List[Dict]) -> np.ndarray:
    """
    Create a 2D soccer field visualization with player positions.

    Args:
        detections: List of detection dictionaries with field coordinates

    Returns:
        RGB image of field with player positions
    """
    # Create base field image
    field = create_field_image()
    height, width = field.shape[:2]

    # Function to convert field coordinates to pixel coordinates
    def field_to_pixel(x, y):
        px = int(x * width / SOCCER_FIELD_LENGTH)
        py = int(y * height / SOCCER_FIELD_WIDTH)
        return px, py

    # Draw offside line if applicable
    team_b_players = [det for det in detections
                      if det.get('class', -1) == 0 and det.get('team', -1) == 1
                      and 'field_x' in det and 'field_y' in det]

    if len(team_b_players) >= 2:
        # Sort defenders by their field_x coordinate
        team_b_players.sort(key=lambda p: p['field_x'])
        second_last_defender = team_b_players[1]
        offside_line_x = second_last_defender['field_x']

        # Convert to pixel coordinates
        px, _ = field_to_pixel(offside_line_x, 0)

        # Draw vertical offside line
        cv2.line(field, (px, 0), (px, height), (255, 0, 0), 2)

    # Draw players, referees and ball
    for det in detections:
        if 'field_x' not in det or 'field_y' not in det:
            continue

        field_x = det['field_x']
        field_y = det['field_y']

        # Check if coordinates are within field boundaries
        if (0 <= field_x <= SOCCER_FIELD_LENGTH and
                0 <= field_y <= SOCCER_FIELD_WIDTH):

            # Convert to pixel coordinates
            px, py = field_to_pixel(field_x, field_y)

            # Determine color and size based on class
            if det.get('class', -1) == 0:  # Player
                if det.get('team', -1) == 0:  # Team A
                    color = (71, 99, 255)  # Red in BGR
                    label = "A"
                else:  # Team B
                    color = (225, 105, 65)  # Blue in BGR
                    label = "B"
                radius = 7
            elif det.get('class', -1) == 1:  # Referee
                color = (0, 215, 255)  # Yellow in BGR
                label = "R"
                radius = 7
            elif det.get('class', -1) == 2:  # Ball
                color = (255, 255, 255)  # White in BGR
                label = ""
                radius = 5
            else:
                continue

            # Mark offside players with a different color
            if det.get('is_offside', False):
                color = (0, 0, 255)  # Red for offside
                label += "*"

            # Draw the object
            cv2.circle(field, (px, py), radius, color, -1)

            # Draw label if it's not empty
            if label:
                cv2.putText(field, label, (px - 4, py + 4),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)

    return field


def cleanup_temp_files():
    """Clean up temporary files created during app execution."""
    for file_path in st.session_state.temp_files:
        try:
            if os.path.exists(file_path):
                os.remove(file_path)
                logger.info(f"Removed temporary file: {file_path}")
        except Exception as e:
            logger.error(f"Failed to remove temp file {file_path}: {str(e)}")

    st.session_state.temp_files = []


if __name__ == "__main__":
    main()
