
# Soccer Offside Detection System ⚽

A computer vision system for detecting offside situations in soccer matches using player tracking, ball tracking, and perspective transformation.

## ✨ Features

- 🚀 **Player Detection and Tracking**: Using **YOLOv8** for accurate player localization and movement tracking.
- 🎽 **Team Classification**: Automatic team identification based on **jersey color clustering**.
- 🏐 **Ball Detection and Pass Detection**: Tracking ball movement and identifying key passes.
- 🗺️ **Perspective Transformation**: Mapping detections to 2D pitch coordinates for accurate analysis.
- 📏 **Offside Line Detection and Visualization**: Drawing the dynamic offside line in real-time.
- 📈 **Performance Evaluation**: Compare system predictions against **ground truth annotations**.
- 🖥️ **Real-Time Visualization**: Display player positions, ball tracking, offside lines, and pass events on the match feed.

## ⚙️ Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/Abdullah-shaikh03/VAR-Major-Project.git
   cd VAR-Major-Project
   ```

2. (Optional) Create and activate a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # For Linux/Mac
   venv\Scripts\activate     # For Windows
   ```

3. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## 📂 Project Structure

```
VAR-Major-Project/
│
├── models/           # YOLOv8 models and configuration
├── utils/            # Utility scripts for detection, tracking, transformations
├── datasets/         # Dataset and ground truth annotations
├── results/          # Output visualizations and evaluation metrics
├── main.py           # Main script for running the detection pipeline
├── requirements.txt  # Required Python packages
└── README.md         # Project overview and instructions
```

## 🚀 Usage

Run the main detection pipeline:
```bash
python main.py --video_path path/to/match_video.mp4
```

Arguments:
- `--video_path`: Path to the input soccer match video.
- (Optional) Add flags for real-time visualization, saving output, etc.

## 📊 Evaluation

We evaluate the offside detection system based on:
- Player tracking accuracy
- Pass detection precision/recall
- Offside call correctness compared to ground truth data

Evaluation metrics and visualizations are saved under the `results/` directory.

## 📢 Notes

- The system currently assumes a fixed camera setup for perspective transformation.
- Performance can vary based on video resolution, lighting conditions, and jersey similarities.
- Future work includes handling moving cameras and improving pass intent detection.

## 📜 License

This project is licensed under the [MIT License](LICENSE).
