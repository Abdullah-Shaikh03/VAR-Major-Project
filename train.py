import os
import yaml
import argparse
import shutil
from ultralytics import YOLO
from pathlib import Path

def parse_args():
    parser = argparse.ArgumentParser(description='Train YOLOv11n for soccer player tracking')
    parser.add_argument('--data_dir', type=str, default='./archive', 
                        help='Path to SoccerNet tracking dataset')
    parser.add_argument('--output_dir', type=str, default='./runs/train',
                        help='Directory to save training results')
    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=16,
                        help='Batch size for training')
    parser.add_argument('--img_size', type=int, default=640,
                        help='Image size for training')
    parser.add_argument('--workers', type=int, default=8,
                        help='Number of worker threads')
    parser.add_argument('--device', type=str, default='0',
                        help='Device to train on (cpu, 0, 0,1,2,3)')
    parser.add_argument('--resume', action='store_true',
                        help='Resume training from last checkpoint')
    return parser.parse_args()

def convert_to_yolo_dataset(data_dir):
    """
    Convert SoccerNet dataset to YOLO format with proper directory structure
    """
    # Create YOLO dataset directories
    yolo_dir = os.path.join(os.path.dirname(data_dir), 'yolo_dataset')
    os.makedirs(os.path.join(yolo_dir, 'images', 'train'), exist_ok=True)
    os.makedirs(os.path.join(yolo_dir, 'images', 'val'), exist_ok=True)
    os.makedirs(os.path.join(yolo_dir, 'labels', 'train'), exist_ok=True)
    os.makedirs(os.path.join(yolo_dir, 'labels', 'val'), exist_ok=True)
    
    train_dir = os.path.join(data_dir, 'train')
    val_dir = os.path.join(data_dir, 'test')  # Using test set as validation
    
    # Process training sequences
    if os.path.exists(train_dir):
        process_sequences(train_dir, yolo_dir, 'train')
    else:
        print(f"Warning: Training directory not found at {train_dir}")
    
    # Process validation sequences
    if os.path.exists(val_dir):
        process_sequences(val_dir, yolo_dir, 'val')
    else:
        print(f"Warning: Validation directory not found at {val_dir}")
    
    # Create dataset configuration
    dataset_config = {
        'path': yolo_dir,
        'train': os.path.join(yolo_dir, 'images', 'train'),
        'val': os.path.join(yolo_dir, 'images', 'val'),
        'nc': 3,
        'names': ['team_left', 'team_right', 'ball']
    }
    
    os.makedirs('./data', exist_ok=True)
    with open('./data/soccernet.yaml', 'w') as f:
        yaml.dump(dataset_config, f)
    
    print(f"YOLO dataset created at {yolo_dir}")
    print(f"Dataset configuration saved to ./data/soccernet.yaml")
    return './data/soccernet.yaml'

def process_sequences(dataset_dir, yolo_dir, split):
    """
    Process sequences for a given split (train/val)
    """
    for sequence_dir in os.listdir(dataset_dir):
        seq_path = os.path.join(dataset_dir, sequence_dir)
        if not os.path.isdir(seq_path):
            continue
        
        img_dir = os.path.join(seq_path, 'img1')
        if not os.path.exists(img_dir):
            print(f"Warning: Image directory not found at {img_dir}")
            continue
        
        # Get sequence info
        seqinfo_path = os.path.join(seq_path, 'seqinfo.ini')
        seq_info = {}
        if os.path.exists(seqinfo_path):
            with open(seqinfo_path, 'r') as f:
                for line in f:
                    if '=' in line:
                        key, value = line.strip().split('=')
                        seq_info[key.strip()] = value.strip()
        
        img_width = int(seq_info.get('imWidth', 1920))
        img_height = int(seq_info.get('imHeight', 1080))
        
        # Process ground truth annotations
        gt_path = os.path.join(seq_path, 'gt', 'gt.txt')
        frame_annotations = {}
        
        if os.path.exists(gt_path):
            with open(gt_path, 'r') as f:
                for line in f:
                    parts = line.strip().split(',')
                    if len(parts) < 6:
                        continue
                    
                    frame_id = int(parts[0])
                    track_id = int(parts[1])
                    
                    # Skip negative tracklet IDs
                    if track_id < 0:
                        continue
                    
                    x, y, w, h = map(float, parts[2:6])
                    
                    # Check if this is a valid bounding box
                    if w <= 0 or h <= 0:
                        continue
                    
                    # Determine class based on team information
                    # This is a placeholder. In a real implementation, use gameinfo.ini
                    # to determine the correct team assignment
                    if len(parts) >= 8:  # Check if class info is available
                        # In SoccerNet format, class info is in column 8
                        class_name = parts[7].lower() if len(parts) > 7 else ""
                        if "team a" in class_name or "left" in class_name:
                            class_id = 0  # team_left
                        elif "team b" in class_name or "right" in class_name:
                            class_id = 1  # team_right
                        elif "ball" in class_name:
                            class_id = 2  # ball
                        else:
                            class_id = 0  # Default to team_left
                    else:
                        # If no class info, use track_id to determine class (simple heuristic)
                        class_id = track_id % 2  # Alternate between team_left and team_right
                    
                    # Convert to YOLO format
                    x_center = (x + w/2) / img_width
                    y_center = (y + h/2) / img_height
                    w_norm = w / img_width
                    h_norm = h / img_height
                    
                    # Ensure values are within 0-1 range
                    x_center = max(0, min(1, x_center))
                    y_center = max(0, min(1, y_center))
                    w_norm = max(0, min(1, w_norm))
                    h_norm = max(0, min(1, h_norm))
                    
                    if frame_id not in frame_annotations:
                        frame_annotations[frame_id] = []
                    
                    frame_annotations[frame_id].append(f"{class_id} {x_center:.6f} {y_center:.6f} {w_norm:.6f} {h_norm:.6f}")
        else:
            print(f"Warning: No annotations found at {gt_path}")
        
        # Check if we have any annotations
        if not frame_annotations:
            print(f"Warning: No valid annotations found for {sequence_dir}")
            # Create at least one dummy annotation to prevent YOLO validation errors
            # This is a workaround - in production you should have real annotations
            frame_annotations[0] = ["0 0.5 0.5 0.1 0.1"]
        
        # Process images and create labels
        img_count = 0
        label_count = 0
        
        img_files = [f for f in os.listdir(img_dir) if f.endswith(('.jpg', '.jpeg', '.png'))]
        if not img_files:
            print(f"Warning: No images found in {img_dir}")
            continue
            
        for img_file in sorted(img_files):
            # Extract frame number from filename
            try:
                frame_id = int(os.path.splitext(img_file)[0])
            except ValueError:
                # If filename is not a number, use the position in the sorted list
                frame_id = img_count
            
            # Copy image to YOLO dataset
            src_img_path = os.path.join(img_dir, img_file)
            dst_img_path = os.path.join(yolo_dir, 'images', split, f"{sequence_dir}_{img_file}")
            shutil.copy2(src_img_path, dst_img_path)
            
            # Create label file
            label_file = os.path.join(yolo_dir, 'labels', split, f"{sequence_dir}_{os.path.splitext(img_file)[0]}.txt")
            
            # If we have annotations for this frame, write them
            if frame_id in frame_annotations:
                with open(label_file, 'w') as f:
                    f.write('\n'.join(frame_annotations[frame_id]))
                label_count += 1
            else:
                # For frames without annotations, use annotations from the closest frame
                # Find closest frame with annotations
                if frame_annotations:
                    closest_frame = min(frame_annotations.keys(), key=lambda x: abs(x - frame_id))
                    with open(label_file, 'w') as f:
                        f.write('\n'.join(frame_annotations[closest_frame]))
                    label_count += 1
                else:
                    # If no annotations at all, create a dummy annotation
                    with open(label_file, 'w') as f:
                        f.write("0 0.5 0.5 0.1 0.1")  # Dummy annotation in the center
                    label_count += 1
            
            img_count += 1
            # Limit number of images per sequence to avoid excessive dataset size
            if img_count >= 100:  # Adjust based on your needs
                break
        
        print(f"Processed {sequence_dir}: {img_count} images, {label_count} labels")

def train(args, dataset_yaml):
    """
    Train YOLOv11n model on the prepared dataset
    """
    # Check if dataset exists and has data
    with open(dataset_yaml, 'r') as f:
        dataset_config = yaml.safe_load(f)
    
    train_dir = dataset_config.get('train', '')
    val_dir = dataset_config.get('val', '')
    
    if not os.path.exists(train_dir) or not os.listdir(train_dir):
        raise RuntimeError(f"Training dataset not found or empty: {train_dir}")
    
    if not os.path.exists(val_dir) or not os.listdir(val_dir):
        print(f"Warning: Validation dataset not found or empty: {val_dir}")
    
    # Initialize model
    model = YOLO('yolo11n.pt')  # Load YOLOv11n pre-trained model
    
    # Train the model
    try:
        results = model.train(
            data=dataset_yaml,
            epochs=args.epochs,
            imgsz=args.img_size,
            batch=args.batch_size,
            workers=args.workers,
            device=args.device,
            project=args.output_dir,
            name='yolov11n_soccer_tracking',
            exist_ok=True,
            resume=args.resume,
            verbose=True,
            patience=50,  # Early stopping patience
            save_period=10  # Save checkpoint every 10 epochs
        )
        
        # Export model in different formats
        model.export(format='onnx')  # ONNX format
        model.export(format='torchscript')  # TorchScript format
        
        print(f"Training completed. Results saved to {args.output_dir}/yolov11n_soccer_tracking")
        return model
    
    except Exception as e:
        print(f"Error during training: {e}")
        raise

def main():
    args = parse_args()
    
    print("Converting SoccerNet tracking dataset to YOLO format...")
    dataset_yaml = convert_to_yolo_dataset(args.data_dir)
    
    print(f"Starting YOLOv11n training for {args.epochs} epochs...")
    try:
        model = train(args, dataset_yaml)
        print("Training complete!")
    except Exception as e:
        print(f"Training failed: {e}")
        print("Please check your dataset and try again.")

if __name__ == "__main__":
    main()