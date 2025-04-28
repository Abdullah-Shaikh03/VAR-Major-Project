import os
import yaml
import argparse
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

def prepare_dataset(data_dir):
    """
    Prepare SoccerNet tracking dataset for YOLO format
    """
    train_dir = os.path.join(data_dir, 'train')
    val_dir = os.path.join(data_dir, 'test')  # Using test set as validation
    
    # Create YOLO dataset configuration
    dataset_config = {
        'path': data_dir,
        'train': train_dir,
        'val': val_dir,
        'nc': 3,  # Number of classes (player team left, player team right, ball)
        'names': ['team_left', 'team_right', 'ball']
    }
    
    # Create dataset YAML file
    os.makedirs('./data', exist_ok=True)
    with open('./data/soccernet.yaml', 'w') as f:
        yaml.dump(dataset_config, f)
    
    print(f"Dataset configuration saved to ./data/soccernet.yaml")
    return './data/soccernet.yaml'

def convert_annotations(data_dir):
    """
    Convert SoccerNet annotations to YOLO format
    """
    train_dir = os.path.join(data_dir, 'train')
    test_dir = os.path.join(data_dir, 'test')
    
    for dataset_dir in [train_dir, test_dir]:
        if not os.path.exists(dataset_dir):
            continue
            
        for sequence_dir in os.listdir(dataset_dir):
            seq_path = os.path.join(dataset_dir, sequence_dir)
            if not os.path.isdir(seq_path):
                continue
                
            # Read sequence info
            seqinfo_path = os.path.join(seq_path, 'seqinfo.ini')
            if not os.path.exists(seqinfo_path):
                continue
                
            with open(seqinfo_path, 'r') as f:
                lines = f.readlines()
                
            seq_info = {}
            for line in lines:
                if '=' in line:
                    key, value = line.strip().split('=')
                    seq_info[key.strip()] = value.strip()
            
            img_width = int(seq_info.get('imWidth', 1920))
            img_height = int(seq_info.get('imHeight', 1080))
            
            # Create labels directory
            img_dir = os.path.join(seq_path, 'img1')
            labels_dir = os.path.join(seq_path, 'labels')
            os.makedirs(labels_dir, exist_ok=True)
            
            # Convert ground truth annotations
            gt_path = os.path.join(seq_path, 'gt', 'gt.txt')
            if os.path.exists(gt_path):
                frame_annotations = {}
                
                with open(gt_path, 'r') as f:
                    lines = f.readlines()
                
                for line in lines:
                    parts = line.strip().split(',')
                    if len(parts) < 6:
                        continue
                        
                    frame_id = int(parts[0])
                    # Skip negative tracklet IDs
                    if int(parts[1]) < 0:
                        continue
                        
                    x, y, w, h = map(float, parts[2:6])
                    
                    # Determine class (0: team_left, 1: team_right, 2: ball)
                    # This is a simplification - in a real scenario, you'd use gameinfo.ini
                    # to determine proper class assignments
                    class_id = 0  # Default to team_left
                    
                    # Gameinfo.ini would provide this mapping
                    # For now using a placeholder logic
                    if "team left" in line.lower():
                        class_id = 0
                    elif "team right" in line.lower():
                        class_id = 1
                    elif "ball" in line.lower():
                        class_id = 2
                    
                    # Convert to YOLO format (normalized centerx, centery, width, height)
                    x_center = (x + w/2) / img_width
                    y_center = (y + h/2) / img_height
                    w_norm = w / img_width
                    h_norm = h / img_height
                    
                    if frame_id not in frame_annotations:
                        frame_annotations[frame_id] = []
                        
                    frame_annotations[frame_id].append(f"{class_id} {x_center} {y_center} {w_norm} {h_norm}")
                
                # Write YOLO format annotations
                for frame_id, annotations in frame_annotations.items():
                    # YOLO expects labels with same name as image but .txt extension
                    label_file = os.path.join(labels_dir, f"{frame_id:06d}.txt")
                    
                    with open(label_file, 'w') as f:
                        f.write('\n'.join(annotations))
            
            print(f"Converted annotations for {sequence_dir}")

def train(args, dataset_yaml):
    """
    Train YOLOv11n model on the prepared dataset
    """
    # Initialize model
    model = YOLO('yolov11n.pt')  # Load YOLOv11n pre-trained model
    
    # Train the model
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
        verbose=True
    )
    
    # Save trained model
    model.export(format='onnx')  # Also export to ONNX format for deployment
    
    print(f"Training completed. Results saved to {args.output_dir}/yolov11n_soccer_tracking")
    return model

def main():
    args = parse_args()
    
    print("Converting SoccerNet tracking dataset to YOLO format...")
    convert_annotations(args.data_dir)
    
    print("Preparing dataset configuration...")
    dataset_yaml = prepare_dataset(args.data_dir)
    
    print(f"Starting YOLOv11n training for {args.epochs} epochs...")
    model = train(args, dataset_yaml)
    
    print("Training complete!")

if __name__ == "__main__":
    main()