#!/usr/bin/env python3
# src/train.py - Soccer Offside Detection Model Training

import argparse
import yaml
from pathlib import Path
from ultralytics import YOLO
from src.utils.logger import setup_logger

def parse_args():
    parser = argparse.ArgumentParser(description='Train YOLOv8 model for soccer offside detection')
    parser.add_argument('--data', type=str, required=True, help='Path to dataset config YAML')
    parser.add_argument('--weights', type=str, default='yolov8x.pt', help='Pretrained weights')
    parser.add_argument('--epochs', type=int, default=100, help='Number of training epochs')
    parser.add_argument('--imgsz', type=int, default=1280, help='Image size for training')
    parser.add_argument('--batch', type=int, default=16, help='Batch size')
    parser.add_argument('--device', type=str, default='0', help='Device to use (e.g. 0,1,2,3 or cpu)')
    parser.add_argument('--name', type=str, default='soccer_offside', help='Training run name')
    parser.add_argument('--save-period', type=int, default=10, help='Save checkpoint every n epochs')
    return parser.parse_args()

def validate_config(config_path):
    """Validate the dataset configuration file"""
    required_keys = ['path', 'train', 'val', 'nc', 'names']
    with open(config_path) as f:
        config = yaml.safe_load(f)
    
    for key in required_keys:
        if key not in config:
            raise ValueError(f"Missing required config key: {key}")
    
    data_path = Path(config['path'])
    if not data_path.exists():
        raise FileNotFoundError(f"Dataset path not found: {data_path}")
    
    return config

def prepare_dataset_structure(base_path):
    """Ensure proper dataset structure exists"""
    base_path = Path(base_path)
    dirs = ['images/train', 'images/val', 'images/test',
            'labels/train', 'labels/val', 'labels/test']
    
    for dir in dirs:
        (base_path / dir).mkdir(parents=True, exist_ok=True)
    
    return base_path

def train_model(args):
    logger = setup_logger('train')
    logger.info(f"Starting training with config: {args.data}")
    
    # Validate config
    config = validate_config(args.data)
    data_path = prepare_dataset_structure(config['path'])
    
    # Load model
    model = YOLO(args.weights)
    
    # Training parameters
    train_args = {
        'data': args.data,
        'epochs': args.epochs,
        'imgsz': args.imgsz,
        'batch': args.batch,
        'device': args.device,
        'name': args.name,
        'save_period': args.save_period,
        'exist_ok': True,  # Overwrite existing runs
        'pretrained': True,
        'optimizer': 'auto',
        'lr0': 0.01,       # Initial learning rate
        'cos_lr': True,    # Cosine learning rate scheduler
        'label_smoothing': 0.1,
        'patience': 50,    # Early stopping patience
        'freeze': None,    # Don't freeze any layers
    }
    
    # Start training
    results = model.train(**train_args)
    
    # Save best model
    best_model_path = Path(results.save_dir) / 'weights' / 'best.pt'
    logger.info(f"Training complete. Best model saved to: {best_model_path}")
    
    return str(best_model_path)

if __name__ == "__main__":
    args = parse_args()
    train_model(args)