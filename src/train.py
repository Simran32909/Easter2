#!/usr/bin/env python3

import os
import sys
import argparse
import wandb
import tensorflow as tf
from easter_model import train
import config


def setup_gpu(gpu_index=None):
    """Setup GPU configuration"""
    gpus = tf.config.experimental.list_physical_devices('GPU')

    if not gpus:
        print("No GPUs found! Training will use CPU.")
        return False

    print(f"Available GPUs: {len(gpus)}")
    for i, gpu in enumerate(gpus):
        print(f"  GPU {i}: {gpu.name}")

    if gpu_index is not None:
        if gpu_index >= len(gpus) or gpu_index < 0:
            print(f"Error: GPU index {gpu_index} is invalid. Available GPUs: 0-{len(gpus) - 1}")
            return False

        try:
            # Restrict TensorFlow to only use the specified GPU
            tf.config.experimental.set_visible_devices(gpus[gpu_index], 'GPU')

            # Enable memory growth to avoid allocating all GPU memory at once
            tf.config.experimental.set_memory_growth(gpus[gpu_index], True)

            print(f"Using GPU {gpu_index}: {gpus[gpu_index].name}")
            return True

        except RuntimeError as e:
            print(f"Error setting up GPU {gpu_index}: {e}")
            return False
    else:
        try:
            # Enable memory growth for all GPUs
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            print("Using all available GPUs with memory growth enabled")
            return True
        except RuntimeError as e:
            print(f"Error setting up GPUs: {e}")
            return False


def check_gpu_memory(gpu_index=None):
    """Check GPU memory usage and availability"""
    try:
        import subprocess
        result = subprocess.run(['nvidia-smi', '--query-gpu=index,memory.used,memory.total,utilization.gpu',
                                 '--format=csv,noheader,nounits'],
                                capture_output=True, text=True)

        if result.returncode == 0:
            lines = result.stdout.strip().split('\n')
            print("\nGPU Memory Status:")
            print("GPU | Memory Used/Total | GPU Util%")
            print("-" * 40)

            for line in lines:
                parts = line.split(', ')
                if len(parts) >= 4:
                    idx, mem_used, mem_total, util = parts
                    marker = " <-- Selected" if gpu_index is not None and int(idx) == gpu_index else ""
                    print(f"{idx:3} | {mem_used:>5}/{mem_total:<5} MB | {util:>3}%{marker}")
            print()

    except Exception as e:
        print(f"Could not check GPU memory: {e}")


def setup_directories():
    """Create necessary directories for training"""
    directories = [
        '../weights',
        '../logs',
        config.DATA_PATH
    ]

    for directory in directories:
        if not os.path.exists(directory):
            os.makedirs(directory)
            print(f"Created directory: {directory}")


def check_dataset():
    """Check if dataset exists and is properly formatted"""
    if not os.path.exists(config.DATA_PATH):
        print(f"Error: Dataset path {config.DATA_PATH} does not exist!")
        return False

    # Look for JSON files
    json_files = []
    png_files = []

    for root, dirs, files in os.walk(config.DATA_PATH):
        for file in files:
            if file.endswith('.json'):
                json_files.append(os.path.join(root, file))
            elif file.endswith('.png'):
                png_files.append(os.path.join(root, file))

    print(f"Found {len(json_files)} JSON files and {len(png_files)} PNG files")

    if len(json_files) == 0:
        print("Error: No JSON files found in dataset!")
        return False

    if len(png_files) == 0:
        print("Error: No PNG files found in dataset!")
        return False

    return True


def main():
    parser = argparse.ArgumentParser(description='Train Easter2 OCR on Sharada Manuscript Dataset')
    parser.add_argument('--data_path', type=str, default=config.DATA_PATH,
                        help='Path to dataset directory')
    parser.add_argument('--batch_size', type=int, default=config.BATCH_SIZE,
                        help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=config.EPOCHS,
                        help='Number of epochs to train')
    parser.add_argument('--learning_rate', type=float, default=config.LEARNING_RATE,
                        help='Learning rate')
    parser.add_argument('--load_checkpoint', type=str, default='',
                        help='Path to checkpoint to resume training from')
    parser.add_argument('--wandb_project', type=str, default=config.WANDB_PROJECT,
                        help='WandB project name')
    parser.add_argument('--wandb_run_name', type=str, default=config.WANDB_RUN_NAME,
                        help='WandB run name')
    parser.add_argument('--no_wandb', action='store_true',
                        help='Disable WandB logging')
    parser.add_argument('--gpu', type=int, default=None,
                        help='GPU index to use (0-3 for your setup). If not specified, uses all available GPUs')

    args = parser.parse_args()

    # Update config with command line arguments
    config.DATA_PATH = args.data_path
    config.BATCH_SIZE = args.batch_size
    config.EPOCHS = args.epochs
    config.LEARNING_RATE = args.learning_rate
    config.WANDB_PROJECT = args.wandb_project
    config.WANDB_RUN_NAME = args.wandb_run_name

    if args.load_checkpoint:
        config.LOAD = True
        config.LOAD_CHECKPOINT_PATH = args.load_checkpoint

    print("=" * 80)
    print("EASTER2 OCR TRAINING - SHARADA MANUSCRIPT DATASET")
    print("=" * 80)
    print(f"Data Path: {config.DATA_PATH}")
    print(f"Batch Size: {config.BATCH_SIZE}")
    print(f"Epochs: {config.EPOCHS}")
    print(f"Learning Rate: {config.LEARNING_RATE}")
    print(f"Input Size: {config.INPUT_WIDTH}x{config.INPUT_HEIGHT}")
    print(f"Load Checkpoint: {config.LOAD}")
    if config.LOAD:
        print(f"Checkpoint Path: {config.LOAD_CHECKPOINT_PATH}")
    print("=" * 80)

    # Check GPU memory status
    check_gpu_memory(args.gpu)

    # Setup GPU
    if not setup_gpu(args.gpu):
        print("GPU setup failed. Exiting...")
        sys.exit(1)

    # Setup directories
    setup_directories()

    # Check dataset
    if not check_dataset():
        print("Dataset check failed. Please ensure your dataset is properly organized.")
        sys.exit(1)

    # Login to WandB if not disabled
    if not args.no_wandb:
        try:
            wandb.login()
            print("WandB login successful")
        except Exception as e:
            print(f"WandB login failed: {e}")
            print("Continuing without WandB logging...")
            # You might want to modify the train function to handle this

    try:
        # Start training
        print("Starting training...")
        history = train()
        print("Training completed successfully!")

        # Print final summary
        print("\n" + "=" * 80)
        print("TRAINING SUMMARY")
        print("=" * 80)
        print(f"Total epochs completed: {len(history.history['loss'])}")
        print(f"Final training loss: {history.history['loss'][-1]:.4f}")
        if 'val_loss' in history.history:
            print(f"Final validation loss: {history.history['val_loss'][-1]:.4f}")
        print(f"Model saved to: {config.BEST_MODEL_PATH}")
        print("=" * 80)

    except KeyboardInterrupt:
        print("\nTraining interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\nTraining failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()