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

    # Add custom train/val sample arguments
    parser.add_argument('--train_samples', type=int, default=None,
                        help='Number of training samples to use (default: use all available)')
    parser.add_argument('--val_samples', type=int, default=None,
                        help='Number of validation samples to use (default: use all available)')
    parser.add_argument('--train_val_split', type=float, default=0.2,
                        help='Validation split ratio (default: 0.2 = 20%% for validation)')
    parser.add_argument('--cer_eval_samples', type=int, default=100,
                        help='Number of samples to use for CER evaluation during training (default: 100)')

    args = parser.parse_args()

    # Update config with command line arguments
    config.DATA_PATH = args.data_path
    config.BATCH_SIZE = args.batch_size
    config.EPOCHS = args.epochs
    config.LEARNING_RATE = args.learning_rate
    config.WANDB_PROJECT = args.wandb_project
    config.WANDB_RUN_NAME = args.wandb_run_name

    # Set custom sample sizes in config
    if args.train_samples is not None:
        config.TRAIN_SAMPLES = args.train_samples
    if args.val_samples is not None:
        config.VAL_SAMPLES = args.val_samples
    config.TRAIN_VAL_SPLIT = args.train_val_split
    config.CER_EVAL_SAMPLES = args.cer_eval_samples

    # Set WandB disable flag
    config.DISABLE_WANDB = args.no_wandb

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
    print(f"Train/Val Split: {args.train_val_split:.1%}")

    # Display custom sample information
    if args.train_samples is not None:
        print(f"Custom Training Samples: {args.train_samples}")
    else:
        print("Training Samples: All available")

    if args.val_samples is not None:
        print(f"Custom Validation Samples: {args.val_samples}")
    else:
        print("Validation Samples: All available")

    print(f"CER Evaluation Samples: {args.cer_eval_samples}")
    print(f"WandB Logging: {'Disabled' if args.no_wandb else 'Enabled'}")
    print(f"Load Checkpoint: {config.LOAD}")
    if config.LOAD:
        print(f"Checkpoint Path: {config.LOAD_CHECKPOINT_PATH}")
    print("=" * 80)

    # Validate sample arguments
    if args.train_samples is not None and args.train_samples <= 0:
        print("Error: train_samples must be a positive integer")
        sys.exit(1)

    if args.val_samples is not None and args.val_samples <= 0:
        print("Error: val_samples must be a positive integer")
        sys.exit(1)

    if not (0.0 < args.train_val_split < 1.0):
        print("Error: train_val_split must be between 0.0 and 1.0")
        sys.exit(1)

    if args.cer_eval_samples <= 0:
        print("Error: cer_eval_samples must be a positive integer")
        sys.exit(1)

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

    # Handle WandB login if not disabled
    if not args.no_wandb:
        try:
            wandb.login()
            print("WandB login successful")
        except Exception as e:
            print(f"WandB login failed: {e}")
            print("Disabling WandB logging for this run...")
            config.DISABLE_WANDB = True

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

        # Check for CER metrics in history
        if 'val_loss' in history.history:
            print(f"Final validation loss: {history.history['val_loss'][-1]:.4f}")

        # Look for CER metrics that might be logged by the CERCallback
        for key in history.history.keys():
            if 'cer' in key.lower():
                print(f"Final {key}: {history.history[key][-1]:.4f}")
            elif 'accuracy' in key.lower():
                print(f"Final {key}: {history.history[key][-1]:.4f}%")

        print(f"Best model saved to: {config.BEST_MODEL_PATH}")
        print(f"Final model saved to: ../weights/final_model.hdf5")
        print("=" * 80)

    except KeyboardInterrupt:
        print("\nTraining interrupted by user")
        # Try to finish WandB run gracefully if it was started
        try:
            if not args.no_wandb and not config.DISABLE_WANDB:
                wandb.finish()
        except:
            pass
        sys.exit(1)
    except Exception as e:
        print(f"\nTraining failed with error: {e}")
        import traceback
        traceback.print_exc()
        # Try to finish WandB run gracefully if it was started
        try:
            if not args.no_wandb and not config.DISABLE_WANDB:
                wandb.finish()
        except:
            pass
        sys.exit(1)


if __name__ == "__main__":
    main()