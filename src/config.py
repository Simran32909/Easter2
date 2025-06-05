# config.py

# Input dataset related settings
DATA_PATH = "/ssd_scratch/jyothi.swaroopa/Simran/data/synthetic/"
INPUT_HEIGHT = 32
INPUT_WIDTH = 2304
INPUT_SHAPE = (INPUT_WIDTH, INPUT_HEIGHT)

TACO_AUGMENTAION_FRACTION = 0.9

# If Long lines augmentation is needed (see paper)
LONG_LINES = False
LONG_LINES_FRACTION = 0.3

# Model training parameters
BATCH_SIZE = 1#64
EPOCHS = 10#100
VOCAB_SIZE = 163
DROPOUT = True
OUTPUT_SHAPE = 143

# Train/Validation/Test split ratios
TRAIN_SPLIT = 0.8
VALIDATION_SPLIT = 0.1
TEST_SPLIT = 0.1

# Train sample control
TRAIN_SAMPLE_SIZE = None  # Pass via CLI
VAL_SAMPLE_SIZE = None    # Pass via CLI

# Checkpointing
LOAD = False
LOAD_CHECKPOINT_PATH = "weights/saved_checkpoint.hdf5"

# Learning parameters
LEARNING_RATE = 0.000001
BATCH_NORM_EPSILON = 1e-5
BATCH_NORM_DECAY = 0.997

# Logging and checkpoint paths
CHECKPOINT_PATH = 'weights/EASTER2--{epoch:02d}.keras'
LOGS_DIR = 'logs'
BEST_MODEL_PATH = "weights/best_model.keras"
FINAL_MODEL_PATH = "weights/final_model.keras"

# WandB Configuration
WANDB_PROJECT = "easter2-sharada-ocr"
WANDB_RUN_NAME = "sharada-manuscript-training"

# CER Evaluation
CER_EVAL_SAMPLES = 100