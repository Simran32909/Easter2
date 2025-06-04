#src/config.py

# Input dataset related settings
DATA_PATH = "../data/synthetic/"  # Path to your synthetic dataset
INPUT_HEIGHT = 32
INPUT_WIDTH = 2304
INPUT_SHAPE = (INPUT_WIDTH, INPUT_HEIGHT)

TACO_AUGMENTAION_FRACTION = 0.9

# If Long lines augmentation is needed (see paper)
LONG_LINES = False  # Disabled for manuscript data
LONG_LINES_FRACTION = 0.3

# Model training parameters
BATCH_SIZE = 16  # Reduced due to larger image dimensions
EPOCHS = 1000
VOCAB_SIZE = 163
DROPOUT = True
OUTPUT_SHAPE = 264  # Adjusted from stats

# Train/Validation/Test split ratios
TRAIN_SPLIT = 0.8
VALIDATION_SPLIT = 0.1
TEST_SPLIT = 0.1

# Initializing weights from pre-trained
LOAD = False  # Set to False for training from scratch on new dataset
LOAD_CHECKPOINT_PATH = "../weights/saved_checkpoint.hdf5"

# Other learning parameters
LEARNING_RATE = 0.001  # Reduced learning rate
BATCH_NORM_EPSILON = 1e-5
BATCH_NORM_DECAY = 0.997

# Checkpoints parameters
CHECKPOINT_PATH = '../weights/EASTER2--{epoch:02d}--{val_loss:.4f}--{val_cer:.4f}.hdf5'
LOGS_DIR = '../logs'
BEST_MODEL_PATH = "../weights/best_model.hdf5"

# WandB Configuration
WANDB_PROJECT = "easter2-sharada-ocr"
WANDB_RUN_NAME = "sharada-manuscript-training"

# CER Evaluation
CER_EVAL_SAMPLES = 100  # Number of samples to evaluate CER on during training