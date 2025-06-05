import config
import tensorflow
import tensorflow as tf
from tensorflow import keras
from data_loader import data_loader
import wandb
from wandb.integration.keras import WandbCallback
import numpy as np
import itertools
from editdistance import eval as edit_distance
import psutil
import GPUtil
from keras.callbacks import Callback
from tqdm import tqdm


def ctc_loss(args):
    """
    Calculates the Connectionist Temporal Classification (CTC) loss.
    Args:
        args (tuple): A tuple containing:
            y_pred (tf.Tensor): The output tensor from the model (softmax probabilities).
            labels (tf.Tensor): The ground truth labels.
            input_length (tf.Tensor): The length of the input sequences.
            label_length (tf.Tensor): The length of the ground truth label sequences.
    Returns:
        tf.Tensor: The CTC loss.
    """
    y_pred, labels, input_length, label_length = args
    return tf.keras.backend.ctc_batch_cost(
        labels,
        y_pred,
        input_length,
        label_length
    )


def ctc_custom(args):
    """
    Custom CTC loss with a focal loss-inspired modification.
    Args:
        args (tuple): A tuple containing:
            y_pred (tf.Tensor): The output tensor from the model (softmax probabilities).
            labels (tf.Tensor): The ground truth labels.
            input_length (tf.Tensor): The length of the input sequences.
            label_length (tf.Tensor): The length of the ground truth label sequences.
    Returns:
        tf.Tensor: The modified CTC loss.
    """
    y_pred, labels, input_length, label_length = args
    
    # Calculate base CTC loss
    ctc_loss_val = tf.keras.backend.ctc_batch_cost(
        labels,
        y_pred,
        input_length,
        label_length
    )
    
    # Simple focal loss-like modification
    # Reduce the impact of easy samples by scaling the loss
    gamma = 0.5
    alpha = 0.25
    
    # Compute a simple "difficulty" measure
    # Lower loss values indicate easier samples
    difficulty = tf.exp(-ctc_loss_val)
    
    # Apply focal loss-like scaling
    # Samples that are easier (higher difficulty) get less weight
    scaled_loss = alpha * tf.pow((1.0 - difficulty), gamma) * ctc_loss_val
    
    return scaled_loss


def batch_norm(inputs):
    """
    Applies Batch Normalization to the inputs.
    Args:
        inputs (tf.Tensor): The input tensor to the batch normalization layer.
    Returns:
        tf.Tensor: The output tensor after batch normalization.
    """
    return tensorflow.keras.layers.BatchNormalization(
        momentum=config.BATCH_NORM_DECAY,
        epsilon=config.BATCH_NORM_EPSILON
    )(inputs)


def add_global_context(data, filters):
    """
    1D Squeeze and Excitation Layer.
    Args:
        data (tf.Tensor): The input tensor to the Squeeze and Excitation block.
        filters (int): The number of filters (channels) in the input data.
    Returns:
        tf.Tensor: The output tensor after applying Squeeze and Excitation.
    """
    pool = tensorflow.keras.layers.GlobalAveragePooling1D()(data)

    pool = tensorflow.keras.layers.Dense(
        filters // 8,
        activation='relu'
    )(pool)

    pool = tensorflow.keras.layers.Dense(
        filters,
        activation='sigmoid'
    )(pool)

    pool = tensorflow.keras.layers.Reshape((1, filters))(pool)
    final = tensorflow.keras.layers.Multiply()([data, pool])
    return final


def easter_unit(old, data, filters, kernel, stride, dropouts):
    """
    Easter unit with dense residual connections.
    Args:
        old (tf.Tensor): The residual connection from a previous block.
        data (tf.Tensor): The current input data for the unit.
        filters (int): The number of filters for the convolutional layers.
        kernel (int): The kernel size for the convolutional layers.
        stride (int): The stride for the convolutional layers.
        dropouts (float): The dropout rate.
    Returns:
        tuple: A tuple containing:
            - tf.Tensor: The output tensor of the Easter unit.
            - tf.Tensor: The updated 'old' tensor for the next residual connection.
    """
    old = tensorflow.keras.layers.Conv1D(
        filters=filters,
        kernel_size=(1),
        strides=(1),
        padding="same"
    )(old)
    old = batch_norm(old)

    this = tensorflow.keras.layers.Conv1D(
        filters=filters,
        kernel_size=(1),
        strides=(1),
        padding="same"
    )(data)
    this = batch_norm(this)

    old = tensorflow.keras.layers.Add()([old, this])

    # First Block
    data = tensorflow.keras.layers.Conv1D(
        filters=filters,
        kernel_size=(kernel),
        strides=(stride),
        padding="same"
    )(data)

    data = batch_norm(data)
    data = tensorflow.keras.layers.Activation('relu')(data)
    data = tensorflow.keras.layers.Dropout(dropouts)(data)

    # Second Block
    data = tensorflow.keras.layers.Conv1D(
        filters=filters,
        kernel_size=(kernel),
        strides=(stride),
        padding="same"
    )(data)

    data = batch_norm(data)
    data = tensorflow.keras.layers.Activation('relu')(data)
    data = tensorflow.keras.layers.Dropout(dropouts)(data)

    # Third Block
    data = tensorflow.keras.layers.Conv1D(
        filters=filters,
        kernel_size=(kernel),
        strides=(stride),
        padding="same"
    )(data)

    data = batch_norm(data)

    # squeeze and excitation
    data = add_global_context(data, filters)

    final = tensorflow.keras.layers.Add()([old, data])

    data = tensorflow.keras.layers.Activation('relu')(final)
    data = tensorflow.keras.layers.Dropout(dropouts)(data)

    return data, old


def Easter2():
    """
    Constructs the Easter2 Keras model.
    Returns:
        tf.keras.Model: The model that outputs prediction logits.
    """
    input_data = tensorflow.keras.layers.Input(
        name='the_input',
        shape=config.INPUT_SHAPE
    )

    data = tensorflow.keras.layers.Conv1D(
        filters=128,
        kernel_size=(3),
        strides=(2),
        padding="same"
    )(input_data)

    data = batch_norm(data)
    data = tensorflow.keras.layers.Activation('relu')(data)
    data = tensorflow.keras.layers.Dropout(0.2)(data)

    data = tensorflow.keras.layers.Conv1D(
        filters=128,
        kernel_size=(3),
        strides=(2),
        padding="same"
    )(data)

    data = batch_norm(data)
    data = tensorflow.keras.layers.Activation('relu')(data)
    data = tensorflow.keras.layers.Dropout(0.2)(data)

    old = data

    # 3 * 3 Easter Blocks (with dense residuals)
    data, old = easter_unit(old, data, 256, 5, 1, 0.2)
    data, old = easter_unit(old, data, 256, 7, 1, 0.2)
    data, old = easter_unit(old, data, 256, 9, 1, 0.3)

    data = tensorflow.keras.layers.Conv1D(
        filters=512,
        kernel_size=(11),
        strides=(1),
        padding="same",
        dilation_rate=2
    )(data)

    data = batch_norm(data)
    data = tensorflow.keras.layers.Activation('relu')(data)
    data = tensorflow.keras.layers.Dropout(0.4)(data)

    data = tensorflow.keras.layers.Conv1D(
        filters=512,
        kernel_size=(1),
        strides=(1),
        padding="same"
    )(data)

    data = batch_norm(data)
    data = tensorflow.keras.layers.Activation('relu')(data)
    data = tensorflow.keras.layers.Dropout(0.4)(data)

    data = tensorflow.keras.layers.Conv1D(
        filters=config.VOCAB_SIZE,
        kernel_size=(1),
        strides=(1),
        padding="same"
    )(data)

    # Output logits (not softmax)
    y_pred = data

    # Create model that outputs prediction logits
    model = tensorflow.keras.models.Model(
        inputs=input_data, 
        outputs=y_pred
    )

    # Defining other training parameters
    Optimizer = tensorflow.keras.optimizers.Adam(
        learning_rate=config.LEARNING_RATE,
        clipnorm=1.0,
        epsilon=1e-8,  # Increased epsilon for better numerical stability
        beta_1=0.9,
        beta_2=0.999
    )
    
    # Compile model with standard categorical crossentropy
    model.compile(
        optimizer=Optimizer, 
        loss=None,  # Loss will be handled externally
        jit_compile=False  # Explicitly disable JIT compilation
    )
    
    return model


def decoder(output, letters):
    """
    Decodes CTC output probabilities into text sequences.
    Args:
        output (np.ndarray): The output probabilities from the CTC layer.
        letters (list): A list of characters representing the vocabulary.
    Returns:
        list: A list of decoded text strings.
    """
    ret = []
    for j in range(output.shape[0]):
        out_best = list(np.argmax(output[j, :], 1))
        out_best = [k for k, g in itertools.groupby(out_best)] # Remove consecutive duplicates
        outstr = ''
        for c in out_best:
            if c < len(letters): # Ensure character index is within bounds
                outstr += letters[c]
        ret.append(outstr)
    return ret


def calculate_cer(predictions, ground_truths):
    """
    Calculates Character Error Rate (CER), accuracy, and perfect matches.
    Args:
        predictions (list): A list of predicted text strings.
        ground_truths (list): A list of ground truth text strings.
    Returns:
        tuple: A tuple containing:
            - float: The Character Error Rate (CER) as a percentage.
            - float: The accuracy as a percentage.
            - int: The number of perfect matches.
    """
    total_chars = 0
    total_errors = 0
    perfect_matches = 0

    for pred, gt in zip(predictions, ground_truths):
        pred = pred.strip()
        gt = gt.strip()

        if pred == gt:
            perfect_matches += 1

        total_chars += len(gt)
        total_errors += edit_distance(pred, gt)

    cer = (total_errors / total_chars) * 100 if total_chars > 0 else 0
    accuracy = ((total_chars - total_errors) / total_chars) * 100 if total_chars > 0 else 0

    return cer, accuracy, perfect_matches


def get_system_metrics():
    """
    Get system resource usage metrics (GPU, CPU, RAM).
    Returns:
        dict: A dictionary containing 'gpu_memory_used', 'cpu_percent', and 'memory_percent'.
              Returns zeros if metrics cannot be retrieved.
    """
    try:
        # GPU metrics
        gpus = GPUtil.getGPUs()
        gpu_memory_used = gpus[0].memoryUsed if gpus else 0

        # CPU and RAM metrics
        cpu_percent = psutil.cpu_percent()
        memory_percent = psutil.virtual_memory().percent

        return {
            'gpu_memory_used': gpu_memory_used,
            'cpu_percent': cpu_percent,
            'memory_percent': memory_percent
        }
    except Exception as e:
        print(f"Error getting system metrics: {e}")
        return {
            'gpu_memory_used': 0,
            'cpu_percent': 0,
            'memory_percent': 0
        }


class CERCallback(Callback):
    """Custom callback to calculate CER during training"""

    def __init__(self, validation_data, char_list):
        super().__init__()
        self.validation_data = validation_data
        self.char_list = char_list
        self.prediction_model = None  # Will be initialized later by Keras calling set_model

    def set_model(self, model):
        """
        Called by Keras to set the model.
        Here we initialize the prediction model using the 'model' provided by Keras.
        """
        super().set_model(model)
        try:
            # Use the 'Final' layer which we now know exists in the model
            self.prediction_model = tf.keras.models.Model(
                inputs=model.input,
                outputs=model.get_layer('Final').output
            )
            print("Prediction model initialized successfully in CERCallback.")
        except Exception as e:
            print(f"Error creating prediction model: {e}")
            self.prediction_model = None

    def on_epoch_end(self, epoch, logs=None):
        """
        Actions to perform at the end of each training epoch.
        Calculates CER on validation data and logs metrics.
        """
        # Ensure prediction_model is initialized before proceeding
        if self.prediction_model is None:
            print("Warning: prediction_model was not initialized. Skipping CER calculation for this epoch.")
            return

        # Get validation samples
        self.validation_data.validationSet()
        imgs, truths, _ = self.validation_data.getValidationImage()

        if not imgs: # Check if imgs list is empty
            print("No validation images found. Skipping CER calculation for this epoch.")
            return

        predictions = []
        ground_truths = []

        # Make predictions
        # Limit the number of samples to evaluate based on config.CER_EVAL_SAMPLES
        for img, truth in zip(imgs[:config.CER_EVAL_SAMPLES], truths[:config.CER_EVAL_SAMPLES]):
            try:
                # Ensure img is correctly shaped for prediction (e.g., add batch dimension if missing)
                # Assuming img is a single image, reshape it to (1, height, width, channels)
                if len(img.shape) == 3: # Assuming (height, width, channels)
                    img_batch = tf.expand_dims(img, axis=0)
                else: # Assuming it's already (batch, height, width, channels) or similar
                    img_batch = img

                output = self.prediction_model.predict(img_batch, verbose=0)
                pred = decoder(output, self.char_list)[0] # Assuming decoder returns a list
                predictions.append(pred)
                ground_truths.append(truth)
            except Exception as e:
                print(f"Error in prediction for a sample: {e}")
                continue # Continue to the next sample even if one fails

        if predictions: # Only proceed if there were successful predictions
            # Calculate CER metrics
            val_cer, val_accuracy, perfect_matches = calculate_cer(predictions, ground_truths)

            # Get system metrics
            system_metrics = get_system_metrics()

            # Log to WandB
            # Ensure self.model.optimizer and self.model.optimizer.lr exist before accessing
            learning_rate = 0.0 # Default if not available
            if hasattr(self.model, 'optimizer') and hasattr(self.model.optimizer, 'lr'):
                learning_rate = float(tf.keras.backend.get_value(self.model.optimizer.lr))

            wandb.log({
                'val_cer': val_cer,
                'val_accuracy': val_accuracy,
                'val_perfect_matches': perfect_matches,
                'learning_rate': learning_rate,
                **system_metrics
            })

            print(
                f"\nValidation CER: {val_cer:.2f}%, Accuracy: {val_accuracy:.2f}%, Perfect matches: {perfect_matches}"
            )
        else:
            print("No successful predictions were made for CER calculation this epoch.")


def train():
    """
    Main training function for the Easter2 model.
    Initializes WandB, loads data, sets up callbacks, and starts model training.
    """
    # Check if WandB should be disabled
    if hasattr(config, 'DISABLE_WANDB') and config.DISABLE_WANDB:
        print("WandB logging disabled by command line argument")
        use_wandb = False
    else:
        use_wandb = True

    # Initialize WandB only if not disabled
    if use_wandb:
        wandb.init(
            project=config.WANDB_PROJECT,
            name=config.WANDB_RUN_NAME,
            config={
                "learning_rate": config.LEARNING_RATE,
                "batch_size": config.BATCH_SIZE,
                "epochs": config.EPOCHS,
                "input_width": config.INPUT_WIDTH,
                "input_height": config.INPUT_HEIGHT,
                "vocab_size": config.VOCAB_SIZE,
                "output_shape": config.OUTPUT_SHAPE,
                "train_samples": getattr(config, 'TRAIN_SAMPLES', None),
                "val_samples": getattr(config, 'VAL_SAMPLES', None),
                "train_val_split": getattr(config, 'TRAIN_VAL_SPLIT', 0.2),
                "cer_eval_samples": getattr(config, 'CER_EVAL_SAMPLES', 100)
            }
        )

    # Creating Easter2 model
    model = Easter2()

    # Loading checkpoint for transfer/resuming learning
    if config.LOAD:
        print("Initializing from checkpoint:", config.LOAD_CHECKPOINT_PATH)
        try:
            model.load_weights(config.LOAD_CHECKPOINT_PATH)
            print("Init weights loaded successfully....")
        except Exception as e:
            print(f"Could not load weights: {e}")

    # Loading Metadata, about training, validation and Test sets
    print("Loading metadata...")
    training_data = data_loader(config.DATA_PATH, config.BATCH_SIZE)
    validation_data = data_loader(config.DATA_PATH, config.BATCH_SIZE)
    test_data = data_loader(config.DATA_PATH, config.BATCH_SIZE)

    training_data.trainSet()
    validation_data.validationSet()
    test_data.testSet()

    # Apply custom sample limits if specified
    if hasattr(config, 'TRAIN_SAMPLES') and config.TRAIN_SAMPLES is not None:
        original_train_samples = len(training_data.samples)
        training_data.samples = training_data.samples[:config.TRAIN_SAMPLES]
        print(f"Limited training samples: {original_train_samples} -> {len(training_data.samples)}")

    if hasattr(config, 'VAL_SAMPLES') and config.VAL_SAMPLES is not None:
        original_val_samples = len(validation_data.samples)
        validation_data.samples = validation_data.samples[:config.VAL_SAMPLES]
        print(f"Limited validation samples: {original_val_samples} -> {len(validation_data.samples)}")

    print("Training Samples:", len(training_data.samples))
    print("Validation Samples:", len(validation_data.samples))
    print("Test Samples:", len(test_data.samples))
    print("CharList Size:", len(training_data.charList))

    # Update config with actual vocab size
    config.VOCAB_SIZE = len(training_data.charList) + 1  # +1 for blank token

    # Prepare callbacks and training parameters
    CHECKPOINT = tensorflow.keras.callbacks.ModelCheckpoint(
        filepath=config.CHECKPOINT_PATH,
        monitor='val_loss',
        verbose=1,
        save_best_only=True,
        mode='min',
        save_freq='epoch'
    )

    BEST_MODEL_CHECKPOINT = tensorflow.keras.callbacks.ModelCheckpoint(
        filepath=config.BEST_MODEL_PATH,
        monitor='val_loss',
        verbose=1,
        save_best_only=True,
        mode='min'
    )

    # Custom CER callback
    CER_CALLBACK = CERCallback(validation_data, training_data.charList)

    # Prepare callbacks list
    callbacks_list = [
        CHECKPOINT,
        BEST_MODEL_CHECKPOINT,
        CER_CALLBACK
    ]

    # Add WandB callback if enabled
    if use_wandb:
        WANDB_CALLBACK = WandbCallback(
            monitor='val_loss',
            mode='min',
            save_model=False
        )
        callbacks_list.append(WANDB_CALLBACK)

    # Prepare training data generator
    train_generator = training_data.getNext('train')
    val_generator = validation_data.getNext('validation')

    # Compile the model with a custom loss function
    def ctc_loss_wrapper(y_true, y_pred):
        input_length = tf.fill([tf.shape(y_pred)[0]], tf.shape(y_pred)[1])
        label_length = tf.reduce_sum(tf.cast(tf.not_equal(y_true, len(training_data.charList)), tf.int32), axis=1)
        
        sparse_labels = tf.keras.backend.ctc_label_dense_to_sparse(y_true, label_length)
        
        loss = tf.nn.ctc_loss(
            labels=sparse_labels,
            logits=y_pred,
            label_length=label_length,
            logit_length=input_length,
            blank_index=config.VOCAB_SIZE - 1,
            logits_time_major=False
        )
        
        return tf.reduce_mean(loss)

    model.compile(
        optimizer=tensorflow.keras.optimizers.Adam(
            learning_rate=config.LEARNING_RATE,
            clipnorm=1.0,
            epsilon=1e-8,
            beta_1=0.9,
            beta_2=0.999
        ),
        loss=ctc_loss_wrapper
    )

    # Prepare validation data
    val_data = next(val_generator)

    # Fit the model
    history = model.fit(
        train_generator,
        validation_data=val_data,
        epochs=config.EPOCHS,
        callbacks=callbacks_list,
        steps_per_epoch=len(training_data.samples) // config.BATCH_SIZE,
        validation_steps=1
    )

    # Save final model
    model.save_weights(config.FINAL_MODEL_PATH)

    # Close WandB run only if it was initialized
    if use_wandb:
        wandb.finish()

    return model  # Return the trained model


if __name__ == "__main__":
    train()