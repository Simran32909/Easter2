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
    Optimizer = tensorflow.keras.optimizers.Adam(learning_rate=config.LEARNING_RATE, clipnorm=1.0)
    
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
        self.validation_data = validation_data # data_loader instance
        self.char_list = char_list
        self.prediction_model = None 

    def set_model(self, model):
        """
        Called by Keras to set the model.
        """
        self.model = model # Keras sets this automatically, but good to be explicit for clarity
        self.prediction_model = model # Use the main model for predictions

    def on_epoch_end(self, epoch, logs=None):
        """
        Actions to perform at the end of each training epoch.
        Calculates CER and val_loss on validation data and logs metrics.
        Updates the logs dictionary for other callbacks.
        """
        logs = logs or {}

        if self.prediction_model is None:
            print("Warning: prediction_model was not initialized in CERCallback. Skipping validation metrics for this epoch.")
            # Ensure logs have placeholders if ModelCheckpoint expects them, to avoid KeyError
            logs['val_loss'] = float('inf') 
            logs['val_cer'] = float('inf')
            return

        self.validation_data.validationSet() # Reset validation data iterator

        # Determine number of validation steps
        if not self.validation_data.samples:
            print("No validation samples found. Skipping validation metrics calculation.")
            logs['val_loss'] = float('inf')
            logs['val_cer'] = float('inf')
            return
            
        val_steps = len(self.validation_data.samples) // self.validation_data.batchSize
        if val_steps == 0 and len(self.validation_data.samples) > 0:
            val_steps = 1
        
        if val_steps == 0:
            print("Not enough validation samples for a single batch. Skipping validation metrics.")
            logs['val_loss'] = float('inf')
            logs['val_cer'] = float('inf')
            return

        all_predictions = []
        all_ground_truths = []
        total_val_loss = 0.0
        val_batches_processed = 0

        # Create a tqdm progress bar for validation batches
        val_batch_iterator = tqdm(
            self.validation_data.getNext('val'),
            total=val_steps,
            desc="Validation Batches",
            position=2, # Ensure it appears below training batch progress bar
            leave=False
        )

        for i, (val_inputs_dict, _) in enumerate(val_batch_iterator):
            if i >= val_steps:
                break
            
            val_imgs = val_inputs_dict['the_input']
            val_gtTexts_encoded = val_inputs_dict['the_labels']
            val_input_length = tf.cast(val_inputs_dict['input_length'], tf.int32)
            val_label_length = tf.cast(tf.reshape(val_inputs_dict['label_length'], [-1]), tf.int32)

            try:
                # Get model predictions (logits)
                # The model output is already logits, shape: (batch_size, time_steps, vocab_size)
                logits = self.prediction_model.predict_on_batch(val_imgs)
                logits = tf.cast(logits, tf.float32)

                # The actual sequence length from logits for ctc_loss's logit_length
                # For Easter2 model: Input width 2304 -> s=2 -> 1152 -> s=2 -> 576.
                # This should dynamically get the length from the logits tensor.
                current_batch_logit_length = np.full((tf.shape(logits)[0].numpy(),), tf.shape(logits)[1].numpy(), dtype=np.int32)

                # Calculate CTC loss for the batch
                sparse_labels = tf.keras.backend.ctc_label_dense_to_sparse(val_gtTexts_encoded, val_label_length)
                
                batch_loss = tf.nn.ctc_loss(
                    labels=sparse_labels,
                    logits=logits,
                    label_length=val_label_length,
                    logit_length=current_batch_logit_length, # Use the actual logit length here
                    blank_index=len(self.char_list), # config.VOCAB_SIZE - 1, where VOCAB_SIZE = len(charList) + 1
                    logits_time_major=False
                )
                batch_loss = tf.reduce_mean(batch_loss)
                total_val_loss += batch_loss.numpy()

                # Decode predictions for CER
                # The decoder expects numpy array
                decoded_preds = decoder(logits.numpy(), self.char_list)
                
                # Decode ground truths for CER (from encoded to string)
                # We need the original string ground truths for CER calculation.
                # The getNext() method yields encoded labels. We need to get the original text.
                # For simplicity, we'll re-fetch a slice of ground truth texts if data_loader can provide it easily,
                # or decode them back if charList is available.
                # The current `getNext` doesn't yield original text.
                # Let's fetch them from self.validation_data.samples directly based on current batch index.
                # This is a bit complex as getNext shuffles.
                # A better way would be for getNext to also yield original texts or sample_ids.

                # For now, let's assume we can get them or approximate.
                # We need to be careful here: val_gtTexts_encoded are indices.
                # We need to convert them back to text for CER.
                
                current_batch_samples = self.validation_data.samples[
                    self.validation_data.currIdx - self.validation_data.batchSize : self.validation_data.currIdx
                ]
                
                batch_ground_truths_text = [sample.gtText for sample in current_batch_samples[:len(decoded_preds)]]


                all_predictions.extend(decoded_preds)
                all_ground_truths.extend(batch_ground_truths_text)
                
                val_batches_processed += 1
                val_batch_iterator.set_postfix({'val_loss_batch': f'{batch_loss.numpy():.4f}'})

            except Exception as e:
                print(f"Error during validation batch processing: {e}")
                import traceback
                traceback.print_exc()
                continue
        
        val_batch_iterator.close()

        avg_val_loss = total_val_loss / val_batches_processed if val_batches_processed > 0 else float('inf')
        
        val_cer = float('inf')
        val_accuracy = 0.0
        perfect_matches = 0

        if all_predictions and all_ground_truths:
            # Calculate CER metrics
            val_cer, val_accuracy, perfect_matches = calculate_cer(all_predictions, all_ground_truths)

        # Update logs dictionary for other callbacks (like ModelCheckpoint)
        logs['val_loss'] = avg_val_loss
        logs['val_cer'] = val_cer 
        logs['val_accuracy'] = val_accuracy
        logs['val_perfect_matches'] = perfect_matches

        # Get system metrics
        system_metrics = get_system_metrics()

        # Log to WandB
        learning_rate = 0.0
        if hasattr(self.model, 'optimizer') and hasattr(self.model.optimizer, 'lr'):
            learning_rate = float(tf.keras.backend.get_value(self.model.optimizer.lr))

        wandb_logs = {
            'val_loss': avg_val_loss, # Log the calculated average validation loss
            'val_cer': val_cer,
            'val_accuracy': val_accuracy,
            'val_perfect_matches': perfect_matches,
            'learning_rate': learning_rate,
            **system_metrics
        }
        # Add training loss to wandb logs if available
        if 'loss' in logs:
             wandb_logs['loss'] = logs['loss'] # training loss from the main loop
        
        if wandb.run: # Check if wandb run is active
            wandb.log(wandb_logs)


        print(
            f"Validation CER: {val_cer:.2f}%, Val Loss: {avg_val_loss:.4f}, Accuracy: {val_accuracy:.2f}%, Perfect matches: {perfect_matches}"
        )


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

    @tf.function
    def train_step(inputs, labels, input_length, label_length):
        with tf.GradientTape() as tape:
            logits = model(inputs, training=True)  # shape: (batch_size, time_steps, vocab_size)

            input_length = tf.cast(input_length, tf.int32)
            label_length = tf.cast(tf.reshape(label_length, [-1]), tf.int32)
            logits = tf.cast(logits, tf.float32)

            sparse_labels = tf.keras.backend.ctc_label_dense_to_sparse(labels, label_length)

            loss = tf.nn.ctc_loss(
                labels=sparse_labels,
                logits=logits,
                label_length=label_length,
                logit_length=input_length,
                blank_index=config.VOCAB_SIZE - 1,
                logits_time_major=False
                )

            loss = tf.reduce_mean(loss)

        gradients = tape.gradient(loss, model.trainable_variables)
        model.optimizer.apply_gradients(zip(gradients, model.trainable_variables))

        return loss
    

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

    # Prepare callbacks list - CER_CALLBACK must come BEFORE ModelCheckpoints
    callbacks_list = [
        CER_CALLBACK, # Calculate val_loss and val_cer first
        CHECKPOINT,
        BEST_MODEL_CHECKPOINT,
    ]

    # Add WandB callback if enabled
    if use_wandb:
        WANDB_CALLBACK = WandbCallback(
            monitor='val_loss',
            mode='min',
            save_model=False
        )
        callbacks_list.append(WANDB_CALLBACK)

    # Manually set the model for each callback
    for callback in callbacks_list:
        if hasattr(callback, 'set_model'):
            callback.set_model(model)

    # Training loop with tqdm progress bars
    print("Starting custom training loop...")

    history_log = {
        'loss': [],
        'val_loss': [],
        'val_cer': [],
        'val_accuracy': [],
        'val_perfect_matches': []
    }

    for epoch in tqdm(range(config.EPOCHS), desc="Epochs", position=0):
        print(f"\nEpoch {epoch+1}/{config.EPOCHS}")
        
        # Reset data generators
        training_data.trainSet()
        validation_data.validationSet()
        
        # Training phase with batch progress bar
        total_train_loss = 0.0
        train_batches = 0
        
        steps_per_epoch = len(training_data.samples) // training_data.batchSize
        if steps_per_epoch == 0 and len(training_data.samples) > 0: # Handle cases where dataset is smaller than batch_size
            steps_per_epoch = 1

        # Create a tqdm progress bar for batches
        batch_iterator = tqdm(training_data.getNext('train'), 
                               total=steps_per_epoch, # Set total steps for tqdm
                               desc="Training Batches", 
                               position=1, 
                               leave=False)
        
        for i, (inputs, outputs) in enumerate(batch_iterator):
            if i >= steps_per_epoch: # Break after completing all steps for the epoch
                break

            # Unpack inputs
            batch_inputs = inputs['the_input']
            batch_labels = inputs['the_labels']
            batch_input_length = inputs['input_length']
            batch_label_length = inputs['label_length']
            
            # Perform training step
            loss = train_step(batch_inputs, batch_labels, batch_input_length, batch_label_length)
            total_train_loss += loss.numpy()
            train_batches += 1
            
            # Update batch progress bar
            batch_iterator.set_postfix({'loss': f'{loss.numpy():.4f}'})
        
        # Close batch progress bar
        if train_batches > 0: # Ensure batch_iterator was used
            batch_iterator.close()
        
        # Average training loss
        avg_train_loss = total_train_loss / train_batches if train_batches > 0 else 0.0
        print(f"Training Loss: {avg_train_loss:.4f}")
        
        # Run validation and callbacks
        # Construct logs dictionary for callbacks
        logs_for_callbacks = {'loss': avg_train_loss}

        # Simulate val_loss for callbacks if validation is performed elsewhere or not strictly needed for all callbacks
        # For example, CERCallback calculates its own metrics.
        # If other callbacks strictly need 'val_loss', it should be computed here.
        # For now, we'll pass what we have.
        # If validation_data is available and a validation step is performed, add 'val_loss' here.
        # e.g. val_loss_value = calculate_validation_loss(model, validation_data)
        # logs_for_callbacks['val_loss'] = val_loss_value

        for callback in callbacks_list:
            if hasattr(callback, 'on_epoch_end'):
                # Pass necessary logs. 'val_loss' is often expected by ModelCheckpoint.
                # Since we don't have a direct val_loss from this loop, we might need to adjust
                # or ensure callbacks handle potentially missing metrics gracefully.
                # For simplicity, passing training loss. Actual val_loss would be better.
                callback.on_epoch_end(epoch, logs_for_callbacks)
        
        # Append metrics to history log
        history_log['loss'].append(avg_train_loss)
        history_log['val_loss'].append(logs_for_callbacks.get('val_loss', float('inf')))
        history_log['val_cer'].append(logs_for_callbacks.get('val_cer', float('inf')))
        history_log['val_accuracy'].append(logs_for_callbacks.get('val_accuracy', 0.0))
        history_log['val_perfect_matches'].append(logs_for_callbacks.get('val_perfect_matches', 0))

        # Optional early stopping logic can be added here
        
    # Save final model
    model.save_weights(config.FINAL_MODEL_PATH)

    # Close WandB run only if it was initialized
    if use_wandb:
        wandb.finish()

    return history_log


if __name__ == "__main__":
    train()