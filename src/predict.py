# src/predict.py

import config
import tensorflow
import tensorflow as tf
import itertools
import numpy as np
from editdistance import eval as edit_distance
from tqdm import tqdm
from data_loader import data_loader
import tensorflow.keras.backend as K
import cv2
import matplotlib.pyplot as plt


def ctc_custom(args):
    y_pred, labels, input_length, label_length = args

    ctc_loss = K.ctc_batch_cost(
        labels,
        y_pred,
        input_length,
        label_length
    )
    p = tensorflow.exp(-ctc_loss)
    gamma = 0.5
    alpha = 0.25
    return alpha * (K.pow((1 - p), gamma)) * ctc_loss


def load_easter_model(checkpoint_path):
    """Load the trained Easter2 model"""
    if checkpoint_path == "Empty":
        checkpoint_path = config.BEST_MODEL_PATH
    try:
        checkpoint = tensorflow.keras.models.load_model(
            checkpoint_path,
            custom_objects={'<lambda>': lambda x, y: y,
                            'tensorflow': tf, 'K': K}
        )

        EASTER = tensorflow.keras.models.Model(
            checkpoint.get_layer('the_input').input,
            checkpoint.get_layer('Final').output
        )
    except:
        print("Unable to Load Checkpoint.")
        return None
    return EASTER


def decoder(output, letters):
    """Decode CTC output to readable text"""
    ret = []
    for j in range(output.shape[0]):
        out_best = list(np.argmax(output[j, :], 1))
        out_best = [k for k, g in itertools.groupby(out_best)]
        outstr = ''
        for c in out_best:
            if c < len(letters):
                outstr += letters[c]
        ret.append(outstr)
    return ret


def calculate_metrics(predictions, ground_truths):
    """Calculate comprehensive evaluation metrics"""
    total_chars = 0
    total_errors = 0
    perfect_matches = 0
    word_errors = 0
    total_words = 0

    for pred, gt in zip(predictions, ground_truths):
        pred = pred.strip()
        gt = gt.strip()

        # Character-level metrics
        if pred == gt:
            perfect_matches += 1

        total_chars += len(gt)
        char_errors = edit_distance(pred, gt)
        total_errors += char_errors

        # Word-level metrics
        pred_words = pred.split()
        gt_words = gt.split()
        total_words += len(gt_words)
        word_errors += edit_distance(pred_words, gt_words)

    cer = (total_errors / total_chars) * 100 if total_chars > 0 else 0
    wer = (word_errors / total_words) * 100 if total_words > 0 else 0
    char_accuracy = ((total_chars - total_errors) / total_chars) * 100 if total_chars > 0 else 0
    perfect_accuracy = (perfect_matches / len(predictions)) * 100 if len(predictions) > 0 else 0

    return {
        'cer': cer,
        'wer': wer,
        'char_accuracy': char_accuracy,
        'perfect_accuracy': perfect_accuracy,
        'perfect_matches': perfect_matches,
        'total_samples': len(predictions)
    }


def predict_single_image(model, image_path, char_list, show_image=False):
    """Predict text from a single image"""
    try:
        # Load and preprocess image
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            print(f"Could not load image: {image_path}")
            return None

        # Create data loader instance for preprocessing
        temp_loader = data_loader(config.DATA_PATH, 1)
        processed_img = temp_loader.preprocess(img, augment=False)
        processed_img = np.expand_dims(processed_img, 0)

        # Make prediction
        output = model.predict(processed_img, verbose=0)
        prediction = decoder(output, char_list)[0]

        if show_image:
            plt.figure(figsize=(15, 3))
            plt.imshow(processed_img[0], cmap='gray')
            plt.title(f'Prediction: {prediction}')
            plt.axis('off')
            plt.show()

        return prediction

    except Exception as e:
        print(f"Error predicting image {image_path}: {e}")
        return None


def test_on_dataset(show=True, partition='test', checkpoint="Empty", max_samples=None):
    """Test the model on the specified dataset partition"""

    print("Loading metadata...")
    training_data = data_loader(config.DATA_PATH, config.BATCH_SIZE)
    validation_data = data_loader(config.DATA_PATH, config.BATCH_SIZE)
    test_data = data_loader(config.DATA_PATH, config.BATCH_SIZE)

    training_data.trainSet()
    validation_data.validationSet()
    test_data.testSet()
    char_list = training_data.charList

    print("Loading checkpoint...")
    model = load_easter_model(checkpoint)

    if model is None:
        print("Failed to load model!")
        return

    print("Calculating results...")

    # Select partition
    if partition == 'validation':
        print("Using Validation Partition")
        data_loader_instance = validation_data
        imgs, truths, reals = validation_data.getValidationImage()
    elif partition == 'train':
        print("Using Training Partition")
        data_loader_instance = training_data
        training_data.currIdx = 0  # Reset index
        imgs, truths, reals = training_data.getValidationImage()
    else:
        print("Using Test Partition")
        data_loader_instance = test_data
        imgs, truths, reals = test_data.getTestImage()

    print(f"Number of Samples: {len(imgs)}")

    if max_samples:
        imgs = imgs[:max_samples]
        truths = truths[:max_samples]
        reals = reals[:max_samples] if reals else None
        print(f"Limited to {max_samples} samples for evaluation")

    predictions = []
    ground_truths = []

    print("Making predictions...")
    for i in tqdm(range(len(imgs))):
        try:
            img = imgs[i]
            truth = truths[i].strip()

            output = model.predict(img, verbose=0)
            prediction = decoder(output, char_list)
            predicted_text = prediction[0].strip()

            predictions.append(predicted_text)
            ground_truths.append(truth)

            if show and i < 10:  # Show first 10 examples
                print(f"\nSample {i + 1}:")
                print(f"Ground Truth: {truth}")
                print(f"Prediction:   {predicted_text}")
                print(f"CER: {(edit_distance(predicted_text, truth) / len(truth) * 100):.2f}%")
                print("-" * 80)

        except Exception as e:
            print(f"Error processing sample {i}: {e}")
            continue

    if predictions:
        # Calculate comprehensive metrics
        metrics = calculate_metrics(predictions, ground_truths)

        print("\n" + "=" * 80)
        print("EVALUATION RESULTS")
        print("=" * 80)
        print(f"Total Samples:        {metrics['total_samples']}")
        print(f"Perfect Matches:      {metrics['perfect_matches']}")
        print(f"Perfect Accuracy:     {metrics['perfect_accuracy']:.2f}%")
        print(f"Character Error Rate: {metrics['cer']:.2f}%")
        print(f"Word Error Rate:      {metrics['wer']:.2f}%")
        print(f"Character Accuracy:   {metrics['char_accuracy']:.2f}%")
        print("=" * 80)

        # Show some example errors
        print("\nExample Predictions (with errors):")
        error_count = 0
        for pred, truth in zip(predictions, ground_truths):
            if pred != truth and error_count < 5:
                cer = (edit_distance(pred, truth) / len(truth)) * 100
                print(f"\nCER: {cer:.1f}%")
                print(f"Truth: {truth}")
                print(f"Pred:  {pred}")
                error_count += 1

        return metrics
    else:
        print("No successful predictions made!")
        return None


def interactive_prediction():
    """Interactive mode for testing individual images"""
    print("Loading model...")
    model = load_easter_model("Empty")

    if model is None:
        print("Failed to load model!")
        return

    # Load character list
    temp_data = data_loader(config.DATA_PATH, 1)
    char_list = temp_data.charList

    print("Interactive Prediction Mode")
    print("Enter image path (or 'quit' to exit):")

    while True:
        image_path = input("\nImage path: ").strip()

        if image_path.lower() == 'quit':
            break

        if not image_path:
            continue

        prediction = predict_single_image(model, image_path, char_list, show_image=True)

        if prediction:
            print(f"Predicted text: {prediction}")
        else:
            print("Failed to predict text from image")


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        mode = sys.argv[1].lower()

        if mode == 'interactive':
            interactive_prediction()
        elif mode in ['test', 'validation', 'train']:
            checkpoint = sys.argv[2] if len(sys.argv) > 2 else "Empty"
            max_samples = int(sys.argv[3]) if len(sys.argv) > 3 else None
            test_on_dataset(show=True, partition=mode, checkpoint=checkpoint, max_samples=max_samples)
        else:
            print("Usage:")
            print("python predict.py interactive")
            print("python predict.py [test|validation|train] [checkpoint_path] [max_samples]")
    else:
        # Default: test on test set
        test_on_dataset(show=True, partition='test')