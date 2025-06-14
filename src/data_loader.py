# src/data_loader.py

import pandas as pd
import numpy as np
import cv2
import random
import itertools
import os
import time
import json
import glob
from sklearn.model_selection import train_test_split
import config
import matplotlib.pyplot as plt
from tacobox import Taco
import tensorflow as tf


class Sample:
    """Sample from the dataset"""

    def __init__(self, gtText, filePath, sample_id):
        self.gtText = gtText
        self.filePath = filePath
        self.sample_id = sample_id


class data_loader:
    def __init__(self, path, batch_size):
        self.batchSize = batch_size
        self.samples = []
        self.currIdx = 0
        self.charList = []

        # Creating taco object for augmentation (checkout Easter2.0 paper)
        self.mytaco = Taco(
            cp_vertical=0.2,
            cp_horizontal=0.25,
            max_tw_vertical=100,
            min_tw_vertical=10,
            max_tw_horizontal=50,
            min_tw_horizontal=10
        )

        # Load samples from synthetic dataset
        self._load_synthetic_dataset(path)

        # Split dataset
        self._split_dataset()

        # Set default to training set
        self.trainSet()

    def _load_synthetic_dataset(self, path):
        """Load samples from synthetic Sharada manuscript dataset"""
        chars = set()

        # Find all JSON files in the dataset
        json_files = glob.glob(os.path.join(path, "**/*.json"), recursive=True)

        print(f"Found {len(json_files)} JSON files")

        for json_file in json_files:
            try:
                with open(json_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)

                sample_id = data['id']
                original_text = data.get('original_text', '')

                # Construct image path
                img_path = os.path.join(os.path.dirname(json_file), f"{sample_id}.png")

                # Check if image exists and text is not empty
                if os.path.exists(img_path) and original_text:
                    chars = chars.union(set(list(original_text)))
                    self.samples.append(Sample(original_text, img_path, sample_id))
                else:
                    if not os.path.exists(img_path):
                        print(f"Warning: Image not found for {sample_id}")
                    if not original_text:
                        print(f"Warning: Empty 'original_text' for {sample_id}. Skipping sample.")

            except Exception as e:
                print(f"Error loading {json_file}: {e}")

        # Create character list
        self.charList = sorted(list(chars))
        print(f"Total samples loaded: {len(self.samples)}")
        print(f"Unique characters: {len(self.charList)}")
        print(f"Character set: {''.join(self.charList[:50])}...")  # Show first 50 chars

    def _split_dataset(self):
        """Split dataset into train, validation, and test sets"""
        # First split: separate train from temp (validation + test)
        train_samples, temp_samples = train_test_split(
            self.samples,
            test_size=(config.VALIDATION_SPLIT + config.TEST_SPLIT),
            random_state=42
        )

        # Second split: separate validation from test
        val_size = config.VALIDATION_SPLIT / (config.VALIDATION_SPLIT + config.TEST_SPLIT)
        val_samples, test_samples = train_test_split(
            temp_samples,
            test_size=(1 - val_size),
            random_state=42
        )

        self.trainSamples = train_samples
        self.validationSamples = val_samples
        self.testSamples = test_samples

        print(f"Dataset split - Train: {len(self.trainSamples)}, "
              f"Validation: {len(self.validationSamples)}, "
              f"Test: {len(self.testSamples)}")

    def trainSet(self):
        self.currIdx = 0
        random.shuffle(self.trainSamples)
        self.samples = self.trainSamples

    def validationSet(self):
        self.currIdx = 0
        self.samples = self.validationSamples

    def testSet(self):
        self.currIdx = 0
        self.samples = self.testSamples

    def getNext(self, what='train'):
        """Generator for training batches"""
        num_batches = len(self.samples) // self.batchSize
        for i in range(num_batches):
            batch_samples = self.samples[i * self.batchSize:(i + 1) * self.batchSize]

            gtTexts = np.ones([self.batchSize, config.OUTPUT_SHAPE], dtype=np.int32) * len(self.charList)
            # input_length should be the sequence length of the model's output (y_pred)
            # For Easter2 model: Input width 2304 -> s=2 -> 1152 -> s=2 -> 576. Stays 576 after that.
            actual_model_output_seq_len = config.INPUT_WIDTH // 4
            input_length = np.full((self.batchSize,), actual_model_output_seq_len, dtype=np.int32)
            label_length = np.zeros((self.batchSize, 1), dtype=np.int32)
            # Initialize imgs with shape (batch_size, INPUT_WIDTH, INPUT_HEIGHT)
            imgs = np.zeros([self.batchSize, config.INPUT_WIDTH, config.INPUT_HEIGHT], dtype=np.float32)

            for j, sample in enumerate(batch_samples):
                try:
                    img = cv2.imread(sample.filePath, cv2.IMREAD_GRAYSCALE)
                    text = sample.gtText

                    # Preprocess image
                    augment = (what == 'train')
                    img = self.preprocess(img, augment=augment)
                    imgs[j] = img

                    # Encode text
                    val = [self.charList.index(char) for char in text if char in self.charList]

                    # Pad sequence
                    if len(val) > config.OUTPUT_SHAPE:
                        val = val[:config.OUTPUT_SHAPE]

                    gtTexts[j, :len(val)] = val
                    label_length[j] = len(val)

                except Exception as e:
                    print(f"Error processing sample {sample.filePath}: {e}")
                    # Create dummy data with shape (INPUT_WIDTH, INPUT_HEIGHT)
                    imgs[j] = np.zeros((config.INPUT_WIDTH, config.INPUT_HEIGHT))
                    gtTexts[j] = np.ones(config.OUTPUT_SHAPE) * len(self.charList)
                    gtTexts[j, 0] = 0  # Dummy label with a valid character index
                    label_length[j] = 1

            inputs = {
                'the_input': imgs,
                'the_labels': gtTexts,
                'input_length': input_length,
                'label_length': label_length,
            }
            # Simple array of zeros as dummy targets for the CTC loss
            outputs = np.zeros([self.batchSize])
            yield (inputs, outputs)

    def getValidationImage(self):
        """Get all validation images for evaluation"""
        batchRange = range(0, min(config.CER_EVAL_SAMPLES, len(self.samples)))
        imgs = []
        texts = []
        reals = []

        for i in batchRange:
            try:
                img1 = cv2.imread(self.samples[i].filePath, cv2.IMREAD_GRAYSCALE)
                real = cv2.imread(self.samples[i].filePath)

                img = self.preprocess(img1, augment=False)
                img = np.expand_dims(img, 0)
                text = self.samples[i].gtText

                imgs.append(img)
                texts.append(text)
                reals.append(real)
            except Exception as e:
                print(f"Error processing validation sample {i}: {e}")
                continue

        return imgs, texts, reals

    def getTestImage(self):
        """Get all test images for evaluation"""
        batchRange = range(0, len(self.samples))
        imgs = []
        texts = []
        reals = []

        for i in batchRange:
            try:
                img1 = cv2.imread(self.samples[i].filePath, cv2.IMREAD_GRAYSCALE)
                real = cv2.imread(self.samples[i].filePath)

                img = self.preprocess(img1, augment=False)
                img = np.expand_dims(img, 0)
                text = self.samples[i].gtText

                imgs.append(img)
                texts.append(text)
                reals.append(real)
            except Exception as e:
                print(f"Error processing test sample {i}: {e}")
                continue

        return imgs, texts, reals

    def preprocess(self, img, augment=True):
        if img is None:
            # Initialize with (Height, Width) as cv2.resize output this format
            img = np.zeros((config.INPUT_HEIGHT, config.INPUT_WIDTH), dtype=np.uint8)

        if augment and len(self.samples) > 3000:  # Only augment on training set
            img = self.apply_taco_augmentations(img)

        # Scaling image [0, 1]
        img = img.astype(np.float32) / 255.0

        # Resize to target dimensions if needed.
        # cv2.resize takes (width, height) for dsize, and returns an array of shape (height, width).
        # So, after this step, img.shape will be (config.INPUT_HEIGHT, config.INPUT_WIDTH).
        if img.shape[0] != config.INPUT_HEIGHT or img.shape[1] != config.INPUT_WIDTH:
            img = cv2.resize(img, (config.INPUT_WIDTH, config.INPUT_HEIGHT))
        
        # Transpose from (INPUT_HEIGHT, INPUT_WIDTH) to (INPUT_WIDTH, INPUT_HEIGHT)
        # This matches the model's expected input_shape = (INPUT_WIDTH, INPUT_HEIGHT)
        img = img.transpose((1, 0))

        # Invert image (black text on white background -> white text on black background)
        img = 1.0 - img

        # Ensure no extra channel dimension is present
        return img

    def apply_taco_augmentations(self, input_img):
        """Apply TACO augmentations"""
        random_value = random.random()
        if random_value <= config.TACO_AUGMENTAION_FRACTION:
            augmented_img = self.mytaco.apply_vertical_taco(
                input_img,
                corruption_type='random'
            )
        else:
            augmented_img = input_img
        return augmented_img