Vehicle Orientation Classification

Model: EfficientNetB0 + TensorFlow Lite (pretrained on Imagenet)


-------------------------------------------------
1. PROJECT OVERVIEW
-------------------------------------------------

This project performs multi-class vehicle orientation classification.

The model predicts one of the following 6 classes:

- front
- frontleft
- frontright
- rear
- rearleft
- rearright

The final model is exported as a TensorFlow Lite (.tflite) model
for lightweight inference.

-------------------------------------------------
2. DATASET PREPARATION
-------------------------------------------------

The dataset was manually organized into 6 class folders:

dataset/
    front/
    frontleft/
    frontright/
    rear/
    rearleft/
    rearright/

Images were labeled based on dominant vehicle orientation.

Dataset was split into:
- 70% training
- 20% validation
- 10% testing

-------------------------------------------------
3. DATA AUGMENTATION AND PREPROCESSING
-------------------------------------------------

Applied during training:

- Rotation (±8 degrees)
- Zoom (0.1)
- Brightness adjustment (0.8 – 1.2)

Horizontal flipping was NOT used since it would
invert vehicle orientation (left ↔ right).

-------------------------------------------------
4. MODEL ARCHITECTURE
-------------------------------------------------

Reason for using pretrained weights:
Training EfficientNet from scratch on a limited dataset did not produce good results. Therefore, ImageNet pretrained weights were used to leverage transfer learning and improve feature extraction.

The backbone was frozen during training

Custom classification head:
- GlobalAveragePooling
- BatchNormalization
- Dropout (0.5)
- Dense(128, ReLU)
- Dropout (0.3)
- Dense(6, Softmax)

Loss Function:
- Sparse Categorical Crossentropy

Optimizer:
- Adam (learning rate = 1e-4)

Callbacks:
- EarlyStopping
- ReduceLROnPlateau

-------------------------------------------------
5. TRAINING DETAILS
-------------------------------------------------

Epochs: 50
Batch Size: 32
Input Size: 224x224

Final Performance:

Training Accuracy: ~75%
Validation Accuracy: ~55%
Test Accuracy: ~60–65%

-------------------------------------------------
6. INFERENCE PIPELINE
-------------------------------------------------

The file test_predict.py performs:

1. Load  model
2. Resize image to 224x224
3. Apply EfficientNet preprocessing
4. Run inference
5. Print predicted class and confidence

To run:

python test_predict.py image.jpg

Output example:

Prediction: frontleft
Confidence: 0.8723
 
 						-------------------------------------------------
								END OF DOCUMENT
						-------------------------------------------------

