#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.metrics import jaccard_score, f1_score

# Custom loss function
class CustomBinaryCrossentropy(tf.keras.losses.BinaryCrossentropy):
    def __init__(self, from_logits=False, label_smoothing=0.0, axis=-1, name='binary_crossentropy', **kwargs):
        super().__init__(from_logits=from_logits, label_smoothing=label_smoothing, axis=axis, name=name)

# Register the custom object
tf.keras.utils.get_custom_objects().update({
    'CustomBinaryCrossentropy': CustomBinaryCrossentropy
})

# Paths to the test B-mode images and delineation masks
bmode_images = [
    r"Z:\Uterine_segmentation\HealthyVolunteers\Bmode\CEUSPILOT030\Bmode_frame_1.jpg",
    r"Z:\Uterine_segmentation\HealthyVolunteers\Bmode\CEUSPILOT031_notHealthy\endometrium_frame1.jpg",
    r"Z:\Uterine_segmentation\HealthyVolunteers\Bmode\UV012\endometrium_frame1.jpg",
    r"Z:\Uterine_segmentation\AdenomyosisPatients\Bmode\UV033\Bmode_frame_1.jpg",
    r"Z:\Uterine_segmentation\AdenomyosisPatients\Bmode\UV021\Bmode_frame_1.jpg",
    r"Z:\Uterine_segmentation\AdenomyosisPatients\Bmode\UV019\Bmode_frame_1.jpg"
]

delineation_masks = [
    r"Z:\Uterine_segmentation\HealthyVolunteers\Delineations\CEUSPILOT030\Bmode_frame_1.jpg",
    r"Z:\Uterine_segmentation\HealthyVolunteers\Delineations\CEUSPILOT031_notHealthy\endometrium_frame1.jpg",
    r"Z:\Uterine_segmentation\HealthyVolunteers\Delineations\UV012\endometrium_frame1.jpg",
    r"Z:\Uterine_segmentation\AdenomyosisPatients\Delineations\UV033\Bmode_frame_1.jpg",
    r"Z:\Uterine_segmentation\AdenomyosisPatients\Delineations\UV021\Bmode_frame_1437.jpg",
    r"Z:\Uterine_segmentation\AdenomyosisPatients\Delineations\UV019\Bmode_frame_1.jpg"
]

filenames = [
    "CEUSPILOT030_Bmode_frame_1",
    "CEUSPILOT031_notHealthy_endometrium_frame1",
    "UV012_endometrium_frame1",
    "UV033_Bmode_frame_1",
    "UV021_Bmode_frame_1",
    "UV019_Bmode_frame_1"
]

# Output folder for saving the processed frames and masks
output_folder = r"Z:\Uterine_segmentation\Auto_segmentation_Mohammad\output image from scratch"
output_display_folder = r"Z:\Uterine_segmentation\Auto_segmentation_Mohammad\six frames"

# Create the output folders if they do not exist
if not os.path.exists(output_folder):
    os.makedirs(output_folder)
if not os.path.exists(output_display_folder):
    os.makedirs(output_display_folder)

# Function to load images and masks
def load_images_and_masks(img_paths, mask_paths, img_size=(256, 256)):
    images = []
    masks = []
    for img_path, mask_path in zip(img_paths, mask_paths):
        img = load_img(img_path, target_size=img_size, color_mode="grayscale")
        img = img_to_array(img) / 255.0  # Normalize to [0, 1]
        
        mask = load_img(mask_path, target_size=img_size, color_mode="grayscale")
        mask = img_to_array(mask) / 255.0  # Normalize to [0, 1]
        
        images.append(img)
        masks.append(mask)
        
    return np.array(images), np.array(masks)

# Load the images and masks
images, masks = load_images_and_masks(bmode_images, delineation_masks)

# Print shapes of loaded images and masks for debugging
print("Images shape:", images.shape)
print("Masks shape:", masks.shape)

# Check for empty arrays
if images.size == 0 or masks.size == 0:
    raise ValueError("Loaded images or masks array is empty. Please check the paths and ensure there are images in the specified directories.")

# Function to perform prediction
def predict(model, test_images):
    predictions = model.predict(test_images)
    return predictions

# Normalize images and masks
def min_max_normalize(X, Y):
    if X.size == 0 or Y.size == 0:
        raise ValueError("Input arrays are empty. Cannot normalize.")
    X = (X - np.min(X)) / (np.max(X) - np.min(X))
    Y = (Y - np.min(Y)) / (np.max(Y) - np.min(Y))
    Y = (Y > 0.5).astype(np.bool_)
    return X, Y

# Calculate metrics: IoU and Dice coefficient
def calculate_metrics(predictions, ground_truth):
    predictions_bin = (predictions > 0.5).astype(np.uint8).flatten()
    ground_truth_bin = (ground_truth > 0.5).astype(np.uint8).flatten()
    iou = jaccard_score(ground_truth_bin, predictions_bin, average='binary')
    dice = f1_score(ground_truth_bin, predictions_bin, average='binary')
    return iou, dice

# Function to display images and their corresponding masks
def display_images_with_metrics(images, masks, predictions, filenames, iou_scores, dice_scores, output_folder):
    plt.figure(figsize=(20, len(images) * 6))
    for i in range(len(images)):
        # Original image
        plt.subplot(len(images), 4, i * 4 + 1)
        plt.imshow(images[i].squeeze(), cmap='gray')
        plt.title(f"Original Image ({filenames[i]})")
        plt.axis("off")
        
        # Ground truth mask
        plt.subplot(len(images), 4, i * 4 + 2)
        plt.imshow(masks[i].squeeze(), cmap='gray')
        plt.title(f"Ground Truth Mask ({filenames[i]})")
        plt.axis("off")
        
        # Predicted mask
        plt.subplot(len(images), 4, i * 4 + 3)
        plt.imshow(predictions[i].squeeze(), cmap='gray')
        plt.title(f"Predicted Mask ({filenames[i]})")
        plt.axis("off")
        
        # IoU and Dice scores
        plt.subplot(len(images), 4, i * 4 + 4)
        plt.text(0.5, 0.5, f"IoU: {iou_scores[i]:.4f}\nDice: {dice_scores[i]:.4f}", 
                 horizontalalignment='center', verticalalignment='center', fontsize=12)
        plt.title("Metrics")
        plt.axis("off")
        
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, "results.png"))
    plt.show()

if __name__ == "__main__":
    # Preprocess test images and masks
    test_images, test_masks = images, masks

    # Check for empty arrays
    if test_images.size == 0 or test_masks.size == 0:
        raise ValueError("Preprocessed images or masks array is empty. Please check the preprocessing steps.")

    # Normalize images and masks
    test_images, test_masks = min_max_normalize(test_images, test_masks)

    # Load the trained model without compiling
    model_path = r'Z:\Uterine_segmentation\Auto_segmentation_Mohammad\unet_model.h5'
    model = load_model(model_path, compile=False, custom_objects={'CustomBinaryCrossentropy': CustomBinaryCrossentropy})

    # Compile the model again with the custom loss function
    model.compile(optimizer='adam', loss=CustomBinaryCrossentropy(), metrics=['accuracy'])

    # Perform prediction
    predictions = predict(model, test_images)

    # Threshold the predictions
    predictions = (predictions > 0.5).astype(np.uint8)

    # Print unique values in predictions to check if it's predicting correctly
    print("Unique values in predictions:", np.unique(predictions))

    iou_scores = []
    dice_scores = []

    # Convert predictions to binary masks
    predicted_masks = (predictions > 0.5).astype(np.uint8)

    # Calculate metrics for each image
    for i in range(len(test_masks)):
        iou, dice = calculate_metrics(predicted_masks[i], test_masks[i])
        iou_scores.append(iou)
        dice_scores.append(dice)

        print(f"Image {filenames[i]} - IoU: {iou:.4f}, Dice: {dice:.4f}")

        # Save preprocessed frame
        frame_image = Image.fromarray((test_images[i] * 255).astype(np.uint8).squeeze(), mode='L')  # Convert back to uint8 for saving
        frame_filename = f'{filenames[i]}_preprocessed.jpg'
        frame_image.save(os.path.join(output_folder, frame_filename))
        
        # Save mask
        mask_image = Image.fromarray((predicted_masks[i] * 255).astype(np.uint8).squeeze(), mode='L')  # mode='L' for grayscale
        mask_filename = f'{filenames[i]}_mask.jpg'
        mask_image.save(os.path.join(output_folder, mask_filename))

    # Print average IoU and Dice scores
    print(f"Average IoU: {np.mean(iou_scores):.4f}")
    print(f"Average Dice: {np.mean(dice_scores):.4f}")

    # Display the images with their ground truth and predicted masks
    display_images_with_metrics(test_images, test_masks, predicted_masks, filenames, iou_scores, dice_scores, output_display_folder)


# In[ ]:




