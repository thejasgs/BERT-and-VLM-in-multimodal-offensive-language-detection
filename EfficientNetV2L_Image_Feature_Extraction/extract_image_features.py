import numpy as np
import os
from tqdm import tqdm
from tensorflow.keras.applications import EfficientNetV2L, efficientnet
from tensorflow.keras.preprocessing import image as keras_image

# EfficientNetV2L expects input images of size 380x380
IMG_SIZE = (380, 380)

def preprocess_and_save_features(IMG_DIR, image_paths, save_path, batch_size=32):
    """
    Processes images in batches and saves features to a compressed .npz file.
    Inputs:
        - IMG_DIR: Directory where images are stored
        - image_paths: List of image file paths (relative to IMG_DIR)
        - save_path: Path to save the .npz file
        - batch_size: Number of images to process in each batch
     Outputs:
        - Final_features_array: Numpy array of extracted features
    """
    all_features = []
    
    # Utilize pre-trained EfficientNetV2L model for feature extraction
    img_model = EfficientNetV2L (
        include_top=False, 
        weights='imagenet',
        pooling='avg',
        input_shape=(IMG_SIZE+(3,))  # RGB images
    )
        
    for i in tqdm(range(0, len(image_paths), batch_size), desc="Extracting Features"):
        batch_paths = image_paths[i:i+batch_size]
        batch_images = []
        
        for path in batch_paths:
            try:
                # Construct full path
                full_path = os.path.join(IMG_DIR, path)
                img = keras_image.load_img(full_path, target_size=IMG_SIZE)
                img_array = keras_image.img_to_array(img)
                batch_images.append(img_array)
            except Exception as e:
                # Fallback for missing images: match the expected input shape
                # Ensure IMG_SIZE is a tuple, e.g., (380, 380)
                batch_images.append(np.zeros((*IMG_SIZE, 3)))
        
        # Convert list to array and preprocess for EfficientNet
        batch_array = np.array(batch_images)
        batch_array = efficientnet.preprocess_input(batch_array)
        
        # Extract features
        features = img_model.predict(batch_array, verbose=0)
        all_features.extend(features)
        
        # Memory cleanup
        del batch_images, batch_array

    # Convert full list to a single numpy matrix
    final_features_array = np.array(all_features)
    
    # Save as compressed .npz
    np.savez_compressed(
        save_path, 
        features=final_features_array, 
        paths=np.array(image_paths)
    )
    del img_model
    print(f"\nSuccessfully saved features to {save_path}")
    return final_features_array

# Usage example:
# features = preprocess_and_save_features(IMG_DIR, df_all['image'], "harmemeC_data.npz")