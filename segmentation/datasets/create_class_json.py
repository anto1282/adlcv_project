import os
import json
from PIL import Image
import numpy as np
from tqdm import tqdm  # Optional: for progress bar

def extract_classes_from_masks(gt_folder, output_json_path, mask_extension=".png", original_img_extension=".jpg"):
    """
    Iterates through semantic segmentation masks in a folder, extracts unique
    class IDs present in each mask, and saves the information to a JSON file.

    Args:
        gt_folder (str): Path to the folder containing the ground truth masks (e.g., PNG files).
        output_json_path (str): Path where the output JSON file will be saved.
        mask_extension (str): The file extension of the mask files (default: ".png").
        original_img_extension (str): The file extension of the original images
                                      corresponding to the masks (default: ".jpg").
    """
    class_data = {}
    
    print(f"Scanning folder: {gt_folder}")
    
    # List all files in the ground truth folder
    try:
        all_files = os.listdir(gt_folder)
    except FileNotFoundError:
        print(f"Error: Folder not found at {gt_folder}")
        return
    except Exception as e:
        print(f"Error listing files in {gt_folder}: {e}")
        return
        
    mask_files = [f for f in all_files if f.lower().endswith(mask_extension)]
    
    if not mask_files:
        print(f"No files with extension '{mask_extension}' found in {gt_folder}.")
        return

    print(f"Found {len(mask_files)} mask files. Processing...")

    # Iterate through each mask file with a progress bar
    for filename in tqdm(mask_files, desc="Processing masks"):
        mask_path = os.path.join(gt_folder, filename)
        
        try:
            # Open the mask image using Pillow
            # Ensure it's opened in grayscale ('L' mode) to get single-channel values
            with Image.open(mask_path) as img:
                # Convert image data to a NumPy array
                mask_array = np.array(img)
            
            # Find unique non-zero pixel values (class IDs)
            # If you also want to include the background class (often 0), remove the condition > 0
            unique_classes = np.unique(mask_array) 
            
            # Convert NumPy integers (like uint8) to standard Python int for JSON serialization
            class_list = [int(cls) for cls in unique_classes]
            
            # --- Determine the corresponding original image filename ---
            # Get the filename without the mask extension
            base_name = os.path.splitext(filename)[0]
            # Append the original image extension
            original_filename = base_name + original_img_extension
            
            # Store the data
            class_data[original_filename] = class_list

        except FileNotFoundError:
            print(f"\nWarning: File not found during processing: {mask_path}")
        except Exception as e:
            print(f"\nError processing file {filename}: {e}")
            # Optionally skip this file or handle the error differently

    # Save the collected data to a JSON file
    print(f"\nSaving class information to: {output_json_path}")
    try:
        with open(output_json_path, 'w') as f:
            json.dump(class_data, f, indent=4) # Use indent for nice formatting
        print("JSON file saved successfully.")
    except Exception as e:
        print(f"Error writing JSON file: {e}")

# --- Configuration ---
# !!! UPDATE THESE PATHS !!!
GT_ANNOTATION_FOLDER = "/work3/s203520/advanced_computer_vision/filtered_dataset/annotations/training/"
OUTPUT_JSON_FILE = "training_class_info.json" 

# Optional: Change if your masks or original images have different extensions
MASK_FILE_EXTENSION = ".png" 
ORIGINAL_IMAGE_EXTENSION = ".jpg"

# --- Run the function ---
if __name__ == "__main__":
    extract_classes_from_masks(
        GT_ANNOTATION_FOLDER, 
        OUTPUT_JSON_FILE,
        mask_extension=MASK_FILE_EXTENSION,
        original_img_extension=ORIGINAL_IMAGE_EXTENSION
    )