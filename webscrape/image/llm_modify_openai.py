import os
import time
import base64
import cv2
import numpy as np
from openai import OpenAI

# Initialize the OpenAI client (Automatically picks up OPENAI_API_KEY from environment)
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


def get_modification_percentage(original_path, edited_path):
    """
    Compares the original and edited images using OpenCV.
    Returns the percentage of pixels that were modified.
    """
    img1 = cv2.imread(original_path)
    img2 = cv2.imread(edited_path)
    
    if img1 is None or img2 is None:
        raise ValueError("Could not load images for OpenCV comparison.")

    # Ensure dimensions match before comparison
    if img1.shape != img2.shape:
        img2 = cv2.resize(img2, (img1.shape[1], img1.shape[0]))

    # Calculate absolute difference
    diff = cv2.absdiff(img1, img2)
    gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
    
    # Filter out minor compression artifacts/noise (Threshold set to 30)
    _, thresh = cv2.threshold(gray, 30, 255, cv2.THRESH_BINARY)
    
    # Calculate Modification Percentage
    total_pixels = thresh.shape[0] * thresh.shape[1]
    changed_pixels = cv2.countNonZero(thresh)
    mod_percentage = (changed_pixels / total_pixels) * 100
    
    return mod_percentage

def batch_edit_images(input_dir, prompt, output_dir, max_retries=5, target_percentage=50.0):
    """
    Iterates through a folder, sends each image to the API.
    Retries if the modification percentage is below the target threshold.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for filename in os.listdir(input_dir):
        if not filename.lower().endswith((".png", ".jpg", ".jpeg", ".webp")):
            continue
            
        print(f"\nProcessing '{filename}'...")
        input_path = os.path.join(input_dir, filename)
        output_path = os.path.join(output_dir, f"edited_{filename}")

        # --- RETRY LOOP ---
        for attempt in range(max_retries):
            print(f"  Attempt {attempt + 1} of {max_retries}...")
            try:
                # 1. Call the API
                with open(input_path, "rb") as image_file:
                    result = client.images.edit(
                        model="gpt-image-2-2026-04-21", 
                        image=image_file,
                        prompt=prompt,
                    )

                # 2. Extract and decode the new image
                image_base64 = result.data[0].b64_json
                image_bytes = base64.b64decode(image_base64)

                # 3. Save the image to a file temporarily to check it
                with open(output_path, "wb") as f:
                    f.write(image_bytes)
                
                # 4. Check the modification percentage
                mod_percentage = get_modification_percentage(input_path, output_path)
                
                if mod_percentage >= target_percentage:
                    print(f"  Success! Image modified by {mod_percentage:.2f}% (Saved to: {output_path})")
                    print("-" * 50)
                    break # Break out of the retry loop on success
                else:
                    print(f"  Result rejected: Only {mod_percentage:.2f}% modified (Target: >= {target_percentage}%).")
                    if attempt < max_retries - 1:
                        print("  Retrying in 2 seconds...")
                        time.sleep(2)
                    else:
                        print(f"  Failed to reach {target_percentage}% modification for {filename} after max retries.")
                        print("-" * 50)

            except Exception as e:
                print(f"  Error on attempt {attempt + 1}: {e}")
                if attempt < max_retries - 1:
                    print("  Retrying in 3 seconds...")
                    time.sleep(3)
                else:
                    print(f"  Failed processing {filename} due to persistent errors.")
                    print("-" * 50)

# --- Execute Pipeline ---
if __name__ == "__main__":
    INPUT_FOLDER = "/projects/p32143/RL_human_decision/webscrape/image/temp"
    OUTPUT_FOLDER = "/projects/p32143/RL_human_decision/webscrape/image/temp_edited"
    
    INSTRUCTION = """
Analyze this image and modify at least 50% of its content. Ensure the modifications—whether they are replacements, additions, or deletions—appear as realistic as possible.
"""
    
    # Notice I bumped max_retries to 5, as hitting exactly >50% without changing the background might take the model a few tries.
    batch_edit_images(INPUT_FOLDER, INSTRUCTION, OUTPUT_FOLDER, max_retries=5, target_percentage=50.0)