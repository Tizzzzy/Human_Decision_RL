import os
from google import genai
from PIL import Image

# Initialize the Gemini client (automatically picks up GEMINI_API_KEY from environment)
client = genai.Client(api_key="AIzaSyCpLt9fECSRhiH3c3k-XxarU5HL94yuja8")

def process_image_folder_with_gemini(input_folder, prompt, output_folder):
    """
    Iterates through a folder, calls the Gemini API to edit the image, 
    and captures both the new image and the text log.
    """
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for filename in os.listdir(input_folder):
        if not filename.lower().endswith(('.png', '.jpg', '.jpeg', '.webp')):
            continue
            
        print(f"\nProcessing '{filename}'...")
        input_path = os.path.join(input_folder, filename)
        output_path = os.path.join(output_folder, f"edited_{filename}")

        try:
            # 1. Load the original image using Pillow
            original_image = Image.open(input_path)
            
            # 2. Call the Gemini multimodal model
            # Note: You can also use "gemini-3-pro-image-preview" for higher fidelity
            response = client.models.generate_content(
                model="gemini-3.1-flash-image-preview", 
                contents=[prompt, original_image],
            )
            
            # 3. Parse the response parts to extract both text and image
            text_log = ""
            image_saved = False
            
            for part in response.parts:
                # Capture the text explanation
                if part.text is not None:
                    text_log += part.text + "\n"
                
                # Capture the newly generated image data
                elif part.inline_data is not None:
                    edited_image = part.as_image()
                    edited_image.save(output_path)
                    image_saved = True
            
            # 4. Print the results
            if image_saved:
                print(f"Success! Image saved to: {output_path}")
            else:
                print("Warning: No image data was returned.")
                
            print(f"Gemini Edit Log:\n{text_log.strip()}")
            print("-" * 50)
            
        except Exception as e:
            print(f"Error processing {filename}: {e}")

# --- Execute Pipeline ---
if __name__ == "__main__":
    INPUT_FOLDER = "/projects/p32143/RL_human_decision/webscrape/image/temp"
    OUTPUT_FOLDER = "/projects/p32143/RL_human_decision/webscrape/image/temp_edited"
    
    # Instruct the model on the edit AND ask for the text report
    INSTRUCTION = """Apply at least FIVE distinct modifications based on this image.

CRITICAL OUTPUT INSTRUCTIONS:
Output your modified image and a bulleted list describing what the viewer should look for to spot the changes. Use this exact syntax for every item:

- Look for the [brief description of the new or altered element]
- Look for the [brief description of the new or altered element]
"""
    
    process_image_folder_with_gemini(INPUT_FOLDER, INSTRUCTION, OUTPUT_FOLDER)