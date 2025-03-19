import os
import json
from PIL import Image
import google.generativeai as genai

# Working directory path
working_dir = os.path.dirname(os.path.abspath(__file__))

# Path of config_data file
config_file_path = f"{working_dir}/config.json"
config_data = json.load(open(config_file_path))

# Loading the GOOGLE_API_KEY
GOOGLE_API_KEY = config_data["GOOGLE_API_KEY"]

# Configuring google.generativeai with API key
genai.configure(api_key=GOOGLE_API_KEY)


# List available models for debugging purposes
def list_available_models():
    models = genai.list_models()
    print("\nAvailable Models:")
    for model in models:
        print(model.name)


# Load Gemini 1.5 Pro model
def load_gemini_pro_model():
    gemini_pro_model = genai.GenerativeModel("gemini-1.5-pro")
    return gemini_pro_model


# Gemini 1.5 Pro Vision response - image/text to text
def gemini_pro_vision_response(prompt, image):
    try:
        gemini_pro_vision_model = genai.GenerativeModel("gemini-1.5-pro-vision")
        response = gemini_pro_vision_model.generate_content([prompt, image])
        return response.text
    except Exception as e:
        return f"Error: {str(e)}"


# Gemini 1.5 Pro response - text to text
def gemini_pro_response(user_prompt):
    try:
        gemini_pro_model = load_gemini_pro_model()
        response = gemini_pro_model.generate_content(user_prompt)
        return response.text
    except Exception as e:
        return f"Error: {str(e)}"


# Text Embeddings Model response - text to embeddings
def embeddings_model_response(input_text):
    try:
        embedding_model = "models/embedding-001"
        embedding = genai.embed_content(
            model=embedding_model,
            content=input_text,
            task_type="retrieval_document"
        )
        return embedding["embedding"]
    except Exception as e:
        return f"Error: {str(e)}"


# Uncomment to list available models for debugging
# list_available_models()
