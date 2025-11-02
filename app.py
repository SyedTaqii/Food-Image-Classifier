import streamlit as st
import torch
from transformers import ViTForImageClassification, AutoImageProcessor
from PIL import Image
import torch.nn.functional as F

# --- 1. Load Model and Processor (Cached) ---
# This decorator is the most important part.
# It tells Streamlit to load the model only ONCE,
# making the app fast and efficient.
@st.cache_resource
def load_model():
    """
    Loads the fine-tuned model and processor from the local directory.
    """
    model_path = "syedtaqi/food-images-classifier"
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    st.info(f"Loading model onto {device.upper()}... (This only happens once)")
    
    try:
        model = ViTForImageClassification.from_pretrained(model_path)
        processor = AutoImageProcessor.from_pretrained(model_path)
    except Exception as e:
        st.error(f"Error loading model: {e}")
        st.error(
            "Fatal Error: Could not load model or processor. "
            "Please make sure the 'vit_final_food_model' folder "
            "is in the same directory as 'app.py'."
        )
        return None, None, None
        
    model.to(device)
    model.eval()  # Set model to evaluation mode
    
    st.success("Model loaded successfully!")
    return model, processor, device

# Load the model, processor, and device at startup
model, processor, device = load_model()

# --- 2. The Streamlit UI ---
st.title("🍔Accurate Food Classifier 🍕")
st.markdown(
    "This app uses a **ViT (Vision Transformer)** model fine-tuned on the **Food-101** dataset. "
    "Upload an image of food, and it will try to guess what it is!"
)

# File uploader
uploaded_file = st.file_uploader(
    "Choose a food image...", 
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    # 1. Load and display the user's image
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="You uploaded this image:", use_column_width=True)
    
    # 2. Process the image and predict
    # Show a spinner while the model is "thinking"
    with st.spinner("Classifying..."):
        # The 'processor' is our "tokenizer" for images
        # It resizes, normalizes, and converts the image to a tensor
        inputs = processor(images=image, return_tensors="pt").to(device)

        # Run the model
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits

        # --- 3. Get the results ---
        
        # Get the top prediction
        predicted_class_idx = logits.argmax(-1).item()
        
        # Get the "pretty" name (e.g., 'apple_pie' -> 'Apple Pie')
        predicted_class_name = model.config.id2label[predicted_class_idx]
        pretty_name = predicted_class_name.replace("_", " ").title()
        
        # Get the confidence score
        # Apply softmax to the logits to get probabilities
        probabilities = F.softmax(logits, dim=-1)
        confidence = probabilities[0][predicted_class_idx].item() * 100
        
        # 4. Display the results
        st.success(f"I'm {confidence:.2f}% sure this is: {pretty_name}")

# st.sidebar.header("About This Project")
# st.sidebar.info(
#     "**Task 3: Vision Transformer (ViT)**\n\n"
#     "This app fulfills the third part of the assignment.\n\n"
#     f"**Model:** `google/vit-base-patch16-224`\n"
#     f"**Dataset:** Food-101 (75k images)\n"
#     f"**Final Accuracy:** 90.6%"
# )

