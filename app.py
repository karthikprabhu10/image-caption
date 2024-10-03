import streamlit as st
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration, AutoTokenizer

# Set page configuration
st.set_page_config(page_title="Image Captioning", layout="centered")

# Apply custom CSS for styling
st.markdown(
    """
    <style>
    /* General Styling */
    body {
        background-color: #0E1117;
    }
    h1 {
        color: white;
        font-family: 'Arial', sans-serif;
        text-align: center;
        font-size: 3em;
    }
    .stFileUploader {
        background-color: #262626;
        border: 2px dashed #444444;
        padding: 20px;
        border-radius: 10px;
    }
    .grd-text{
    font-family: Arial;
    font-weight: bolder;
    background: #b7fdb7;
    background: linear-gradient(to left, #aa00ff 0%, #00eaff 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    }
    .stButton > button {
        background-color: #1f2937;
        border: 1px solid #444444;
        color: white;
        padding: 8px 20px;
        font-size: 1em;
        border-radius: 5px;
    }
    .stButton > button:hover {
        background-color: #3b4252;
        border-color: #3b4252;
    }
    footer {
        color: white;
        text-align: center;
        padding: 20px 0;
        font-family: Arial, sans-serif;
        box-shadow: 0px -4px 8px rgba(0, 0, 0, 0.2);
    }
    .footer-container {
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
        gap: 10px;
    }
    .footer-name {
        font-size: 1.2em;
        color: white;
        margin: 0;
    }
    .social-icons {
        display: flex;
        justify-content: center;
        gap: 15px;
    }
    .social-icon {
        width: 25px;
        height: 25px;
        transition: all 0.3s ease;
    }
    .social-icon:hover {
        transform: scale(1.2);
        filter: brightness(1.5);
    }
    /* Responsive Styling */
    @media only screen and (max-width: 768px) {
        h1 {
            font-size: 2em;
        }
        .stButton > button {
            font-size: 0.9em;
            padding: 6px 16px;
        }
    }
    @media only screen and (max-width: 480px) {
        .social-icons {
            gap: 10px;
        }
        .social-icon {
            width: 20px;
            height: 20px;
        }
    }
    </style>
    """, unsafe_allow_html=True
)

# Load the BLIP model and processor
@st.cache_resource()
def load_model():
    tokenizer = AutoTokenizer.from_pretrained("Salesforce/blip-image-captioning-base", clean_up_tokenization_spaces=False)
    model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    return model, processor

model, processor = load_model()

# Function to generate caption
def generate_caption(image_input):
    inputs = processor(image_input, return_tensors="pt")
    outputs = model.generate(**inputs)
    caption = processor.decode(outputs[0], skip_special_tokens=True)
    return caption

# Main function to handle the app
def main():
    st.markdown("<h1>AI Image Captioning</h1>", unsafe_allow_html=True)

    # Create an area for file upload
    uploaded_image = st.file_uploader("Drag and drop file here", type=["jpg", "jpeg", "png"])

    if uploaded_image is not None:
        try:
            # Convert the uploaded file to a PIL Image
            image = Image.open(uploaded_image).convert("RGB")
            
            # Display the uploaded image
            st.image(image, caption="Uploaded Image", use_column_width=True)

            # Generate and display the caption
            st.write("Generating caption...")
            caption = generate_caption(image)
            st.markdown(f"**Caption:** {caption}")

        except Exception as e:
            st.error(f"Error processing image: {e}")

    # Adding developer credits in the footer with social media icons
    st.markdown(
        """
        <footer>
            <div class="footer-container">
                <p class="footer-name">Developed by Karthik Prabhu</p>
                <div class="social-icons">
                    <a href="https://www.linkedin.com/in/karthikprabhu010/" target="_blank">
                        <img class="social-icon" src="https://upload.wikimedia.org/wikipedia/commons/c/ca/LinkedIn_logo_initials.png" alt="LinkedIn">
                    </a>
                    <a href="https://www.instagram.com/karthik10.__" target="_blank">
                        <img class="social-icon" src="https://upload.wikimedia.org/wikipedia/commons/a/a5/Instagram_icon.png" alt="Instagram">
                    </a>
                </div>
            </div>
        </footer>
        """,
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()
