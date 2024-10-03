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
    .profile-card-container {
        display: flex;
        justify-content: space-around;
        flex-wrap: wrap;
        gap: 20px;
        margin: 20px 0;
    }
    .profile-card {
        background-color: #2E3440;
        width: 150px;
        height: 180px;
        padding: 10px;
        border-radius: 10px;
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.2);
        transition: all 0.3s ease;
        text-align: center;
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
    }
    .profile-card:hover {
        box-shadow: 0 8px 16px rgba(0, 0, 0, 0.3);
    }
    .profile-pic {
        width: 80px;
        height: 80px;
        border-radius: 50%;
        margin-bottom: 5px;
        object-fit: cover;
        border: 3px solid #444444;
    }
    .profile-name {
        color: white;
        font-size: 1em;
        margin-bottom: 5px;
    }
    .profile-student-id {
        color: #A0A0A0;
        font-size: 0.8em;
        margin-bottom: 5px;
    }
    .linkedin-icon {
        width: 25px;
        height: 25px;
        transition: all 0.3s ease;
    }
    .linkedin-icon:hover {
        transform: scale(1.2);
        filter: brightness(1.5);
    }

    /* Responsive Styling */
    @media only screen and (max-width: 768px) {
        .profile-card-container {
            flex-wrap: wrap;
            justify-content: space-between; /* Two cards per row */
        }

        .profile-card {
            width: 45%; /* Two cards in one row with some gap */
            margin-bottom: 20px;
        }

        h1 {
            font-size: 2em; /* Adjust heading size for smaller screens */
        }

        .stButton > button {
            font-size: 0.9em;
            padding: 6px 16px;
        }

        .profile-pic {
            width: 60px;
            height: 60px;
        }

        .profile-name {
            font-size: 0.9em;
        }

        .profile-student-id {
            font-size: 0.7em;
        }

        .linkedin-icon {
            width: 20px;
            height: 20px;
        }
    }

    @media only screen and (max-width: 480px) {
        .profile-card {
            width: 100%; /* Full width for one card per row on small screens */
        }

        .profile-pic {
            width: 50px;
            height: 50px;
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

    # Adding developer credits in the footer with profile cards
    st.markdown(
        """
        <footer>
        <center><h2 class="grd-text">Developed By</h2></center>
            <div class="profile-card-container">
                <div class="profile-card">
                    <img class="profile-pic" src="https://i.imgur.com/rYwkUxY.jpeg">
                    <div class="profile-name">Karthik Prabhu</div>
                    <div class="profile-student-id">Student ID: 22BBTCS135</div>
                    <a href="https://www.linkedin.com/in/karthikprabhu010/" target="_blank">
                        <img class="linkedin-icon" src="https://upload.wikimedia.org/wikipedia/commons/c/ca/LinkedIn_logo_initials.png">
                    </a>
                </div>
                <div class="profile-card">
                    <img class="profile-pic" src="https://via.placeholder.com/80">
                    <div class="profile-name">Kruthika K</div>
                    <div class="profile-student-id">Student ID: 22BBTCS150</div>
                    <a href="https://www.linkedin.com/in/dev2" target="_blank">
                        <img class="linkedin-icon" src="https://upload.wikimedia.org/wikipedia/commons/c/ca/LinkedIn_logo_initials.png">
                    </a>
                </div>
                <div class="profile-card">
                    <img class="profile-pic" src="https://via.placeholder.com/80">
                    <div class="profile-name">Kruthika M</div>
                    <div class="profile-student-id">Student ID: 22BBTCS151</div>
                    <a href="https://www.linkedin.com/in/dev3" target="_blank">
                        <img class="linkedin-icon" src="https://upload.wikimedia.org/wikipedia/commons/c/ca/LinkedIn_logo_initials.png">
                    </a>
                </div>
                <div class="profile-card">
                    <img class="profile-pic" src="https://via.placeholder.com/80">
                    <div class="profile-name">Maanya S</div>
                    <div class="profile-student-id">Student ID: 22BBTCS165</div>
                    <a href="https://www.linkedin.com/in/dev4" target="_blank">
                        <img class="linkedin-icon" src="https://upload.wikimedia.org/wikipedia/commons/c/ca/LinkedIn_logo_initials.png">
                    </a>
                </div>
            </div>
        </footer>
        """,
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()
