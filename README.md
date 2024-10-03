
# AI Image Captioning Web App

This is a **Streamlit-based web application** that uses a **BLIP (Bootstrapped Language-Image Pre-training)** model for generating captions based on user-uploaded images. The app allows users to upload images in JPG, JPEG, or PNG formats, and it provides an AI-generated caption for the uploaded image.

## Features

-   **Image Upload**: Users can upload images directly into the app.
-   **Image Display**: The uploaded image is displayed on the screen.
-   **AI Captioning**: The app uses the **BLIP model** to generate and display a caption for the uploaded image.
-   **Developer Credits**: Displays developer profiles with customizable information in the footer.

## Requirements

The following Python libraries are required to run the application:





``streamlit==1.26.0
Pillow==10.0.0
transformers==4.33.0
torch==2.0.1``

Install all dependencies using the following command:





```sh
pip install streamlit Pillow transformers torch
```

## How to Run the Application

1.  Clone the repository or download the code.
2.  Install the required packages by running the command:
    
    
    
  
    
    ```sh
    pip install -r requirements.txt
    ``` 
    
4.  Run the Streamlit app by executing the following command in the terminal:
    
    
    

    
    ```sh
    streamlit run app.py
    ``` 
    
6.  Open your browser and go to the address:  
  
   http://localhost:8501

  or the URL provided by Streamlit after running the app.

## Code Structure

-   **`app.py`**: Contains the main logic of the application.
-   **`requirements.txt`**: Lists all required Python libraries.
-   **`README.md`**: Documentation for the project (this file).

### Model

-   The app uses the **BLIP Image Captioning Model** from Hugging Face's model hub (`Salesforce/blip-image-captioning-base`) to generate captions based on uploaded images.

## Developer Credits

**Instagram :** [KARTHIK PRABHU](https://www.instagram.com/karthik10.__)
**LinkedIn** : [KARTHIK PRABHU](https://www.linkedin.com/in/karthik-prabhu-4165b7290/)
[karthikprabhu.netlify.app](https://karthikprabhu.netlify.app)

### Customization

-   You can easily customize the developer details in the `footer` section by updating the profile card HTML and the LinkedIn URLs in the `app.py` file.

## Future Improvements

-   **Add more styling** to enhance user experience.
-   **Include support for multiple languages** in caption generation.
-   **Optimize image processing speed** for larger images.
