import streamlit as st
import torch
from torchvision import transforms
from PIL import Image
import torch.nn.functional as F

st.set_page_config(
    page_title="My Streamlit App",
    page_icon=":smiley:",
    layout="wide",
)

# Load the PyTorch model
model = torch.load('best_rgmodel.pth', map_location=torch.device('cpu'))
model.eval()

# Define image transformation
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Function to make predictions
def predict(image):
    img = Image.open(image).convert('RGB')
    img = transform(img)
    img = img.unsqueeze(0)  # Add batch dimension
    output = model(img)
    return output

# Title
st.title('Corbin App - Rubber Guardian')

# Sidebar menu
menu_option = st.sidebar.selectbox("Menu", ["Description", "Detection"])

if menu_option == "Detection":
    st.sidebar.title("Settings")
    threshold = st.sidebar.slider("Define threshold (adjustable based on model)", 0.0, 1.0, 0.5)

    # File uploader
    uploaded_file = st.file_uploader("Choose an image...", type="jpg")

    if uploaded_file is not None:
        # Display the selected image
        st.sidebar.image(uploaded_file, caption='Uploaded Image.', use_column_width=True)

        st.write("")

        # Placeholder for animation
        animation_placeholder = st.empty()

        # Make prediction
        prediction = predict(uploaded_file)

        # Apply softmax to convert logits to probabilities
        probabilities = F.softmax(prediction, dim=1)

        # Extract the probability for each class
        prob_good = probabilities[0, 1].item()
        prob_defect = probabilities[0, 0].item()

        # Interpret the prediction
        result = "Good" if prob_good >= threshold else "Defect"

        if result == "Good":
            st.markdown(f"<span style='font-size: 30px; font-weight: 600'>Prediction Results: </span> <span style='color: green;font-size: 30px; font-weight: 600'>{result}</span> <span> <br> Feasible to use</span>", unsafe_allow_html=True)
        else:
            st.markdown(f"<span style='font-size: 30px; font-weight: 600'>Prediction Results: </span> <span style='color: red;font-size: 30px; font-weight: 600'>{result}</span> <span> <br> Not worth using</span>", unsafe_allow_html=True)

elif menu_option == "Description":
    #st.title("Description")
    st.markdown("<span style='font-size: 20px; font-weight: 500'><b>Rubber Guardian</b> is a cutting-edge application designed to provide real-time assessments of tire conditions, empowering users to determine whether their vehicle tires are in optimal or compromised states. This intuitive and user-friendly tool utilizes advanced technology to analyze tire conditions accurately, promoting safety and extending the lifespan of tires.</span>", unsafe_allow_html=True)
    st.markdown("<span style='font-size: 20px; font-weight: 500'>With <b>Rubber Guardian</b>, users can quickly and easily capture images of their vehicle tires using their smartphones or cameras. The application then analyzes these images instantly, providing a clear and concise evaluation of the tire's overall health. Users receive clear visual feedback on the condition of their tires through a simple and intuitive interface. The application categorizes the tire as either 'Good' or 'Defect,' offering users a quick and understandable assessment of their tire health.</span>", unsafe_allow_html=True)
