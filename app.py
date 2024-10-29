import streamlit as st
import pandas as pd
from optimized_zomato_predictor import OptimizedZomatoPredictor

# Load the trained model
@st.cache_resource
def load_predictor():
    return OptimizedZomatoPredictor.load_model(r"C:\Users\khush\Downloads\ZOM\khush\zomato_model.pkl")

predictor = load_predictor()

# Load your dataset to get unique locations
data_path = r"C:\Users\khush\Downloads\ZOM\khush\zomato.csv"  # Update with the actual dataset path
data = pd.read_csv(data_path)

# Extract unique values for drop-downs
unique_locations = data['location'].unique().tolist()
unique_rest_types = data['rest_type'].unique().tolist()
unique_cuisines = data['cuisines'].unique().tolist()
unique_listed_types = data['listed_in(type)'].unique().tolist()

# App layout
st.title("Zomato Restaurant Price Prediction")
st.write("Get an estimate for a meal cost based on restaurant details.")

# Custom CSS for improved styling with larger input fields and borders on dropdown inputs
st.markdown(
    """
    <style>
    /* Main background and overlay */
    .stApp {
        background: url('https://www.washingtonpost.com/r/2010-2019/WashingtonPost/2017/03/29/Weekend/Images/Weekend0331methode.gif');
        background-size: cover;
        background-repeat: no-repeat;
        background-attachment: fixed;
        color: #FFFFFF;
    }
    
    .stApp::before {
        content: "";
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        bottom: 0;
        background: rgba(0, 0, 0, 0.6); /* Darker transparent overlay */
        z-index: 0;
    }

    /* Title and Subtitle styling */
    h1 {
        font-size: 40px; /* Increase title size */
        color: #FFCC00; /* Bright yellow */
        text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.8);
        margin-bottom: 5px; /* Reduce gap below title */
    }
    p {
        font-size: 22px; /* Adjust subtitle size */
        font-weight: 500;
        color: #FFFFFF;
        text-shadow: 1px 1px 2px rgba(0, 0, 0, 0.6);
        margin-top: -10px; /* Reduce gap above subtitle */
    }

    /* Form input styling with border and increased width */
    .stSelectbox, .stMultiselect, .stNumberInput, .stTextInput {
        background-color: rgba(255, 255, 255, 0.1);
        color: #FFFFFF;
        font-weight: bold;
        padding: 10px;
        border-radius: 10px;
        border: 2px solid #FFCC00; /* Yellow border */
        box-shadow: 2px 2px 8px rgba(0, 0, 0, 0.6);
        width: 90%; /* Increase the width of the input fields */
    }
    
    /* Create a translucent div with an orange border for the multiselect fields */
    .stMultiselect {
  background-color: rgba(255, 255, 255, 0.3); /* Semi-transparent background */
  border: 2px solid #FFCC00; /* Orange border */
  border-radius: 10px; /* Rounded corners */
  padding: 10px;
  !important; /* Increase specificity */
}

    /* Button styling */
    .stButton button {
        background-color: #FF7F50; /* Coral */
        color: white;
        font-weight: bold;
        font-size: 18px;
        padding: 10px 20px;
        border-radius: 8px;
        border: 2px solid #FFCC00;
        box-shadow: 2px 2px 10px rgba(0, 0, 0, 0.5);
        transition: 0.4s;
    }

    .stButton button:hover {
        background-color: #FF4500; /* Darker coral on hover */
        border-color: #FFFFFF;
    }

    /* Success message styling */
    .stAlert {
        background-color: rgba(255, 204, 0, 0.1);
        color: #FFCC00;
        border: 2px solid #FFCC00;
        font-weight: bold;
        text-align: center;
        padding: 10px;
        border-radius: 10px;
    }

    </style>
    """,
    unsafe_allow_html=True
)

# Create columns for inputs with extra space between them
col1, _, col2 = st.columns([3, 2, 3])

with col1:
    # Inputs on the left
    online_order = st.selectbox("Online Order (Yes/No)", ['Yes', 'No'])
    location = st.selectbox("Location", unique_locations)
    book_table = st.selectbox("Table Booking (Yes/No)", ['Yes', 'No'])
    votes = st.number_input("Votes", min_value=0, step=1)

with col2:
    # Inputs on the right
    rest_type = st.multiselect("Restaurant Type (Select Multiple)", unique_rest_types)
    cuisines = st.multiselect("Cuisines (Select Multiple)", unique_cuisines)
    listed_type = st.multiselect("Listed In (Select Multiple)", unique_listed_types)
    rate = st.number_input("Rating", min_value=0.0, max_value=5.0, step=0.1)

# Button layout at the bottom center
_, button_col, _ = st.columns([3, 2, 3])

with button_col:
    if st.button("Predict"):
        # Data preparation
        sample = pd.DataFrame([{
            "online_order": online_order,
            "book_table": book_table,
            "location": location,
            "rest_type": ', '.join(rest_type) if rest_type else "",
            "cuisines": ', '.join(cuisines) if cuisines else "",
            "listed_in(type)": ', '.join(listed_type) if listed_type else "",
            "listed_in(city)": location,
            "votes": votes,
            "rate": rate
        }])

        # Prediction
        try:
            prediction = predictor.predict(sample)
            st.success(f"Predicted Cost for Two: ₹{prediction[0]:.2f}")
        except Exception as e:
            st.error(f"Error making prediction: {str(e)}")