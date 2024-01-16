import streamlit as st
import pandas as pd
import pickle as pk
import plotly.express as px

# Load models
def load_models():
    with open('SandModelWithBands.pkl', 'rb') as f:
        sand_model_with_bands = pk.load(f)

    with open('SiltModelWithBands.pkl', 'rb') as f:
        silt_model_with_bands = pk.load(f)

    with open('ClayModelWithBands.pkl', 'rb') as f:
        clay_model_with_bands = pk.load(f)

    with open('TextureModelWithBands.pkl', 'rb') as f:
        texture_model_with_bands = pk.load(f)

    with open('SandModel.pkl', 'rb') as f:
        sand_model = pk.load(f)

    with open('SiltModel.pkl', 'rb') as f:
        silt_model = pk.load(f)

    with open('ClayModel.pkl', 'rb') as f:
        clay_model = pk.load(f)

    with open('TextureModel.pkl', 'rb') as f:
        texture_model = pk.load(f)

    return sand_model, silt_model, clay_model, texture_model, clay_model_with_bands, silt_model_with_bands, sand_model_with_bands, texture_model_with_bands

# Function to make predictions based on input type
def make_predictions(features, sand_model, silt_model, clay_model, texture_model, use_bands=False):
    if use_bands:
        sand_pred = sand_model.predict(features)
        silt_pred = silt_model.predict(features)
        clay_pred = clay_model.predict(features)
        texture_pred = texture_model.predict(features)
    else:
        # Assume you have a function to make predictions for soil properties based on features
        sand_pred = sand_model.predict(features)
        silt_pred = silt_model.predict(features)
        clay_pred = clay_model.predict(features)
        texture_pred = texture_model.predict(features)

    return sand_pred, silt_pred, clay_pred, texture_pred

# Function to get user input for Soil Properties
def get_soil_properties_input():
    EC, ph, OM, P, K = (
        st.slider('EC', 0.0, 2.0, 0.76), st.slider('pH', 0.0, 14.0, 7.92),
        st.slider('Organic Matter', 0.0, 2.0, 0.57), st.slider('Phosphorus (P)', 0, 10, 4),
        st.slider('Potassium (K)', 0, 100, 63)
    )
    soil_data = {'EC': EC, 'ph': ph, 'OM': OM, 'P': P, 'K': K}
    soil_features = pd.DataFrame(soil_data, index=[0])
    return soil_features

# Function to get user input for Remote Sensing Bands
def get_band_input():
    band_2, band_3, band_4, band5, band_6, band_7, band8A, band_11, band_12 = (
        st.slider('Band 2', 0.0, 1.0, 0.2237), st.slider('Band 3', 0.0, 1.0, 0.2859),
        st.slider('Band 4', 0.0, 1.0, 0.3337), st.slider('Band 5', 0.0, 1.0, 0.3806),
        st.slider('Band 6', 0.0, 1.0, 0.3806), st.slider('Band 7', 0.0, 1.0, 0.4135),
        st.slider('Band 8A', 0.0, 1.0, 0.4302), st.slider('Band 11', 0.0, 1.0, 0.4834),
        st.slider('Band 12', 0.0, 1.0, 0.4391)
    )
    band_data = {'band_2': band_2, 'band_3': band_3, 'band_4': band_4,
                 'band5': band5, 'band_6': band_6, 'band_7': band_7,
                 'band8A': band8A, 'band_11': band_11, 'band_12': band_12}
    band_features = pd.DataFrame(band_data, index=[0])
    return band_features

latitude = 37.7749
longitude = -122.4194

def plot_on_map_page(sand_pred, silt_pred, clay_pred):
    # Create a DataFrame for the plot
    data = {'Lat': [latitude, latitude, latitude],
            'Lon': [longitude, longitude + 0.001, longitude - 0.001],
            'Property': ['Clay', 'Silt', 'Sand'],
            'Prediction': [clay_pred, silt_pred, sand_pred]}

    # Create a scatter_geo plot using Plotly Express
    fig = px.scatter_geo(data, lat='Lat', lon='Lon', color='Property', size='Prediction',
                         projection='natural earth', title='Predicted Soil Properties')

    # Display the map on the Graph page
    st.write("Predicted values on map:")
    st.plotly_chart(fig)
# Streamlit App
def app():
    st.set_page_config(page_title="Soil Prediction App", page_icon="🌱", layout="wide")

    st.markdown(
        """
        <style>
            body {
                background-image: url('background.png'); /* Replace with your background image URL */
                background-size: cover;
            }
            .main-content {
                padding: 20px;
            }
        </style>
        """,
        unsafe_allow_html=True
    )

    # Sidebar
    st.sidebar.title("🌱 Soil Prediction Menu")
    pages = ["Home", "Graph", "Comparison"]
    selected_page = st.sidebar.radio("Select Page", pages)

    # Load models
    sand_model, silt_model, clay_model, texture_model, clay_model_with_bands, silt_model_with_bands, sand_model_with_bands, texture_model_with_bands = load_models()

    if selected_page == "Home":
        st.title("Welcome to the Home Page!")

        # User Input Selection (Soil Properties or Remote Sensing Bands)
        input_type = st.radio("Select Input Type", ["Soil Properties", "Remote Sensing Bands"])

        # Display user input fields based on selection
        if input_type == "Soil Properties":
            soil_features = get_soil_properties_input()
            if st.button("Predict"):
                sand_pred, silt_pred, clay_pred, texture_pred = make_predictions(soil_features, sand_model, silt_model, clay_model, texture_model )
                st.subheader('User Input Parameters')
                st.write(soil_features)

                columns = st.columns(3)

                with columns[0]:
                    st.info(f"Sand: {sand_pred[0]:.2f}")

                with columns[1]:
                    st.info(f"Silt: {silt_pred[0]:.2f}")

                with columns[2]:
                    st.info(f"Clay: {clay_pred[0]:.2f}")

            # Display texture prediction for soil properties or remote sensing bands
                st.subheader('Texture Prediction')
                st.info(f"Predicted Texture: {texture_pred[0]}")
        else:
            soil_features = get_band_input()
            if st.button("Predict"):
                sand_pred, silt_pred, clay_pred, texture_pred = make_predictions(soil_features, sand_model_with_bands, silt_model_with_bands, clay_model_with_bands, texture_model_with_bands )
                st.subheader('User Input Parameters')
                st.write(soil_features)

                columns = st.columns(3)

                with columns[0]:
                    st.info(f"Sand: {sand_pred[0]:.2f}")

                with columns[1]:
                    st.info(f"Silt: {silt_pred[0]:.2f}")

                with columns[2]:
                    st.info(f"Clay: {clay_pred[0]:.2f}")

            # Display texture prediction for soil properties or remote sensing bands
                st.subheader('Texture Prediction')
                st.info(f"Predicted Texture: {texture_pred[0]}")

    elif selected_page == "Graph":
        st.title("Welcome to the Graph Page!")
        # User Input Selection (Soil Properties or Remote Sensing Bands)
        input_type = st.radio("Select Input Type", ["Soil Properties", "Remote Sensing Bands"])

        # Display user input fields based on selection
        if input_type == "Soil Properties":
            soil_features = get_soil_properties_input()
            if st.button("Predict"):
                sand_pred, silt_pred, clay_pred, texture_pred = make_predictions(soil_features, sand_model, silt_model, clay_model, texture_model)
                st.subheader('User Input Parameters')
                st.write(soil_features)

                columns = st.columns(3)

                with columns[0]:
                    st.info(f"Sand: {sand_pred[0]:.2f}")

                with columns[1]:
                    st.info(f"Silt: {silt_pred[0]:.2f}")

                with columns[2]:
                    st.info(f"Clay: {clay_pred[0]:.2f}")

                # Display texture prediction for soil properties or remote sensing bands
                st.subheader('Texture Prediction')
                st.info(f"Predicted Texture: {texture_pred[0]}")

                # Plot predicted values on map on the Graph page
                plot_on_map_page(sand_pred[0], silt_pred[0], clay_pred[0])

        else:
            soil_features = get_band_input()
            if st.button("Predict"):
                sand_pred, silt_pred, clay_pred, texture_pred = make_predictions(soil_features, sand_model_with_bands, silt_model_with_bands, clay_model_with_bands, texture_model_with_bands)
                st.subheader('User Input Parameters')
                st.write(soil_features)

                columns = st.columns(3)

                with columns[0]:
                    st.info(f"Sand: {sand_pred[0]:.2f}")

                with columns[1]:
                    st.info(f"Silt: {silt_pred[0]:.2f}")

                with columns[2]:
                    st.info(f"Clay: {clay_pred[0]:.2f}")

                # Display texture prediction for soil properties or remote sensing bands
                st.subheader('Texture Prediction')
                st.info(f"Predicted Texture: {texture_pred[0]}")

                # Plot predicted values on map on the Graph page
                plot_on_map_page(sand_pred[0], silt_pred[0], clay_pred[0])

    elif selected_page == "Comparison":
        st.title("Welcome to the Comparison Page!")
        # Add content for the comparison page as needed

# Run the app
if __name__ == "__main__":
    app()
