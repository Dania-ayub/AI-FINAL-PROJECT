import numpy as np
import pandas as pd
import pickle as pk
import openpyxl
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import mean_squared_error, accuracy_score, classification_report

def createModel(data):
    # Extracting features and target variables
    features = data[['EC', 'ph', 'OM', 'P', 'K']]
    target_sand = data['Sand']
    target_silt = data['Silt']
    target_clay = data['Clay']
    target_texture = data['Texture']

    # Split the data for predicting sand, silt, clay
    xTrain, xTest, yTrain_sand, yTest_sand = train_test_split(features, target_sand, test_size=0.2, random_state=42)
    yTrain_silt, yTest_silt = train_test_split(target_silt, test_size=0.2, random_state=42)
    yTrain_clay, yTest_clay = train_test_split(target_clay, test_size=0.2, random_state=42)

    # Train the models for sand, silt, clay
    sand_model = RandomForestRegressor(n_estimators=100, random_state=42)
    silt_model = RandomForestRegressor(n_estimators=100, random_state=42)
    clay_model = RandomForestRegressor(n_estimators=100, random_state=42)

    sand_model.fit(xTrain, yTrain_sand)
    silt_model.fit(xTrain, yTrain_silt)
    clay_model.fit(xTrain, yTrain_clay)

    # Test the models for sand, silt, clay
    yPred_sand = sand_model.predict(xTest)
    yPred_silt = silt_model.predict(xTest)
    yPred_clay = clay_model.predict(xTest)

    print("Mean Squared Error for Sand:", mean_squared_error(yTest_sand, yPred_sand))
    print("Mean Squared Error for Silt:", mean_squared_error(yTest_silt, yPred_silt))
    print("Mean Squared Error for Clay:", mean_squared_error(yTest_clay, yPred_clay))

    # Split the data for predicting texture
    xTrain_texture, xTest_texture, yTrain_texture, yTest_texture = train_test_split(features, target_texture, test_size=0.2, random_state=42)

    # Train the model for texture
    texture_model = RandomForestClassifier(n_estimators=100, random_state=42)
    texture_model.fit(xTrain_texture, yTrain_texture)

    # Test the model for texture
    yPred_texture = texture_model.predict(xTest_texture)

    print("Accuracy for Texture:", accuracy_score(yTest_texture, yPred_texture) * 100)
    print("Classification Report for Texture: \n", classification_report(yTest_texture, yPred_texture))

    return sand_model, silt_model, clay_model, texture_model

def createModelWithBands(data):
    # Extracting features and target variables for bands
    features_bands = data[['band_2', 'band_3', 'band_4', 'band5', 'band_6', 'band_7', 'band8A', 'band_11', 'band_12']]
    target_sand = data['Sand']
    target_silt = data['Silt']
    target_clay = data['Clay']
    target_texture = data['Texture']

    # Split the data for predicting sand, silt, clay
    xTrain, xTest, yTrain_sand, yTest_sand = train_test_split(features_bands, target_sand, test_size=0.2, random_state=42)
    yTrain_silt, yTest_silt = train_test_split(target_silt, test_size=0.2, random_state=42)
    yTrain_clay, yTest_clay = train_test_split(target_clay, test_size=0.2, random_state=42)

    # Train the models for sand, silt, clay
    sand_model = RandomForestRegressor(n_estimators=100, random_state=42)
    silt_model = RandomForestRegressor(n_estimators=100, random_state=42)
    clay_model = RandomForestRegressor(n_estimators=100, random_state=42)

    sand_model.fit(xTrain, yTrain_sand)
    silt_model.fit(xTrain, yTrain_silt)
    clay_model.fit(xTrain, yTrain_clay)

    # Test the models for sand, silt, clay
    yPred_sand = sand_model.predict(xTest)
    yPred_silt = silt_model.predict(xTest)
    yPred_clay = clay_model.predict(xTest)

    print("Mean Squared Error for Sand:", mean_squared_error(yTest_sand, yPred_sand))
    print("Mean Squared Error for Silt:", mean_squared_error(yTest_silt, yPred_silt))
    print("Mean Squared Error for Clay:", mean_squared_error(yTest_clay, yPred_clay))

    # Split the data for predicting texture
    xTrain_texture, xTest_texture, yTrain_texture, yTest_texture = train_test_split(features_bands, target_texture, test_size=0.2, random_state=42)

    # Train the model for texture
    texture_model = RandomForestClassifier(n_estimators=100, random_state=42)
    texture_model.fit(xTrain_texture, yTrain_texture)

    # Test the model for texture
    yPred_texture = texture_model.predict(xTest_texture)

    print("Accuracy for Texture:", accuracy_score(yTest_texture, yPred_texture) * 100)
    print("Classification Report for Texture: \n", classification_report(yTest_texture, yPred_texture))

    return sand_model, silt_model, clay_model, texture_model

def main():
    data = pd.read_excel("extractedpoints-20m.xlsx")  # Replace with your dataset file name
    sand_model, silt_model, clay_model, texture_model = createModel(data)
    bands_model_sand, bands_model_silt, bands_model_clay, bands_model_texture = createModelWithBands(data)

    # Save models to pickle files
    with open('SandModel.pkl', 'wb') as f:
        pk.dump(sand_model, f)

    with open('SiltModel.pkl', 'wb') as f:
        pk.dump(silt_model, f)

    with open('ClayModel.pkl', 'wb') as f:
        pk.dump(clay_model, f)

    with open('TextureModel.pkl', 'wb') as f:
        pk.dump(texture_model, f)

    # Save models with bands to pickle files
    with open('SandModelWithBands.pkl', 'wb') as f:
        pk.dump(bands_model_sand, f)

    with open('SiltModelWithBands.pkl', 'wb') as f:
        pk.dump(bands_model_silt, f)

    with open('ClayModelWithBands.pkl', 'wb') as f:
        pk.dump(bands_model_clay, f)

    with open('TextureModelWithBands.pkl', 'wb') as f:
        pk.dump(bands_model_texture, f)

if __name__ == '__main__':
    main()
