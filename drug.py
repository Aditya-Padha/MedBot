import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from tensorflow.keras.models import load_model


best_model = load_model('drugmodel1.h5')
data = pd.read_csv('Drug.csv')


label_encoders = {}
for column in ['Disease', 'Gender']:
    label_encoders[column] = LabelEncoder()
    data[column] = label_encoders[column].fit_transform(data[column])

# Scale numerical variables
scaler = StandardScaler()
data['Age'] = scaler.fit_transform(data[['Age']])

# Encode target variable 'Drug'
label_encoder_drug = LabelEncoder()
data['Drug'] = label_encoder_drug.fit_transform(data['Drug'])


def recommend_drug(disease, gender, age):
    gender = gender.capitalize()
    new_data = pd.DataFrame({'Disease': [disease], 'Gender': [gender], 'Age': [age]})
    new_data['Disease'] = label_encoders['Disease'].transform(new_data['Disease'])
    new_data['Gender'] = label_encoders['Gender'].transform(new_data['Gender'])
    new_data['Age'] = scaler.transform(new_data[['Age']])
    predicted_drug_prob = best_model.predict(new_data)
    predicted_drug_index = np.argmax(predicted_drug_prob, axis=1)
    predicted_drug = label_encoder_drug.inverse_transform(predicted_drug_index)
    drug_dict = {
        "drug": predicted_drug[0]
    }

    return drug_dict


