import streamlit as st
import pickle
import pandas as pd
import numpy as np
import ast
from sklearn.preprocessing import LabelEncoder

# Page config
st.set_page_config(page_title="Medicine Recommendation System", layout="centered")

# Load datasets and model
@st.cache_data
def load_data():
    data = pd.read_csv("Datasets/Training.csv")
    return data

@st.cache_resource
def load_model():
    with open("Models/model.pkl", "rb") as file:
        model = pickle.load(file)
    return model

# Load all datasets
dataset = load_data()
model = load_model()
X = dataset.drop("prognosis", axis=1)
y = dataset["prognosis"]

le = LabelEncoder()
le.fit(y)
y_encoded = le.transform(y)

# Load supporting data
sys_des = pd.read_csv("Datasets/symtoms_df.csv")
precautions = pd.read_csv("Datasets/precautions_df.csv")
workout_data = pd.read_csv("Datasets/workout_df.csv")
description = pd.read_csv("Datasets/description.csv")
medications = pd.read_csv("Datasets/medications.csv")
diets = pd.read_csv("Datasets/diets.csv")

# Helper function
def get_details(disease):
    desc = " ".join(description[description["Disease"] == disease]["Description"].values)

    precaution_data = precautions[precautions["Disease"] == disease]
    precautions_list = precaution_data.iloc[0, 2:].tolist() if not precaution_data.empty else []

    symptoms_data = sys_des[sys_des["Disease"] == disease]
    symptoms_list = symptoms_data.iloc[0, 2:].tolist() if not symptoms_data.empty else []

    meds = ast.literal_eval(medications[medications["Disease"] == disease]["Medication"].values[0])
    diet = ast.literal_eval(diets[diets["Disease"] == disease]["Diet"].values[0])
    workout = workout_data[workout_data["disease"] == disease]["workout"].tolist()

    return desc, precautions_list, meds, diet, workout, symptoms_list

# UI
st.title("🩺 MedSage")
st.write("Select your symptoms from the list below to get a disease prediction and medical recommendations.")

symptom_options = list(X.columns)
selected_symptoms = st.multiselect("Select Symptoms", symptom_options)

if st.button("Predict Disease & Recommend"):
    if not selected_symptoms:
        st.warning("⚠️ Please select at least one symptom.")
    else:
        input_data = pd.DataFrame(0, index=[0], columns=X.columns)
        input_data[selected_symptoms] = 1

        prediction = model.predict(input_data)
        predicted_disease = le.inverse_transform(prediction)[0]

        st.success(f"✅ Predicted Disease: **{predicted_disease}**")

        desc, precaution, medicine, diet, workout, related_symptoms = get_details(predicted_disease)

        # Tabs for detailed output
        tabs = st.tabs(["📝 Description", "🛡️ Precautions", "💊 Medications", "🥗 Diet", "🏃 Workout", "🔍 Symptoms"])

        with tabs[0]:
            st.markdown("### 📝 Description")
            st.write(desc)

        with tabs[1]:
            st.markdown("### 🛡️ Precautions")
            if precaution:
                for item in precaution:
                    st.markdown(f"- {item}")
            else:
                st.write("No specific precautions found.")

        with tabs[2]:
            st.markdown("### 💊 Medications")
            if medicine:
                for item in medicine:
                    st.markdown(f"- {item}")
            else:
                st.write("No specific medication found.")

        with tabs[3]:
            st.markdown("### 🥗 Diet Recommendations")
            if diet:
                for item in diet:
                    st.markdown(f"- {item}")
            else:
                st.write("No specific diet information available.")

        with tabs[4]:
            st.markdown("### 🏃 Workout Suggestions")
            if workout:
                for item in workout:
                    st.markdown(f"- {item}")
            else:
                st.write("No specific workout recommended.")

        with tabs[5]:
            st.markdown("### 🔍 Related Symptoms")
            if related_symptoms:
                for item in related_symptoms:
                    st.markdown(f"- {item}")
            else:
                st.write("No related symptoms found.")


st.markdown("---")
st.markdown("Developed by **Akash Sharma** 🚀")
