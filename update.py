import pandas as pd
import numpy as np
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.cluster import KMeans
import joblib
import os

# Load datasets (CSV version)
exercise_file = "C:/coding/fitness_tracker/exercise.csv"
calories_file = "C:/coding/fitness_tracker/calories.csv"

df_exercise = pd.read_csv(exercise_file)
df_calories = pd.read_csv(calories_file)

df = pd.merge(df_exercise, df_calories, left_on='User_ID', right_on='User_ID').drop(columns=['User_ID'])

# Ensure necessary columns exist
required_columns = ["Age", "Height", "Weight", "Duration", "Heart_Rate", "Body_Temp", "Calories"]
if not all(col in df.columns for col in required_columns):
    raise ValueError("Missing required columns in the dataset!")

# Split data into features and target
X = df[["Age", "Height", "Weight", "Duration", "Heart_Rate", "Body_Temp"]]
y = df["Calories"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Load or train the calorie prediction model
model_path = "calorie_predictor.pkl"
if os.path.exists(model_path):
    model = joblib.load(model_path)
else:
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    joblib.dump(model, model_path)

# Train K-Means model for clustering
kmeans_path = "kmeans_model.pkl"
if os.path.exists(kmeans_path):
    kmeans = joblib.load(kmeans_path)
else:
    kmeans = KMeans(n_clusters=3, random_state=42)  # 3 fitness levels
    kmeans.fit(df[["Age", "Height", "Weight"]])
    joblib.dump(kmeans, kmeans_path)

# Streamlit UI
st.title("Calorie Prediction & Fitness Analysis App 🏋️‍♂️🔥")

# User Input fields
age = st.number_input("Age", min_value=10, max_value=100, value=25)
height = st.number_input("Height (cm)", min_value=100, max_value=250, value=170)
weight = st.number_input("Weight (kg)", min_value=30, max_value=200, value=70)
duration = st.number_input("Exercise Duration (mins)", min_value=1, max_value=180, value=30)
heart_rate = st.number_input("Heart Rate (bpm)", min_value=50, max_value=200, value=100)
body_temp = st.number_input("Body Temperature (°C)", min_value=35.0, max_value=42.0, value=36.5)

if st.button("Predict Calories"):
    model = joblib.load(model_path)
    input_data = pd.DataFrame([[age, height, weight, duration, heart_rate, body_temp]],
                              columns=["Age", "Height", "Weight", "Duration", "Heart_Rate", "Body_Temp"])
    prediction = model.predict(input_data)
    st.success(f"🔥 Predicted Calories Burned: {prediction[0]:.2f} kcal")

    # Personalized Exercise Recommendation
    user_cluster = kmeans.predict([[age, height, weight]])[0]
    cluster_data = df[kmeans.labels_ == user_cluster]
    avg_duration = cluster_data["Duration"].mean()
    st.info(f"🔹 Recommended Exercise Duration: {avg_duration:.1f} minutes")

    # Compare With Similar Users
    similar_users = df[(df["Age"] >= age - 2) & (df["Age"] <= age + 2) &
                       (df["Height"] >= height - 5) & (df["Height"] <= height + 5) &
                       (df["Weight"] >= weight - 5) & (df["Weight"] <= weight + 5)]
    avg_calories = similar_users["Calories"].mean()
    percentile = (prediction[0] / avg_calories) * 100 if avg_calories else 100
    st.info(f"📊 You burn {percentile:.1f}% of the calories compared to similar users!")
