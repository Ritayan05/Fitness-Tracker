import pandas as pd
import numpy as np
import streamlit as st
import joblib
import os
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

# Load datasets
exercise_file = "C:/coding/fitness_tracker/exercise.csv"
calories_file = "C:/coding/fitness_tracker/calories.csv"

df_exercise = pd.read_csv(exercise_file)
df_calories = pd.read_csv(calories_file)

df = pd.merge(df_exercise, df_calories, left_on='User_ID', right_on='User_ID').drop(columns=['User_ID'])

# Check for necessary columns
required_columns = ["Age", "Height", "Weight", "Duration", "Heart_Rate", "Body_Temp", "Calories"]
if not all(col in df.columns for col in required_columns):
    raise ValueError("Missing required columns in the dataset!")

# Load or train calorie prediction model
model_path = "calorie_predictor.pkl"
if os.path.exists(model_path):
    model = joblib.load(model_path)
else:
    X = df[["Age", "Height", "Weight", "Duration", "Heart_Rate", "Body_Temp"]]
    y = df["Calories"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    joblib.dump(model, model_path)

# Load or train KMeans model
kmeans_path = "kmeans_model.pkl"
if os.path.exists(kmeans_path):
    kmeans = joblib.load(kmeans_path)
else:
    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    df["Cluster"] = kmeans.fit_predict(df[["Age", "Height", "Weight", "Calories"]])
    joblib.dump(kmeans, kmeans_path)

# Sidebar Navigation
st.sidebar.title("🏋️‍♂️ Fitness Tracker")
page = st.sidebar.radio("📌 Navigation", ["Calorie Prediction", "Insights Dashboard"])

# Calorie Prediction Page
if page == "Calorie Prediction":
    st.title("🔥 Calorie Prediction")
    st.subheader("Enter your details to estimate calories burned")
    
    col1, col2 = st.columns(2)
    with col1:
        age = st.number_input("📅 Age", min_value=10, max_value=100, value=25)
        height = st.number_input("📏 Height (cm)", min_value=100, max_value=250, value=170)
        weight = st.number_input("⚖️ Weight (kg)", min_value=30, max_value=200, value=70)
    with col2:
        duration = st.number_input("⏳ Exercise Duration (mins)", min_value=1, max_value=180, value=30)
        heart_rate = st.number_input("💓 Heart Rate (bpm)", min_value=50, max_value=200, value=100)
        body_temp = st.number_input("🌡️ Body Temperature (°C)", min_value=35.0, max_value=42.0, value=36.5)

    if st.button("🔮 Predict Calories", use_container_width=True):
        input_data = pd.DataFrame([[age, height, weight, duration, heart_rate, body_temp]],
                                  columns=["Age", "Height", "Weight", "Duration", "Heart_Rate", "Body_Temp"])
        prediction = model.predict(input_data)[0]

        st.success(f"🔥 Estimated Calories Burned: **{prediction:.2f} kcal**")

# Insights Dashboard Page
elif page == "Insights Dashboard":
    st.title("📊 Insights & Recommendations")

    # Ensure the prediction is available
    if "prediction" not in locals():
        st.warning("⚠️ Please go to 'Calorie Prediction' and estimate your burned calories first!")
    else:
        # Predict user's cluster
        user_cluster = kmeans.predict([[age, height, weight, prediction]])[0]
        st.info(f"📍 You belong to **Cluster {user_cluster}** based on your attributes.")

        # Get data of users in the same cluster
        cluster_data = df[df["Cluster"] == user_cluster]
        avg_calories = cluster_data["Calories"].mean()
        
        st.metric(label="🔍 Average Calories Burned (Similar Users)", value=f"{avg_calories:.2f} kcal")

        # Visualization - Calorie Distribution
        fig, ax = plt.subplots(figsize=(8, 5))
        sns.histplot(cluster_data["Calories"], bins=20, kde=True, color="blue", ax=ax)
        plt.axvline(prediction, color='red', linestyle='--', label='Your Calories')
        plt.legend()
        st.pyplot(fig)

        # Personalized Recommendations
        st.subheader("💡 Personalized Exercise Tip")
        if prediction < avg_calories:
            st.warning("⚡ You may need to **increase exercise duration or intensity** to match your peers!")
        else:
            st.success("🏆 Great job! You're **burning more calories than average** for your group!")

