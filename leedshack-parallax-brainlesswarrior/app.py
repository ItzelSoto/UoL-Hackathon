import streamlit as st 
import pandas as pd 
import joblib
import numpy as np
from tensorflow.keras.models import load_model
@st.cache_resource
def load_models():
    model = load_model('best_lstm_model.h5')
    scaler = joblib.load('feature_scaler.pkl')
    label_encoder = joblib.load('label_encoder.pkl')
    
    # Check expected features
    st.info(f"✓ Model loaded | Scaler expects: {scaler.n_features_in_} features")
    
    return model, scaler, label_encoder
try:
    model, scaler, label_encoder = load_models()
except Exception as e:
    st.error(f"❌ Error loading models: {e}")
    st.stop()
st.title("🎓 LeedsHack: Student Result Predictor (LSTM)")
st.write("Predicting: Pass, Fail, Distinction, or Withdrawn")

st.markdown("---")

# Input form - Collect all necessary features
col1, col2, col3 = st.columns(3)

with col1:
    week = st.slider("Current Week", 1, 40, 20)
    studied_credits = st.number_input("Studied Credits", value=60)
    weekly_clicks = st.number_input("Weekly Clicks", min_value=0, value=50)
    cumulative_clicks = st.number_input("Cumulative Clicks", min_value=0, value=1000)
    unique_activities = st.number_input("Unique Activities", min_value=0, value=15)
    weekly_interactions = st.number_input("Weekly Interactions", min_value=0, value=10)
    cumulative_avg_score = st.slider("Cumulative Avg Score", 0, 100, 50)

with col2:
    cumulative_assessments = st.number_input("Cumulative Assessments", min_value=0, value=3)
    cumulative_banked = st.number_input("Cumulative Banked", min_value=0, value=2)
    date_registration = st.number_input("Date Registration", value=-30)
    gender_encoded = st.selectbox("Gender", [0, 1], format_func=lambda x: "Male" if x == 0 else "Female")
    region_encoded = st.selectbox("Region", list(range(13)))
    highest_education_encoded = st.selectbox("Highest Education", list(range(7)))
    imd_band_encoded = st.selectbox("IMD Band", list(range(11)))

with col3:
    age_band_encoded = st.selectbox("Age Band", list(range(4)))
    disability_encoded = st.selectbox("Disability", [0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
    prev_week_clicks = st.number_input("Previous Week Clicks", min_value=0, value=40)
    prev_week_interactions = st.number_input("Previous Week Interactions", min_value=0, value=8)
    rolling_avg_clicks_3w = st.number_input("Rolling Avg Clicks (3w)", min_value=0.0, value=45.0)
    rolling_avg_interactions_3w = st.number_input("Rolling Avg Interactions (3w)", min_value=0.0, value=9.0)
    code_module_encoded = st.selectbox("Code Module", list(range(7)))
    code_presentation_encoded = st.selectbox("Code Presentation", list(range(4)))
if st.button("🔮 Predict Student Outcome"):
    
    # STEP 1: Create feature array for ONE student across 40 weeks
    # Shape: (40 weeks, 22 features) - CORRECTED TO 22!
    sequence = np.zeros((40, 22))
    # STEP 2: Fill in the 22 features for each week
    for i in range(40):
        sequence[i, 0] = i + 1  # week
        sequence[i, 1] = studied_credits  # studied_credits
        sequence[i, 2] = weekly_clicks  # weekly_clicks
        sequence[i, 3] = cumulative_clicks  # cumulative_clicks
        sequence[i, 4] = unique_activities  # unique_activities
        sequence[i, 5] = weekly_interactions  # weekly_interactions
        sequence[i, 6] = cumulative_avg_score  # cumulative_avg_score
        sequence[i, 7] = cumulative_assessments  # cumulative_assessments
        sequence[i, 8] = cumulative_banked  # cumulative_banked
        sequence[i, 9] = date_registration  # date_registration
        sequence[i, 10] = gender_encoded  # gender_encoded
        sequence[i, 11] = region_encoded  # region_encoded
        sequence[i, 12] = highest_education_encoded  # highest_education_encoded
        sequence[i, 13] = imd_band_encoded  # imd_band_encoded
        sequence[i, 14] = age_band_encoded  # age_band_encoded
        sequence[i, 15] = disability_encoded  # disability_encoded
        sequence[i, 16] = prev_week_clicks  # prev_week_clicks
        sequence[i, 17] = prev_week_interactions  # prev_week_interactions
        sequence[i, 18] = rolling_avg_clicks_3w  # rolling_avg_clicks_3w
        sequence[i, 19] = rolling_avg_interactions_3w  # rolling_avg_interactions_3w
        sequence[i, 20] = code_module_encoded  # code_module_encoded
        sequence[i, 21] = code_presentation_encoded  # code_presentation_encoded
    # STEP 3: Reshape to 3D for LSTM
    sequence_3d = sequence.reshape(1, 40, 22)  # Shape: (1, 40, 22)
    # STEP 4: Scale the features
    sequence_2d = sequence_3d.reshape(-1, 22)  # Shape: (40, 22)
    sequence_scaled = scaler.transform(sequence_2d)
    # Reshape back to 3D
    sequence_final = sequence_scaled.reshape(1, 40, 22)  # Shape: (1, 40, 22)
    # STEP 5: Make prediction
    prediction_probs = model.predict(sequence_final, verbose=0)
    prediction_encoded = np.argmax(prediction_probs[0])
    prediction = label_encoder.inverse_transform([prediction_encoded])[0]
    confidence = prediction_probs[0][prediction_encoded] * 100
    st.markdown("---")
    st.subheader("📊 Prediction Results")
    if prediction == 'Distinction':
        st.success(f"### 🌟 Predicted Outcome: {prediction}")
        st.balloons()
    elif prediction == 'Pass':
        st.info(f"### ✅ Predicted Outcome: {prediction}")
    elif prediction == 'Fail':
        st.error(f"### ⚠️ Predicted Outcome: {prediction}")
    else:  # Withdrawn
        st.warning(f"### 🚪 Predicted Outcome: {prediction}")
    
    st.metric("Confidence", f"{confidence:.1f}%")
    
    # Display all class probabilities
    st.markdown("#### Class Probabilities:")
    prob_df = pd.DataFrame({
        'Outcome': label_encoder.classes_,
        'Probability': [f"{p*100:.1f}%" for p in prediction_probs[0]]
    })
    st.dataframe(prob_df, use_container_width=True)
    
    # Visualization
    st.bar_chart(
        data=pd.DataFrame({
            'Probability': prediction_probs[0]
        }, index=label_encoder.classes_)
    )

# Sidebar
st.sidebar.title("ℹ️ Model Information")
st.sidebar.markdown(f"""
**Model Type:** LSTM Neural Network

**Input Shape:** (40 weeks, 22 features)

**Features Used:**
1. Week, Credits, Clicks
2. Activities, Interactions, Scores
3. Demographics (Gender, Region, Age)
4. Education, IMD Band, Disability
5. Historical patterns (rolling averages)

**Output Classes:** {', '.join(label_encoder.classes_)}
""")
