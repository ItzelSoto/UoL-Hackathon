
"""
Quick Load Script - Minimal Version
"""
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# Load data
df = pd.read_csv('MASTER_timeseries_combined.csv')

# Define features
feature_cols = [
    'week', 'studied_credits', 'weekly_clicks', 'cumulative_clicks',
    'unique_activities', 'weekly_interactions', 'cumulative_avg_score',
    'cumulative_assessments', 'cumulative_banked', 'date_registration',
    'gender_encoded', 'region_encoded', 'highest_education_encoded',
    'imd_band_encoded', 'age_band_encoded', 'disability_encoded',
    'prev_week_clicks', 'prev_week_interactions',
    'rolling_avg_clicks_3w', 'rolling_avg_interactions_3w',
    'code_module_encoded', 'code_presentation_encoded'
]

# Split dataset
X = df[feature_cols]
y = df['target']
student_ids = df[['id_student', 'code_module', 'code_presentation', 'week']]

# Encode target
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
)

print(f"✅ Data loaded: X_train {X_train.shape}, X_test {X_test.shape}")
print(f"Classes: {le.classes_}")

# Now build your model!
