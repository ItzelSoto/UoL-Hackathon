
"""
============================================
OULAD Time Series Dataset Loader
Ready for Machine Learning Models
============================================
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
import os

print("=" * 80)
print("LOADING OULAD TIME SERIES DATASET")
print("=" * 80)

# ============================================
# STEP 1: LOAD THE MASTER DATASET
# ============================================
print("\n[1/5] Loading master dataset...")

# Update this path to where you saved the file
DATA_PATH = 'MASTER_timeseries_combined.csv'

df = pd.read_csv(DATA_PATH)
print(f"✓ Loaded {df.shape[0]:,} rows and {df.shape[1]} columns")

# ============================================
# STEP 2: DEFINE FEATURE COLUMNS
# ============================================
print("\n[2/5] Defining features...")

feature_cols = [
    'week',
    'studied_credits',
    'weekly_clicks',
    'cumulative_clicks',
    'unique_activities',
    'weekly_interactions',
    'cumulative_avg_score',
    'cumulative_assessments',
    'cumulative_banked',
    'date_registration',
    'gender_encoded',
    'region_encoded',
    'highest_education_encoded',
    'imd_band_encoded',
    'age_band_encoded',
    'disability_encoded',
    'prev_week_clicks',
    'prev_week_interactions',
    'rolling_avg_clicks_3w',
    'rolling_avg_interactions_3w',
    'code_module_encoded',
    'code_presentation_encoded'
]

identifier_cols = ['id_student', 'code_module', 'code_presentation', 'week']
target_col = 'target'

print(f"✓ {len(feature_cols)} features defined")

# ============================================
# STEP 3: SPLIT INTO X, y, AND IDENTIFIERS
# ============================================
print("\n[3/5] Splitting dataset...")

X = df[feature_cols].copy()
y = df[target_col].copy()
student_ids = df[identifier_cols].copy()

print(f"✓ X shape: {X.shape}")
print(f"✓ y shape: {y.shape}")
print(f"✓ student_ids shape: {student_ids.shape}")

# Check for missing values
print(f"\n✓ Missing values in X: {X.isnull().sum().sum()}")
print(f"✓ Missing values in y: {y.isnull().sum()}")

# ============================================
# STEP 4: ENCODE TARGET VARIABLE
# ============================================
print("\n[4/5] Encoding target variable...")

# Create label encoder for target
le_target = LabelEncoder()
y_encoded = le_target.fit_transform(y)

# Show the mapping
print(f"\n✓ Target classes: {le_target.classes_}")
print(f"✓ Encoding mapping:")
for idx, label in enumerate(le_target.classes_):
    print(f"   {label} → {idx}")

# Distribution
print(f"\n✓ Target distribution:")
print(pd.Series(y).value_counts())

# ============================================
# STEP 5: TRAIN-TEST SPLIT (TIME-AWARE)
# ============================================
print("\n[5/5] Creating train-test split...")

# Option A: Random split (simple)
X_train, X_test, y_train, y_test, ids_train, ids_test = train_test_split(
    X, y_encoded, student_ids, 
    test_size=0.2, 
    random_state=42,
    stratify=y_encoded
)

print(f"✓ Training set: {X_train.shape[0]:,} samples")
print(f"✓ Test set: {X_test.shape[0]:,} samples")

# ============================================
# OPTIONAL: FEATURE SCALING
# ============================================
print("\n[OPTIONAL] Scaling features...")

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Convert back to dataframes for convenience
X_train_scaled = pd.DataFrame(X_train_scaled, columns=feature_cols, index=X_train.index)
X_test_scaled = pd.DataFrame(X_test_scaled, columns=feature_cols, index=X_test.index)

print(f"✓ Features scaled (mean=0, std=1)")

# ============================================
# SUMMARY
# ============================================
print("\n" + "=" * 80)
print("DATASET READY FOR MACHINE LEARNING!")
print("=" * 80)
print(f"\n📊 Available Variables:")
print(f"   X               - Features (unscaled) - shape {X.shape}")
print(f"   y               - Target (original labels)")
print(f"   y_encoded       - Target (numeric 0-3) - shape {y_encoded.shape}")
print(f"   student_ids     - Identifiers - shape {student_ids.shape}")
print(f"\n🔄 Train-Test Split:")
print(f"   X_train         - shape {X_train.shape}")
print(f"   X_test          - shape {X_test.shape}")
print(f"   y_train         - shape {y_train.shape}")
print(f"   y_test          - shape {y_test.shape}")
print(f"\n📐 Scaled Versions:")
print(f"   X_train_scaled  - shape {X_train_scaled.shape}")
print(f"   X_test_scaled   - shape {X_test_scaled.shape}")
print(f"\n🎯 You can now train models:")
print(f"   - Use X_train, y_train for training")
print(f"   - Use X_test, y_test for evaluation")
print(f"   - Use X_train_scaled if using neural networks/SVM")
print("=" * 80)

# ============================================
# QUICK DATA PREVIEW
# ============================================
print("\n📋 Sample Data Preview:")
print(df[['id_student', 'week', 'weekly_clicks', 'cumulative_avg_score', 'target']].head(10))

# ============================================
# EXAMPLE: Quick Model Training (Optional)
# ============================================
print("\n" + "=" * 80)
print("EXAMPLE: TRAINING A QUICK RANDOM FOREST")
print("=" * 80)

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score

# Train a simple model
print("\nTraining Random Forest...")
rf_model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
rf_model.fit(X_train, y_train)

# Predictions
y_pred = rf_model.predict(X_test)

# Evaluate
accuracy = accuracy_score(y_test, y_pred)
print(f"\n✓ Model trained!")
print(f"✓ Test Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")

print("\n📊 Classification Report:")
print(classification_report(y_test, y_pred, target_names=le_target.classes_))

# Feature importance
print("\n🔝 Top 10 Most Important Features:")
feature_importance = pd.DataFrame({
    'feature': feature_cols,
    'importance': rf_model.feature_importances_
}).sort_values('importance', ascending=False)
print(feature_importance.head(10).to_string(index=False))

print("\n" + "=" * 80)
print("✅ SETUP COMPLETE - START BUILDING YOUR MODELS!")
print("=" * 80)
