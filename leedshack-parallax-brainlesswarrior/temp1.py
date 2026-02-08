
import streamlit as st
import joblib as jb
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Bidirectional
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
df = pd.read_csv('C:\\APP\\leeds University\\extra_curricular\\leedshack\\output\\MASTER_timeseries_combined.csv')
print(f"✓ Loaded {df.shape[0]:,} rows")
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

# Encode target
le_target = LabelEncoder()
df['target_encoded'] = le_target.fit_transform(df['target'])

print(f"✓ Target classes: {le_target.classes_}")
print(f"   {le_target.classes_[0]} → 0")
print(f"   {le_target.classes_[1]} → 1")
print(f"   {le_target.classes_[2]} → 2")
print(f"   {le_target.classes_[3]} → 3")
# Group by student to create sequences
def create_sequences(data, feature_cols, target_col, max_weeks=40): 
    sequences = []
    targets = []
    student_info = []

    # Group by student (unique combination of module, presentation, id)
    grouped = data.groupby(['code_module', 'code_presentation', 'id_student'])

    for (module, presentation, student_id), group in grouped:
        # Sort by week
        group = group.sort_values('week')

        # Get feature values for this student across all weeks
        sequence = group[feature_cols].values

        # Pad if less than max_weeks (some students have missing weeks)
        if len(sequence) < max_weeks:
            # Pad with zeros
            padding = np.zeros((max_weeks - len(sequence), len(feature_cols)))
            sequence = np.vstack([sequence, padding])
        elif len(sequence) > max_weeks:
            # Truncate if more than max_weeks
            sequence = sequence[:max_weeks]

        # Get target (same for all weeks for this student)
        target = group[target_col].iloc[0]

        sequences.append(sequence)
        targets.append(target)
        student_info.append((module, presentation, student_id))

    return np.array(sequences), np.array(targets), student_info

# Create sequences
X_sequences, y_sequences, student_info = create_sequences(
    df, feature_cols, 'target_encoded', max_weeks=40
)

print(f"✓ Created sequences!")
print(f"   X shape: {X_sequences.shape}")
print(f"   Format: [samples={X_sequences.shape[0]}, timesteps={X_sequences.shape[1]}, features={X_sequences.shape[2]}]")
print(f"   y shape: {y_sequences.shape}")
print(f"\n   Example: Each of {X_sequences.shape[0]} students has {X_sequences.shape[1]} weeks of data")
print(f"            with {X_sequences.shape[2]} features per week")
X_train, X_test, y_train, y_test = train_test_split(
    X_sequences, y_sequences, 
    test_size=0.2, 
    random_state=42,
    stratify=y_sequences
)

print(f"✓ Train: {X_train.shape[0]:,} students")
print(f"✓ Test:  {X_test.shape[0]:,} students")
X_train_2d = X_train.reshape(-1, X_train.shape[-1])
X_test_2d = X_test.reshape(-1, X_test.shape[-1])

# Fit scaler on training data
scaler = StandardScaler()
X_train_2d_scaled = scaler.fit_transform(X_train_2d)
X_test_2d_scaled = scaler.transform(X_test_2d)

# Reshape back to 3D
X_train_scaled = X_train_2d_scaled.reshape(X_train.shape)
X_test_scaled = X_test_2d_scaled.reshape(X_test.shape)
model = Sequential([
    # First LSTM layer - returns sequences for next LSTM layer
    LSTM(128, return_sequences=True, input_shape=(X_train_scaled.shape[1], X_train_scaled.shape[2])),
    Dropout(0.3),  # Prevent overfitting

    # Second LSTM layer - returns sequences
    LSTM(64, return_sequences=True),
    Dropout(0.3),

    # Third LSTM layer - returns final output only
    LSTM(32, return_sequences=False),
    Dropout(0.3),

    # Dense layers for classification
    Dense(64, activation='relu'),
    Dropout(0.2),

    Dense(32, activation='relu'),

    # Output layer - 4 classes (Pass, Distinction, Fail, Withdrawn)
    Dense(4, activation='softmax')
])

# Compile model
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

print("\n✓ LSTM Architecture:")
model.summary()
# Callbacks for better training
early_stopping = EarlyStopping(
    monitor='val_loss',
    patience=5,
    restore_best_weights=True,
    verbose=1
)

checkpoint = ModelCheckpoint(
    'best_lstm_model.h5',
    monitor='val_accuracy',
    save_best_only=True,
    verbose=1
)

# Train model
history = model.fit(
    X_train_scaled, y_train,
    validation_split=0.2,  # Use 20% of training data for validation
    epochs=27,
    batch_size=64,
    callbacks=[early_stopping, checkpoint],
    verbose=1
)
# Predictions
y_pred_probs = model.predict(X_test_scaled)
y_pred = np.argmax(y_pred_probs, axis=1)

# Accuracy
test_loss, test_accuracy = model.evaluate(X_test_scaled, y_test, verbose=0)
print(f"\n✓ Test Accuracy: {test_accuracy:.4f} ({test_accuracy*100:.2f}%)")
print(f"✓ Test Loss: {test_loss:.4f}")

# Classification report
print("\n📊 Classification Report:")
print(classification_report(y_test, y_pred, target_names=le_target.classes_))

# Confusion matrix
print("\n📊 Confusion Matrix:")
cm = confusion_matrix(y_test, y_pred)
print(cm)
# Plot training history
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# Accuracy plot
ax1.plot(history.history['accuracy'], label='Train Accuracy')
ax1.plot(history.history['val_accuracy'], label='Validation Accuracy')
ax1.set_title('Model Accuracy Over Epochs')
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Accuracy')
ax1.legend()
ax1.grid(True)

# Loss plot
ax2.plot(history.history['loss'], label='Train Loss')
ax2.plot(history.history['val_loss'], label='Validation Loss')
ax2.set_title('Model Loss Over Epochs')
ax2.set_xlabel('Epoch')
ax2.set_ylabel('Loss')
ax2.legend()
ax2.grid(True)

plt.tight_layout()
plt.savefig('lstm_training_history.png', dpi=300, bbox_inches='tight')
print("\n✓ Saved: lstm_training_history.png")

# Confusion matrix heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=le_target.classes_, 
            yticklabels=le_target.classes_)
plt.title('Confusion Matrix - LSTM Model')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.tight_layout()
plt.savefig('lstm_confusion_matrix.png', dpi=300, bbox_inches='tight')
model.save('student_outcome_lstm_final.h5')
# Save scaler
import joblib
joblib.dump(scaler, 'feature_scaler.pkl')
joblib.dump(le_target, 'label_encoder.pkl')
print("\n🎯 Final Test Accuracy: {:.2f}%".format(test_accuracy*100))
