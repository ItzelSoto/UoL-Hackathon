import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
df = pd.read_csv('C:\\APP\\leeds University\\extra_curricular\\leedshack\\output\\MASTER_timeseries_combined.csv')
print(f"✓ Loaded {df.shape[0]:,} rows and {df.shape[1]} columns")
print("\n[2/6] Preparing features and target...")
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

X = df[feature_cols].copy()
y = df['target'].copy()
le_target = LabelEncoder()
y_encoded = le_target.fit_transform(y)
print(f"✓ Features: {X.shape}")
print(f"✓ Target classes: {le_target.classes_}")
print(f"\n✓ Class distribution:")
for idx, label in enumerate(le_target.classes_):
    count = np.sum(y_encoded == idx)
    print(f"   {label}: {count:,} ({count/len(y_encoded)*100:.1f}%)")
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded,
    test_size=0.2,
    random_state=42,
    stratify=y_encoded
)
xgb_model = xgb.XGBClassifier(
    n_estimators=200,           # Number of trees
    max_depth=8,                # Maximum tree depth
    learning_rate=0.1,          # Step size shrinkage
    subsample=0.8,              # Fraction of samples for each tree
    colsample_bytree=0.8,       # Fraction of features for each tree
    objective='multi:softmax',  # Multi-class classification
    num_class=4,                # Number of classes
    eval_metric='mlogloss',     # Evaluation metric
    random_state=42,
    n_jobs=-1,                  # Use all CPU cores
    tree_method='hist'          # Faster training
)
eval_set = [(X_train, y_train), (X_test, y_test)]
xgb_model.fit(
    X_train, y_train,
    eval_set=eval_set,
    verbose=50  # Print every 50 iterations
)
y_pred = xgb_model.predict(X_test)
y_pred_proba = xgb_model.predict_proba(X_test)

# Accuracy
accuracy = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred, average='weighted')
print(f"\n✓ Test Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
print(f"✓ Weighted F1-Score: {f1:.4f}")

# Classification report
print("\n📊 Classification Report:")
print(classification_report(y_test, y_pred, target_names=le_target.classes_))
print("\n📊 Confusion Matrix:")
cm = confusion_matrix(y_test, y_pred)
print(cm)
feature_importance = pd.DataFrame({
    'feature': feature_cols,
    'importance': xgb_model.feature_importances_
}).sort_values('importance', ascending=False)
print("\n🔝 Top 15 Most Important Features:")
print(feature_importance.head(15).to_string(index=False))
print("\n" + "=" * 80)
print("CREATING VISUALIZATIONS")
print("=" * 80)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=le_target.classes_,
            yticklabels=le_target.classes_)
plt.title('Confusion Matrix - XGBoost Model', fontsize=16, fontweight='bold')
plt.ylabel('True Label', fontsize=12)
plt.xlabel('Predicted Label', fontsize=12)
plt.tight_layout()
plt.savefig('xgboost_confusion_matrix.png', dpi=300, bbox_inches='tight')
print("\n✓ Saved: xgboost_confusion_matrix.png")
plt.close()
# 2. Feature Importance (Top 20)
plt.figure(figsize=(12, 8))
top_features = feature_importance.head(20)
plt.barh(range(len(top_features)), top_features['importance'])
plt.yticks(range(len(top_features)), top_features['feature'])
plt.xlabel('Importance Score', fontsize=12)
plt.title('Top 20 Most Important Features - XGBoost', fontsize=16, fontweight='bold')
plt.gca().invert_yaxis()
plt.tight_layout()
plt.savefig('xgboost_feature_importance.png', dpi=300, bbox_inches='tight')
print("✓ Saved: xgboost_feature_importance.png")
plt.close()

# 3. Training History (if available)
results = xgb_model.evals_result()
epochs = len(results['validation_0']['mlogloss'])
x_axis = range(0, epochs)

fig, ax = plt.subplots(figsize=(12, 6))
ax.plot(x_axis, results['validation_0']['mlogloss'], label='Train')
ax.plot(x_axis, results['validation_1']['mlogloss'], label='Test')
ax.legend()
ax.set_ylabel('Log Loss', fontsize=12)
ax.set_xlabel('Iteration', fontsize=12)
ax.set_title('XGBoost Training Progress', fontsize=16, fontweight='bold')
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('xgboost_training_curve.png', dpi=300, bbox_inches='tight')
print("✓ Saved: xgboost_training_curve.png")
plt.close()

# 4. Per-Class Performance
class_report = classification_report(y_test, y_pred, target_names=le_target.classes_, output_dict=True)
metrics_df = pd.DataFrame(class_report).transpose().iloc[:-3, :3]  # Exclude avg rows

fig, ax = plt.subplots(figsize=(10, 6))
metrics_df.plot(kind='bar', ax=ax, width=0.8)
ax.set_title('Per-Class Performance Metrics - XGBoost', fontsize=16, fontweight='bold')
ax.set_ylabel('Score', fontsize=12)
ax.set_xlabel('Class', fontsize=12)
ax.set_ylim([0, 1])
ax.legend(['Precision', 'Recall', 'F1-Score'])
ax.grid(True, alpha=0.3, axis='y')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('xgboost_class_metrics.png', dpi=300, bbox_inches='tight')
print("✓ Saved: xgboost_class_metrics.png")
plt.close()
# Save XGBoost model
xgb_model.save_model('xgboost_model.json')
joblib.dump(xgb_model, 'xgboost_model.pkl')
joblib.dump(le_target, 'xgboost_label_encoder.pkl')
# Prediction probabilities for sample students
print("\n📊 Sample Predictions (First 5 students):")
sample_df = pd.DataFrame({
    'True_Label': [le_target.classes_[i] for i in y_test[:5]],
    'Predicted_Label': [le_target.classes_[i] for i in y_pred[:5]],
    'Distinction_Prob': y_pred_proba[:5, 0],
    'Fail_Prob': y_pred_proba[:5, 1],
    'Pass_Prob': y_pred_proba[:5, 2],
    'Withdrawn_Prob': y_pred_proba[:5, 3]
})
print(sample_df.to_string(index=False))

# Model parameters
print("\n⚙️  Model Parameters:")
print(f"   Number of trees: {xgb_model.n_estimators}")
print(f"   Max depth: {xgb_model.max_depth}")
print(f"   Learning rate: {xgb_model.learning_rate}")
print(f"   Subsample: {xgb_model.subsample}")
print(f"\n🎯 Final Test Accuracy: {accuracy*100:.2f}%")
print(f"🎯 Weighted F1-Score: {f1:.4f}")

