"""
ML Model Comparison Script - OPTIMIZED FOR SPEED
Faster version with reduced complexity and subset sampling
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
import time
import warnings
warnings.filterwarnings('ignore')
import os
start_time = time.time()

# Load data
print("Loading data...")
df = pd.read_csv('C:\\APP\\leeds University\\extra_curricular\\leedshack\\output\\MASTER_timeseries_combined.csv')

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

X = df[feature_cols]
y = df['target']

# Encode target
le = LabelEncoder()
y_encoded = le.fit_transform(y)

print(f"Original dataset size: {len(X):,} samples")

# SPEED OPTIMIZATION: Use subset for very large datasets
if len(X) > 50000:
    print(f"⚡ Large dataset detected. Using 50,000 sample subset for faster training...")
    X_subset, _, y_subset, _ = train_test_split(
        X, y_encoded, train_size=50000, random_state=42, stratify=y_encoded
    )
    X = X_subset
    y_encoded = y_subset

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
)

print(f"✅ Using X_train {X_train.shape}, X_test {X_test.shape}")
print(f"Classes: {le.classes_}\n")

# OPTIMIZED MODELS - Reduced complexity for speed
models = {
    'KNN (k=5)': KNeighborsClassifier(n_neighbors=5, n_jobs=-1),
    'Decision Tree': DecisionTreeClassifier(random_state=42, max_depth=8),
    'Random Forest': RandomForestClassifier(n_estimators=50, random_state=42, 
                                           max_depth=8, n_jobs=-1),
    'Gradient Boosting': GradientBoostingClassifier(n_estimators=50, random_state=42,
                                                    max_depth=5),
    'Logistic Regression': LogisticRegression(max_iter=500, random_state=42, n_jobs=-1),
    'Naive Bayes': GaussianNB()
}

# Train and evaluate models
results = []

for name, model in models.items():
    model_start = time.time()
    print(f"Training {name}...")

    # Train model
    model.fit(X_train, y_train)

    # Predictions
    y_pred = model.predict(X_test)

    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
    recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)

    model_time = time.time() - model_start

    results.append({
        'Model': name,
        'Accuracy': accuracy,
        'Precision': precision,
        'Recall': recall,
        'F1-Score': f1,
        'Time (sec)': model_time
    })

    print(f"  ✓ Accuracy: {accuracy:.4f}, F1: {f1:.4f}, Time: {model_time:.1f}s\n")

# Create results dataframe
results_df = pd.DataFrame(results)
results_df = results_df.sort_values('Accuracy', ascending=False)

total_time = time.time() - start_time

print("\n" + "="*80)
print("FINAL RESULTS")
print("="*80)
print(results_df.to_string(index=False))
print(f"\nTotal execution time: {total_time:.1f} seconds")

# Save results
results_df.to_csv('model_comparison_results.csv', index=False)
print("✅ Results saved to 'model_comparison_results.csv'")

# Create visualizations
fig, axes = plt.subplots(2, 2, figsize=(15, 10))
fig.suptitle('ML Model Performance Comparison (Optimized)', fontsize=16, fontweight='bold')

# Plot 1: Accuracy Comparison
ax1 = axes[0, 0]
colors1 = plt.cm.Blues(np.linspace(0.4, 0.8, len(results_df)))
bars1 = ax1.barh(results_df['Model'], results_df['Accuracy'], color=colors1)
ax1.set_xlabel('Accuracy', fontsize=12)
ax1.set_title('Test Set Accuracy', fontsize=13, fontweight='bold')
ax1.set_xlim(0, 1)
ax1.grid(axis='x', alpha=0.3)
for i, bar in enumerate(bars1):
    width = bar.get_width()
    ax1.text(width + 0.01, bar.get_y() + bar.get_height()/2, 
             f'{width:.3f}', ha='left', va='center', fontsize=10, fontweight='bold')

# Plot 2: Multiple Metrics Comparison
ax2 = axes[0, 1]
x = np.arange(len(results_df))
width = 0.2
ax2.bar(x - width*1.5, results_df['Accuracy'], width, label='Accuracy', color='#3498db')
ax2.bar(x - width*0.5, results_df['Precision'], width, label='Precision', color='#e74c3c')
ax2.bar(x + width*0.5, results_df['Recall'], width, label='Recall', color='#2ecc71')
ax2.bar(x + width*1.5, results_df['F1-Score'], width, label='F1-Score', color='#f39c12')
ax2.set_ylabel('Score', fontsize=12)
ax2.set_title('All Metrics Comparison', fontsize=13, fontweight='bold')
ax2.set_xticks(x)
ax2.set_xticklabels(results_df['Model'], rotation=45, ha='right', fontsize=10)
ax2.legend(loc='lower right')
ax2.set_ylim(0, 1.05)
ax2.grid(axis='y', alpha=0.3)

# Plot 3: Training Time Comparison
ax3 = axes[1, 0]
colors3 = plt.cm.RdYlGn_r(np.linspace(0.3, 0.7, len(results_df)))
bars3 = ax3.barh(results_df['Model'], results_df['Time (sec)'], color=colors3)
ax3.set_xlabel('Training Time (seconds)', fontsize=12)
ax3.set_title('Model Training Time', fontsize=13, fontweight='bold')
ax3.grid(axis='x', alpha=0.3)
for i, bar in enumerate(bars3):
    width = bar.get_width()
    ax3.text(width + 0.1, bar.get_y() + bar.get_height()/2, 
             f'{width:.1f}s', ha='left', va='center', fontsize=10, fontweight='bold')

# Plot 4: F1-Score Comparison
ax4 = axes[1, 1]
colors4 = plt.cm.viridis(np.linspace(0.2, 0.8, len(results_df)))
bars4 = ax4.barh(results_df['Model'], results_df['F1-Score'], color=colors4)
ax4.set_xlabel('F1-Score', fontsize=12)
ax4.set_title('F1-Score (Weighted)', fontsize=13, fontweight='bold')
ax4.set_xlim(0, 1)
ax4.grid(axis='x', alpha=0.3)
for i, bar in enumerate(bars4):
    width = bar.get_width()
    ax4.text(width + 0.01, bar.get_y() + bar.get_height()/2, 
             f'{width:.3f}', ha='left', va='center', fontsize=10, fontweight='bold')
output_dir = 'C:\\APP\\leeds University\\extra_curricular\\leedshack\\outputpng'
os.makedirs(output_dir, exist_ok=True)
plt.tight_layout()
output_file = os.path.join(output_dir, 'model_comparison_plots.png')
plt.savefig(output_file, dpi=300, bbox_inches='tight')
print(f"✅ Plots saved to '{output_file}'")
plt.show()

print(f"\n🎉 Analysis complete in {total_time:.1f} seconds!")
