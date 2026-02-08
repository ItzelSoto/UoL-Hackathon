
import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy import stats
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
studentVle = pd.read_csv("C:\\APP\\leeds University\\extra_curricular\\leedshack\\anonymisedData\\studentVle.csv")
vle = pd.read_csv("C:\\APP\\leeds University\\extra_curricular\\leedshack\\anonymisedData\\vle.csv")
studentInfo = pd.read_csv("C:\\APP\\leeds University\\extra_curricular\\leedshack\\anonymisedData\\studentInfo.csv")
studentRegistration = pd.read_csv("C:\\APP\\leeds University\\extra_curricular\\leedshack\\anonymisedData\\studentRegistration.csv")
assessments = pd.read_csv("C:\\APP\\leeds University\\extra_curricular\\leedshack\\anonymisedData\\assessments.csv")
studentAssessment = pd.read_csv("C:\\APP\\leeds University\\extra_curricular\\leedshack\\anonymisedData\\studentAssessment.csv")
courses = pd.read_csv("C:\\APP\\leeds University\\extra_curricular\\leedshack\\anonymisedData\\courses.csv")
output_dir = 'C:\\APP\\leeds University\\extra_curricular\\leedshack\\output'
os.makedirs(output_dir, exist_ok=True)
studentInfo['imd_band'].dropna()
le_gender = LabelEncoder()
le_region = LabelEncoder()
le_education = LabelEncoder()
le_imd = LabelEncoder()
le_age = LabelEncoder()
le_disability = LabelEncoder()
studentInfo['gender_encoded'] = le_gender.fit_transform(studentInfo['gender'])
studentInfo['region_encoded'] = le_region.fit_transform(studentInfo['region'])
studentInfo['highest_education_encoded'] = le_education.fit_transform(studentInfo['highest_education'])
studentInfo['imd_band_encoded'] = le_imd.fit_transform(studentInfo['imd_band'])
studentInfo['age_band_encoded'] = le_age.fit_transform(studentInfo['age_band'])
studentInfo['disability_encoded'] = le_disability.fit_transform(studentInfo['disability'])
# Encode code_module
le_module = LabelEncoder()
studentInfo['code_module_encoded'] = le_module.fit_transform(studentInfo['code_module'])
# Encode code_presentation
le_presentation = LabelEncoder()
studentInfo['code_presentation_encoded'] = le_presentation.fit_transform(studentInfo['code_presentation'])
# Merge with vle to get activity types
studentVle_full = studentVle.merge(vle[['id_site', 'activity_type']], on='id_site', how='left')
# Create week column
studentVle_full['week'] = (studentVle_full['date'] // 7).clip(0, 39)
# Aggregate weekly metrics using groupby
print("   Aggregating weekly clicks and interactions...")
weekly_agg = studentVle_full.groupby(['code_module', 'code_presentation', 'id_student', 'week']).agg({
    'sum_click': 'sum',
    'id_site': 'count',
    'activity_type': 'nunique'
}).reset_index()
weekly_agg.columns = ['code_module', 'code_presentation', 'id_student', 'week', 
                      'weekly_clicks', 'weekly_interactions', 'unique_activities']
weekly_agg = weekly_agg.sort_values(['code_module', 'code_presentation', 'id_student', 'week'])
weekly_agg['cumulative_clicks'] = weekly_agg.groupby(['code_module', 'code_presentation', 'id_student'])['weekly_clicks'].cumsum()
# Merge to get dates
assessments_with_dates = studentAssessment.merge(
    assessments[['id_assessment', 'date']], 
    on='id_assessment', 
    how='left'
)
assessments_with_dates['week'] = (assessments_with_dates['date'] // 7).clip(0, 39)
# Sort for cumulative calculations
assessments_with_dates = assessments_with_dates.sort_values(['id_student', 'week'])
# Calculate cumulative assessment metrics
print("   Calculating cumulative assessment scores...")
assess_cumsum = assessments_with_dates.groupby(['id_student', 'week']).agg({
    'score': ['sum', 'count'],
    'is_banked': 'sum'
}).reset_index()
assess_cumsum.columns = ['id_student', 'week', 'score_sum', 'score_count', 'banked_count']
# Cumulative sum across weeks
assess_cumsum = assess_cumsum.sort_values(['id_student', 'week'])
assess_cumsum['cumulative_score_sum'] = assess_cumsum.groupby('id_student')['score_sum'].cumsum()
assess_cumsum['cumulative_assessments'] = assess_cumsum.groupby('id_student')['score_count'].cumsum()
assess_cumsum['cumulative_banked'] = assess_cumsum.groupby('id_student')['banked_count'].cumsum()
assess_cumsum['cumulative_avg_score'] = assess_cumsum['cumulative_score_sum'] / assess_cumsum['cumulative_assessments']
assess_features = assess_cumsum[['id_student', 'week', 'cumulative_avg_score', 'cumulative_assessments', 'cumulative_banked']]
print("\nStep 6: Merging time series features...")
time_series_features = weekly_agg.merge(
    assess_features,
    on=['id_student', 'week'],
    how='left'
)
time_series_features['cumulative_avg_score'].fillna(0, inplace=True)
time_series_features['cumulative_assessments'].fillna(0, inplace=True)
time_series_features['cumulative_banked'].fillna(0, inplace=True)
print("\nStep 7: Adding static student features...")
time_series_full = time_series_features.merge(
    studentInfo[['code_module', 'code_presentation', 'id_student', 'studied_credits', 
             'gender_encoded', 'region_encoded', 'highest_education_encoded',
             'imd_band_encoded', 'age_band_encoded', 'disability_encoded', 
             'code_module_encoded', 'code_presentation_encoded',  # ADD THIS
             'final_result']],

    on=['code_module', 'code_presentation', 'id_student'],
    how='left'
)
registration_dates = studentRegistration.groupby(['code_module', 'code_presentation', 'id_student']).agg({
    'date_registration': 'first'
}).reset_index()
time_series_full = time_series_full.merge(
    registration_dates,
    on=['code_module', 'code_presentation', 'id_student'],
    how='left'
)
time_series_full['date_registration'].fillna(0, inplace=True)
time_series_full = time_series_full.sort_values(['code_module', 'code_presentation', 'id_student', 'week'])
time_series_full['prev_week_clicks'] = time_series_full.groupby(['code_module', 'code_presentation', 'id_student'])['weekly_clicks'].shift(1).fillna(0)
time_series_full['prev_week_interactions'] = time_series_full.groupby(['code_module', 'code_presentation', 'id_student'])['weekly_interactions'].shift(1).fillna(0)
time_series_full['rolling_avg_clicks_3w'] = time_series_full.groupby(['code_module', 'code_presentation', 'id_student'])['weekly_clicks'].transform(lambda x: x.rolling(window=3, min_periods=1).mean())
time_series_full['rolling_avg_interactions_3w'] = time_series_full.groupby(['code_module', 'code_presentation', 'id_student'])['weekly_interactions'].transform(lambda x: x.rolling(window=3, min_periods=1).mean())
feature_columns_ts = [
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

X_timeseries = time_series_full[feature_columns_ts]
y_timeseries = time_series_full['final_result']

# Remove rows with missing target
valid_indices = y_timeseries.notna()
X_timeseries = X_timeseries[valid_indices]
y_timeseries = y_timeseries[valid_indices]

student_ids = time_series_full[valid_indices][['code_module', 'code_presentation', 'id_student', 'week']].reset_index(drop=True)
X_timeseries = X_timeseries.reset_index(drop=True)
y_timeseries = y_timeseries.reset_index(drop=True)
X_timeseries.to_csv(os.path.join(output_dir, 'X_timeseries_features.csv'), index=False)
y_timeseries.to_csv(os.path.join(output_dir, 'y_timeseries_target.csv'), index=False)
student_ids.to_csv(os.path.join(output_dir, 'student_ids_timeseries.csv'), index=False)
# Create ONE master dataframe with EVERYTHING
combined_dataset = pd.concat([
    student_ids,     # id_student, code_module, code_presentation, week not features but useful for reference
    X_timeseries,    # All 22 features
    y_timeseries     # Target variable
], axis=1)
combined_dataset.rename(columns={'final_result': 'target'}, inplace=True)
combined_dataset.to_csv(os.path.join(output_dir, 'MASTER_timeseries_combined.csv'), index=False)
