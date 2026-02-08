# OULAD Time Series ML Setup

## Files Included:
1. `MASTER_timeseries_combined.csv` - The main dataset
2. `setup_ml_environment.py` - Full setup with example model
3. `quick_load.py` - Minimal code to just load data

## Dataset Info:
- **Rows**: 582,772 (student-week combinations)
- **Features**: 22 time-series features
- **Target**: 4 classes (Pass, Distinction, Fail, Withdrawn)
- **Time dimension**: 40 weeks (0-39)

## Key Variables After Loading:
- `X_train`, `X_test` - Training and test features
- `y_train`, `y_test` - Training and test labels (encoded 0-3)
- `student_ids` - Student identifiers for grouping sequences
- `le.classes_` - ['Distinction', 'Fail', 'Pass', 'Withdrawn']
