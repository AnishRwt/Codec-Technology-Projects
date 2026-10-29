# Employee Attrition Analysis

## Project Overview
This project analyzes employee attrition using HR data to identify workforce patterns associated with employees leaving an organization. It combines exploratory data analysis (EDA), business insights, and machine-learning classification models.

## Objectives
- Measure the overall employee attrition rate.
- Analyze attrition by department, overtime, job satisfaction, age, and monthly income.
- Build and compare Logistic Regression and Random Forest models.
- Evaluate model performance using accuracy, precision, recall, and F1-score.
- Explore the impact of different classification thresholds on Random Forest predictions.

## Dataset
- **Records:** 1,470 employees
- **Original columns:** 35
- **Constant columns removed:** EmployeeCount, Over18, StandardHours
- **Target variable:** Attrition (Yes/No)

## Key Business Findings
1. **Overall attrition:** 237 employees left, representing 16.12% of the dataset.
2. **Department:** Sales had the highest department-level attrition rate at 20.63%.
3. **Overtime:** Employees working overtime had a 30.53% attrition rate, compared with 10.44% among those not working overtime.
4. **Job satisfaction:** The lowest satisfaction group had a 22.84% attrition rate, compared with 11.33% for the highest satisfaction group.
5. **Monthly income:** The below-3,000 income group had the highest attrition rate at 28.61%, while the above-10,000 group had the lowest at 8.90%.

These findings represent associations in the dataset and do not establish causation.

## Machine Learning Models

Two classification models were evaluated on a held-out test set of 294 records.

| Model | Accuracy | Precision — Left | Recall — Left | F1-score — Left |
|---|---:|---:|---:|---:|
| Logistic Regression | 75.85% | 36.05% | 65.96% | 46.62% |
| Random Forest (default threshold) | 84.35% | 57.14% | 8.51% | 14.81% |

Random Forest achieved higher accuracy at its default threshold but identified far fewer employees who left. Logistic Regression provided substantially higher recall for the minority class.

### Random Forest Threshold Analysis
A threshold experiment showed that a threshold of 0.20 produced a 49.15% F1-score for the "Left" class, with 61.70% recall and 40.85% precision. A threshold of 0.15 achieved 76.60% recall but lower precision and accuracy.

These threshold results should be validated before selecting a final model.

## Tools and Technologies
- Python
- Pandas and NumPy
- Matplotlib and Seaborn
- Scikit-learn
- Google Colab
- CSV data export

## Project Deliverables
- Data exploration and cleaning notebook
- Exploratory data visualizations
- Employee attrition summary tables
- Model comparison results
- Random Forest threshold analysis
- Business findings and recommendations

## Conclusion
The analysis identified associations between employee attrition and overtime, job satisfaction, department, and monthly income. Model evaluation also demonstrated why accuracy alone can be misleading when the target classes are imbalanced. Recall, precision, and F1-score provide additional information about how well a model identifies employees who left.

## Limitations
The analysis is based on a single HR dataset. Observed relationships are not necessarily causal, and model predictions should not be used to make employment decisions about individual employees.

## Author
Anish Rawat

**Project Type:** Data Analytics Internship Project
