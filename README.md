# ABC Tech ITSM Intelligence – Flask App

This Flask app wraps the saved ABC Tech ML models from the supplied Colab notebook into one web application.

## Implemented capabilities

1. Task 1 – High Priority Prediction (XGBoost)
2. Task 2 – Incident Forecasting (Drift / Extra Trees)
3. Task 3 – Priority + Department Auto Tagging
4. Task 4 – RFC Prediction + Asset Failure Risk

## Exact notebook feature sets

### Task 1 X
- CI_Cat
- CI_Subcat
- WBS
- Impact
- Urgency
- number_cnt
- Category
- KB_number
- Alert_Status
- No_of_Related_Interactions
- Open_Year
- Open_Month
- Open_DayOfWeek
- Open_Hour
- Is_Weekend

Y: High_Priority

### Task 3 X
- CI_Subcat
- WBS
- Category
- Impact
- Urgency
- Open_Hour
- Open_DayOfWeek
- Open_Month
- Open_Quarter
- Open_IsWeekend

Y priority: Priority
Y department: CI_Cat

### Task 4 X
The app uses all 45 features from the notebook and reads the feature metadata from the saved Task 4 packages when available.

RFC target: RFC_Target
Asset target: Asset_Failure_Risk_Target

## Forecasting

The upload page expects a CSV containing `Open_Time`, matching the notebook's time-series construction. It:

- parses Open_Time using DD/MM/YYYY day-first by default;
- counts incidents per month;
- fills missing months with zero;
- forecasts 1–24 months;
- shows numerical monthly and quarterly forecasts;
- shows an interactive graph;
- calculates forecast total, average monthly volume and peak month;
- displays model strengths/weaknesses and the model validation context from the notebook.

## First-time setup in PyCharm

1. Create/open this project folder in PyCharm.
2. Put the seven model files into `models/`.
3. Open the PyCharm Terminal.
4. Create a virtual environment.
5. Install the requirements.
6. Run `python app.py`.
7. Open the local address printed by Flask, normally `http://127.0.0.1:5000`.

## Recommended Python environment

Your installed Python is 3.14.5. Current scikit-learn supports Python 3.14, and XGBoost provides a Python 3.14 classifier in its current documentation. pandas 2.3.3 was the first broadly compatible pandas release for Python 3.14. The bigger issue for serialized models is matching the libraries used when the pickle/joblib files were created. If a model fails to load, check the exact Colab versions first and install those compatible versions in the PyCharm venv.

## Important Task 4 production note

The notebook's 45 Task 4 features include historical CI, Category and WBS statistics. The form exposes them all so the saved model can be tested immediately. In a production app, these should be generated automatically from your incident/change/failure database.

## Deployment later

Once the local app works, it can be packaged for a cloud web service. Keep model files in object storage or a mounted model directory when the platform imposes image-size or artifact-size constraints.
