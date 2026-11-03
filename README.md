# Addiction Recovery Analytics Dashboard

An interactive Streamlit dashboard for exploring a student addiction dataset. It combines data visualization, machine learning prediction, community insights, and simulated real-time monitoring.

## Features

- **Dashboard overview:** key metrics, addiction distribution, risk-factor overview, and group comparisons.
- **Data explorer:** filter and inspect the dataset.
- **Correlation and time-series analysis:** heatmaps, trends, and simulated progression timelines.
- **ML predictions:** trains Random Forest, Gradient Boosting, Logistic Regression, and SVM models to predict addiction class, shows feature importance, a confusion matrix, and risk-probability distribution, and gives an individual prediction with personalized recommendations.
- **User segmentation:** groups users by risk level.
- **Community insights:** behavioral patterns, intervention points, user personas, engagement strategies, success metrics, and an implementation roadmap.
- **Real-time monitoring:** simulated live data ingestion with alerts and a live chart.

## Dataset

The project includes `attached_assets/student_addiction_dataset_train_*.csv` (about 50,000 lines). Columns are:

- Experimentation, Academic_Performance_Decline, Social_Isolation, Financial_Issues, Physical_Mental_Health_Problems, Legal_Consequences, Relationship_Strain, Risk_Taking_Behavior, Withdrawal_Symptoms, Denial_and_Resistance_to_Treatment (risk factors, Yes/No)
- Addiction_Class (target label)

You can also upload your own CSV from the sidebar.

## Project Structure

| File | Purpose |
| --- | --- |
| `app.py` | Streamlit entry point and page layout |
| `data_processor.py` | Loading, cleaning, standardizing, and feature engineering |
| `ml_models.py` | Model training, evaluation, and prediction |
| `visualizations.py` | Plotly charts and statistical tests |
| `community_insights.py` | Pattern analysis, personas, and recommendations |
| `real_time_ingestion.py` | Simulated real-time data monitoring and alerts |
| `pyproject.toml` | Dependencies |

## Requirements

- Python 3.11 or higher
- Dependencies: matplotlib, numpy, openai, pandas, plotly, scikit-learn, scipy, seaborn, streamlit

## Installation

These are the same dependencies listed in `pyproject.toml`.

## Running the App

```bash
streamlit run app.py --server.port 5000
```

Then open http://localhost:5000 in your browser.


## License

Specify a license here, or remove this section.
