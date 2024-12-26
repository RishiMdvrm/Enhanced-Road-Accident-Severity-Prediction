
# Enhanced Road Accident Severity Prediction

## Project Overview
This project leverages machine learning techniques to predict the severity of road accidents in the United States. Using the comprehensive "US-Accidents" dataset from Kaggle, augmented with weather data and location metadata, the project explores multiple machine learning models to identify key factors influencing accident severity and provide accurate predictions. The results have potential applications in road safety, traffic management, and emergency response planning.

---

## Features
- **Data Augmentation:** Integration of weather data from the National Weather Service (NWS) API.
- **Machine Learning Models:** Evaluation of multiple algorithms, including Logistic Regression, Decision Trees, Random Forest, Gradient Boosting, SVM, and Multi-Layer Perceptron (MLP).
- **Performance Tuning:** Hyperparameter optimization using Grid Search.
- **Comprehensive Analysis:** Comparative analysis of model performance metrics like accuracy, precision, recall, and F1-score.

---

## Repository Structure
- `2. Weather data entry.py`: Python script for fetching and filling missing weather data in the dataset.
- `Project Report.pdf`: Detailed project report, including methodology, results, and future directions.

---

## Dataset
- **Source:** Kaggle's "[US-Accidents](https://www.kaggle.com/datasets/sobhanmoosavi/us-accidents)" dataset.
- **Size:** 7.7 million records.
- **Key Features:**
  - GPS Coordinates (Start Latitude, Start Longitude)
  - Weather Conditions (Temperature, Humidity, Wind Speed, etc.)
  - Road Metadata (Severity, Description, Traffic Signals, etc.)

---

## Methodology
### 1. Data Preprocessing
- **Handling Missing Values:** Missing weather data was fetched using the haversine formula to identify the nearest weather station.
- **Stratified Sampling:** Reduced data size for efficient processing.
- **Feature Engineering:** Grouped data by cities and states for better analysis.

### 2. Model Evaluation
- **Algorithms Used:**
  - Logistic Regression
  - Decision Trees
  - Random Forest
  - Gradient Boosting
  - Support Vector Machines (SVM)
  - Multi-Layer Perceptron (MLP)
- **Metrics:** Accuracy, Precision, Recall, F1-Score

### 3. Hyperparameter Tuning
- Grid Search was used to fine-tune model parameters.
- Gradient Boosting emerged as the top-performing model with an accuracy of 93.38% post-tuning.

---

## Results
### Performance Metrics (Post-Tuning):
| Model              | Accuracy | Precision | Recall | F1-Score |
|--------------------|----------|-----------|--------|----------|
| Gradient Boosting  | 93.38%   | 0.93      | 0.93   | 0.93     |
| Random Forest      | 92.00%   | 0.92      | 0.92   | 0.92     |
| Decision Trees     | 91.40%   | 0.91      | 0.91   | 0.91     |
| Multi-Layer Perceptron | 86.38% | 0.86    | 0.86   | 0.86     |
| Logistic Regression | 80.37%  | 0.78      | 0.80   | 0.79     |

---

## Installation and Usage
### Prerequisites
- Python 3.7+
- Required Python Libraries:
  - `pandas`
  - `numpy`
  - `requests`
  - `haversine`
  - `pytz`
  - `scikit-learn`

### Setup
1. Clone the repository.
2. Install the required dependencies using pip:
   ```bash
   pip install -r requirements.txt
   ```
3. Update the file paths in `2. Weather data entry.py` to match your local environment.

### Running the Weather Data Script
- Use the script `2. Weather data entry.py` to fetch and fill missing weather data:
  ```bash
  python 2. Weather data entry.py
  ```
- The updated dataset will be saved to the specified output path.

---

## Future Work
- Integration of real-time data for dynamic predictions.
- Exploration of advanced deep learning architectures (CNNs, RNNs) for unstructured data.
- Expansion of the dataset to include diverse geographical regions.

---

## Contributors
- **Manan Jain**  
  Email: mjain35@uic.edu
- **Hemanth Nagulapalli**  
  Email: hnagul2@uic.edu
- **Rishi Madhavaran**  
  Email: rmadha4@uic.edu
- **Francis Pagulayan**  
  Email: ppagu2@uic.edu

---

## References
1. [US-Accidents Dataset](https://www.kaggle.com/datasets/sobhanmoosavi/us-accidents)
2. [Scikit-learn Documentation](https://scikit-learn.org/stable/)
3. [Folium Library](https://pypi.org/project/folium/)
4. [Matplotlib API](https://matplotlib.org/stable/api/index)
