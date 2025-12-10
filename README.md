
# Uber Fare Prediction using Machine Learning

This project predicts **Uber trip fares** based on historical trip data using a combination of **machine learning algorithms**, feature engineering techniques, and a fully interactive **Streamlit web application**.  
The objective is to build an end-to-end ML pipeline — from data exploration and preprocessing to training, evaluation, and deployment.

---

## 🚀 Key Features

### 1. **End-to-End Machine Learning Pipeline**
- Data cleaning and preprocessing  
- Feature engineering  
- Coordinate distance calculation (Haversine formula)  
- Handling outliers and missing values  
- Train/Test split automation  

### 2. **Multiple ML Models Implemented**
- Linear Regression  
- Random Forest Regressor  
- Decision Tree Regressor  
- K-Nearest Neighbors (KNN)  
- Gradient Boosting  

Model comparison is provided through:
- RMSE (Root Mean Squared Error)  
- MAE (Mean Absolute Error)  
- R² Score  

### 3. **Streamlit Web Application**
A full user-friendly webapp where users can:
- Input pickup & drop coordinates  
- Select passenger count  
- View predicted fare instantly  
- Visualize dataset insights  
- Compare model performances  

### 4. **Production-Ready Model**
- Model trained and saved using **joblib**  
- Streamlit UI for real-time predictions  
- Modular code for easy updates  

---

## 📂 Project Structure

```
Uber-Fare-Prediction/
│── app.py                     # Streamlit application
│── model.py                   # Model training pipeline
│── utils.py                   # Helper functions (distance calculation etc.)
│── uber.csv                   # Dataset
│── model.pkl                  # Trained ML model (ignored if large)
│── README.md                  # Project documentation
```

---

## 📊 Dataset Overview

The dataset contains Uber pickup/drop-off details such as:

- `fare_amount`  
- `pickup_datetime`  
- `pickup_longitude`, `pickup_latitude`  
- `dropoff_longitude`, `dropoff_latitude`  
- `passenger_count`  

Data preprocessing includes:

- Removing invalid lat/long values  
- Filtering unrealistic fares  
- Extracting datetime components  
- Calculating distance using Haversine formula  

---

## 🔍 Exploratory Data Analysis (EDA)

The following insights were visualized:

- Fare distribution  
- Passenger count trends  
- Distance vs Fare correlation  
- Outlier detection  
- Heatmaps and scatterplots  

Visualizations were built using:
- Matplotlib  
- Seaborn  
- Plotly (optional)  

---

## 🧠 Model Training & Evaluation

Models tested:

| Model | RMSE | MAE | R² |
|-------|------|------|------|
| Linear Regression | Good baseline | – | – |
| Random Forest | Best overall | – | – |
| Gradient Boosting | Strong performer | – | – |

(Randomized Search CV can be used for tuning.)

---

## 🖥️ Streamlit App Features

**User Inputs**
- Pickup longitude & latitude  
- Dropoff longitude & latitude  
- Passenger count  

**Outputs**
- Predicted fare  
- Interactive map visuals  
- Model performance charts  
- Dataset preview  

Run the app:

```
streamlit run app.py
```

---

## 🛠️ Installation & Setup

Clone the repository:

```
git clone https://github.com/yourusername/Uber-Fare-Prediction.git
cd Uber-Fare-Prediction
```

Create virtual environment:

```
python -m venv venv
venv\Scripts\activate   # Windows
source venv/bin/activate # Mac/Linux
```

Install dependencies:

```
pip install -r requirements.txt
```

Run Streamlit:

```
streamlit run app.py
```

---

## 📌 Technologies Used

- Python  
- Pandas  
- NumPy  
- Scikit-learn  
- Matplotlib  
- Streamlit  
- Joblib  
- Jupyter Notebook  

---

## 🧪 Future Improvements

You can add these to strengthen your resume:

- Add hyperparameter tuning (GridSearchCV / RandomSearchCV)  
- Add XGBoost and CatBoost models  
- Integrate geolocation maps (Folium)  
- Deploy on Streamlit Cloud / Render / HuggingFace Spaces  
- Add a REST API using FastAPI or Flask  
- Use MLflow for model tracking  
- Create Docker container for deployment  

---

## 🙌 Contribution

Pull requests are welcome. For major changes, open an issue first to discuss your ideas.

---

## 📄 License

This project is licensed under the MIT License.

---

## 📧 Contact

For inquiries or collaboration:

**Muhammad Qaseem**  
Email: qaseem2109@gmail.com 
GitHub: https://github.com/qaseem2109
