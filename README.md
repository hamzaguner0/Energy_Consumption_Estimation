# Energy_Consumption_Estimation

Energy Consumption Estimation Project
This project predicts energy consumption using simulated data. The goal is to develop a machine learning model that can accurately estimate energy usage based on features such as temperature, electricity price, month, weekday, and previous day's energy consumption.

Project Structure
plaintext
Copy code
Project Folder/
├── notebooks/
│   └── Energy Consumption Estimation.ipynb  # Jupyter notebook with full analysis and code
├── models/
│   └── energy_consumption_model.pkl         # Trained Random Forest model
├── data/
│   ├── simulated_energy_data.csv            # Simulated energy data used for training/testing
│   ├── train_set.csv                        # Training data
│   ├── test_set.csv                         # Test data without predictions
│   └── test_set_with_predictions.csv        # Test data with model predictions
How to Run the Project
Clone the Repository or Download Files: Download all project files and ensure they are in the appropriate structure.

Environment Setup:

Install Python (3.8 or above recommended).
Install required libraries:
bash
Copy code
pip install pandas scikit-learn
Run the Notebook: Open Energy Consumption Estimation.ipynb in Jupyter Notebook to see the full workflow, from data simulation to model training and evaluation.

Use the Trained Model: To use the trained model (energy_consumption_model.pkl) in another Python script, load it as follows:

python
Copy code
import joblib
model = joblib.load("models/energy_consumption_model.pkl")
predictions = model.predict(new_data)
Files Description
Energy Consumption Estimation.ipynb: Contains all the steps of the project, including:

Data simulation
Preprocessing and feature engineering
Model training (Linear Regression, Random Forest)
Evaluation (MAE, MSE)
Predictions
simulated_energy_data.csv: The initial dataset with features such as temperature, electricity price, and energy consumption.

train_set.csv: Subset of the data used for model training.

test_set.csv: Subset of the data used for testing model performance.

test_set_with_predictions.csv: Test set with an additional column showing the model's predictions.

energy_consumption_model.pkl: The trained Random Forest model saved for reuse.

Key Results
Model Performance (Random Forest):

Mean Absolute Error (MAE): 6.71
Mean Squared Error (MSE): 57.43
The trained model can predict energy consumption based on the given features.

Future Improvements
Use real-world energy consumption data instead of simulated data.
Add more advanced models (e.g., Gradient Boosting or Neural Networks).
Implement hyperparameter tuning for better performance.
Visualize predictions and model performance more effectively.
Contact
For any questions or feedback, feel free to contact the project creator.

