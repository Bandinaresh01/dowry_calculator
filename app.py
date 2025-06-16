import pickle
import pandas as pd
from flask import Flask, request, render_template
import numpy as np

# Initialize Flask app
app = Flask(__name__)

# Load the trained model
with open("C:\programming\machine_learning_projects\host_project_dowry\model.pkl", "rb") as file:
    model = pickle.load(file)

# Feature columns (ensure the order matches the training data)
FEATURE_COLUMNS = [
    "Age", "Income", "Family_Wealth", "Siblings",
    "Marital_Status_First", "Marital_Status_Second",
    "Education_Bachelor's", "Education_Doctorate", "Education_Master's",
    "Job_Type_Business", "Job_Type_Engineer", "Job_Type_Govt Job",
    "Job_Type_Professor", "Job_Type_Researcher", "Job_Type_Software Dev.",
    "Job_Type_Teacher", "Location_Rural", "Location_Urban",
    "Property_Ownership_No Property", "Property_Ownership_Owns Flat",
    "Property_Ownership_Owns House", "Caste_Community_General",
    "Caste_Community_OBC", "Caste_Community_SC", "Caste_Community_ST",
    "Religious_Affiliation_Buddhist", "Religious_Affiliation_Christian",
    "Religious_Affiliation_Hindu", "Religious_Affiliation_Muslim",
    "Religious_Affiliation_Sikh"
]

@app.route("/")
def home():
    return render_template("index.html")  # HTML file with the form

@app.route("/predict", methods=["POST"])
def predict():
    # Collect inputs from the form
    age = int(request.form.get("age"))
    income = int(request.form.get("income"))
    family_wealth = int(request.form.get("family_wealth"))
    siblings = int(request.form.get("siblings"))
    marital_status = request.form.get("material_status")
    education = request.form.get("education")
    job_type = request.form.get("job_type")
    locality = request.form.get("locality")
    property_ownership = request.form.get("house")
    religious_affiliation = request.form.get("religious_affiliation")
    
    # Construct a DataFrame for the input
    input_dict = {
        "Age": [age],
        "Income": [income],
        "Family_Wealth": [family_wealth],
        "Siblings": [siblings],
        "Marital_Status": [marital_status],
        "Education": [education],
        "Job_Type": [job_type],
        "Location": [locality],
        "Property_Ownership": [property_ownership],
        "Religious_Affiliation": [religious_affiliation]
    }
    input_df = pd.DataFrame(input_dict)

    # Apply get_dummies to match the training format
    input_df = pd.get_dummies(input_df)
    input_df = input_df.reindex(columns=FEATURE_COLUMNS, fill_value=0)

# # Define the input data as a single-row numpy array
#     input_data = np.array([32, 75122, 11619256, 3, 0, 1, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0]).reshape(1, -1)

#     # Check the shape to ensure compatibility with the model
#     print("Input data shape:", input_data.shape)


        # Use the model to predict
    prediction = model.predict(input_df)

        # Return the prediction result
    return render_template("prediction.html", prediction=prediction)

if __name__ == "__main__":
    app.run(debug=True)

