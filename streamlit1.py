import streamlit as st
import pandas as pd
import pickle
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

# Load the Decision Tree model from the pickle file
try:
    with open('dt_model.pickle', 'rb') as file:
        model = pickle.load(file)
except Exception as e:
    st.error(f"Error loading the model: {e}")

# Define columns_order globally
# columns_order = ['LBE', 'LB', 'AC', 'ASTV', 'mSTV', 'ALTV', 'MLTV', 'DP', 'Width', 'Min', 'Max', 'Mode', 'Mean', 'Median', 'Variance']
columns_order = ['LBE',	'LB',	'AC',	'FM',	'UC',	'ASTV',	'MSTV',	'ALTV',	'MLTV',	'DL',	'DP',	'Width', 'Min', 'Max',	'Nmax',	'Mode',	'Mean',	'Median',	'Variance',	'Tendency']

# Function to preprocess input data
def preprocess_input(data):
    df = pd.DataFrame([data], columns=columns_order)  # Use a list to create a single-row DataFrame
    return df

# Function to predict NSP and display accuracy
def predict_nsp(input_data):
    # Preprocess input data
    input_df = preprocess_input(input_data)

    # Make predictions
    predictions = model.predict(input_df)

    return predictions

# Function to evaluate model accuracy
def evaluate_accuracy(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    return accuracy

# Streamlit web app
def main():
    st.title("Decision Tree NSP Predictor")
    st.sidebar.header("User Input")

    # Get user input for features
    user_input = {}
    for column in columns_order:
        user_input[column] = st.sidebar.number_input(f"Masukkan {column}", value=0.0)

    # Make prediction
    if st.sidebar.button("Predict NSP"):
        input_data = [user_input[column] for column in columns_order]
        prediction = predict_nsp(input_data)
        st.write(f"Predicted NSP: {prediction[0]}")

    # Evaluate model accuracy
    data = pd.read_excel("CTG_resampled.xlsx")  # ini diisi dengan data faktual
    features_to_exclude = ['DR', 'DS', 'Nzeros', 'FileName', 'Date', 'SegFile','A','B','C','D','E','AD','DE','LD','FS','SUSP','CLASS','b','e'] # ini kolom yang dikecualikan, ada 5
    X = data.drop(['NSP'] + features_to_exclude, axis=1)  # pengecualian kolom target dan kolom fitur yang tidak diinginkan, jadi disini ada 6 kolom yang dihindari
    y = data['NSP']  # Menggunakan kolom 'NSP' sebagai target


    accuracy = evaluate_accuracy(X, y)
    st.write(f"Model Accuracy: {accuracy:.2%}")

if __name__ == "__main__":
    main()
