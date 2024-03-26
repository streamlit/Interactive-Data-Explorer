#!/usr/bin/env python
# coding: utf-8

# In[15]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
import streamlit as st


# In[16]:


#read csv files
df = pd.read_csv("Loan train data.csv")
df_test = pd.read_csv("New Customer.csv")


# In[17]:


#drop unimportant data
df.drop(columns=['Loan_ID'], axis=1, inplace=True)
df_test.drop(columns=['Loan_ID'], axis=1, inplace=True)


# In[18]:


#encoding some data
label_encoder = LabelEncoder()
df['Loan_Status'] = label_encoder.fit_transform(df['Loan_Status'])
df['Education'] = label_encoder.fit_transform(df['Education'])
df['Married'] = label_encoder.fit_transform(df['Married'])
df['Self_Employed'] = label_encoder.fit_transform(df['Self_Employed'])
df['Property_Area'] = label_encoder.fit_transform(df['Property_Area'])
df['Gender'] = label_encoder.fit_transform(df['Gender'])

df_test['Education'] = label_encoder.fit_transform(df_test['Education'])
df_test['Married'] = label_encoder.fit_transform(df_test['Married'])
df_test['Self_Employed'] = label_encoder.fit_transform(df_test['Self_Employed'])
df_test['Property_Area'] = label_encoder.fit_transform(df_test['Property_Area'])
df_test['Gender'] = label_encoder.fit_transform(df_test['Gender'])


# In[19]:


#fill NaNs and replace strings
df['LoanAmount'].fillna(df['LoanAmount'].mean(), inplace=True)
df['Loan_Amount_Term'].fillna(df['Loan_Amount_Term'].mean(), inplace=True)
df['Dependents'] = df['Dependents'].fillna(df['Dependents'].mode()[0])
df['Credit_History'] = df['Credit_History'].fillna(df['Credit_History'].mode()[0])
df['Dependents'] = np.where(df['Dependents'] == '3+', 3, df['Dependents'])
df['Dependents'] = df['Dependents'].astype(str).astype(int)

df_test['LoanAmount'].fillna(df_test['LoanAmount'].mean(), inplace=True)
df_test['Loan_Amount_Term'].fillna(df_test['Loan_Amount_Term'].mean(), inplace=True)
df_test['Dependents'] = df_test['Dependents'].fillna(df_test['Dependents'].mode()[0])
df_test['Credit_History'] = df_test['Credit_History'].fillna(df_test['Credit_History'].mode()[0])
df_test['Dependents'] = np.where(df_test['Dependents'] == '3+', 3, df_test['Dependents'])
df_test['Dependents'] = df_test['Dependents'].astype(str).astype(int)


# In[20]:


#train the model
X_train, X_test, y_train, y_test = train_test_split(df.drop('Loan_Status',axis=1), df['Loan_Status'], test_size=1/3, random_state=42)
classifier = LogisticRegression()
classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)


# In[21]:


cr = metrics.classification_report(y_test,y_pred)
cm = metrics.confusion_matrix(y_test,y_pred)
print("Classification Report:")
print(cr)
print("Confusion Matrix:")
print(cm)


# In[22]:


predictions = classifier.predict(df_test)
df_test['Predicted_Loan_Status'] = predictions


# In[23]:


score = accuracy_score(y_test, y_pred)
print ("Accuracy:")
print(str(score * 100) + '%')


# In[24]:


disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot()
plt.show()


# In[25]:


model=KNeighborsClassifier(n_neighbors=3)
model.fit(X_train,y_train)


# In[26]:


y_preds=model.predict(X_test)
print("Accuracy:")
print(str(accuracy_score(y_test,y_preds) * 100) + "%")


# In[27]:


predictions1 = model.predict(df_test.drop('Predicted_Loan_Status', axis=1))
df_test['Predicted_Loan_Status_With_KNN'] = predictions1


# In[28]:


df_test.head(51)


# In[29]:


# Define a function to make predictions
def make_prediction(input_features, model):
    prediction = model.predict(input_features)
    return prediction

def main():
    st.title("Loan Predictor")

    # Add a short description
    st.write("This model predicts if a person is eligible for a loan.")
    # Add input fields for the features required by your model
    gender = st.radio("Gender", ["Male", "Female"])
    married = st.radio("Married", ["Yes", "No"])
    dependents = st.selectbox("Dependents", [0, 1, 2, 3])
    self_employed = st.radio("Self Employed", ["Yes", "No"])
    applicant_income = st.number_input("Applicant Income", value=25000)
    coapplicant_income = st.number_input("Coapplicant Income", value=12500)
    loan_amount_term = st.number_input("Loan Amount Term", value=180)
    credit_history = st.radio("Credit History", [0, 1])
    property_area = st.selectbox("Property Area", ["Urban", "Rural", "Semiurban"])

    # Convert categorical inputs to numerical values
    gender_numeric = 1 if gender == "Male" else 0
    married_numeric = 1 if married == "Yes" else 0
    self_employed_numeric = 1 if self_employed == "Yes" else 0
    property_area_mapping = {"Urban": 2, "Rural": 0, "Semiurban": 1}
    property_area_numeric = property_area_mapping[property_area]

    # Collect input features into a numpy array
    input_features = np.array([[gender_numeric, married_numeric, dependents, self_employed_numeric,
                                applicant_income, coapplicant_income, loan_amount_term,
                                credit_history, property_area_numeric]])

    if st.button("Make Prediction"):
        # Make predictions
        try:
            prediction = make_prediction(input_features, model)
            st.write(f"Prediction: {'Yes' if prediction[0] == 1 else 'No'}")
        except Exception as e:
            st.error(f"Failed to make prediction: {e}")

if __name__ == "__main__":
    main()


# In[ ]:




