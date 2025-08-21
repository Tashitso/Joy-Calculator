import streamlit as st
import pandas as pd
from sklearn.linear_model import LinearRegression

st.title("Happiness Predictor App")

# Load & Train (from your data)
df = pd.read_csv('2023.csv')
df = df[['Ladder score', 'Logged GDP per capita', 'Social support', 'Healthy life expectancy', 'Freedom to make life choices', 'Generosity', 'Perceptions of corruption']]
df.rename(columns={'Ladder score': 'Happiness', 'Logged GDP per capita': 'GDP', 'Healthy life expectancy': 'Health', 'Freedom to make life choices': 'Freedom', 'Perceptions of corruption': 'Corruption'}, inplace=True)
df.fillna(df.median(numeric_only=True), inplace=True)
X = df[['GDP', 'Social support', 'Health', 'Freedom', 'Generosity', 'Corruption']]
y = df['Happiness']
model = LinearRegression()
model.fit(X, y)

# Inputs
gdp = st.number_input("GDP (e.g., 10.5)", value=10.5)
support = st.number_input("Social Support (e.g., 1.5)", value=1.5)
health = st.number_input("Health (e.g., 70)", value=70.0)
freedom = st.number_input("Freedom (e.g., 0.8)", value=0.8)
generosity = st.number_input("Generosity (e.g., 0.2)", value=0.2)
corruption = st.number_input("Corruption (e.g., 0.3)", value=0.3)

if st.button("Predict Happiness"):
    new_df = pd.DataFrame([[gdp, support, health, freedom, generosity, corruption]], columns=['GDP', 'Social support', 'Health', 'Freedom', 'Generosity', 'Corruption'])
    pred = model.predict(new_df)[0]
    st.success(f"Predicted Happiness Score: {pred:.2f} (out of 10)")

st.info("Based on World Happiness data—adjust factors for your 'joy' calculation!")
