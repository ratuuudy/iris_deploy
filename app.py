import streamlit as st
import pandas as pd
import numpy as np
import joblib
import seaborn as sns
import matplotlib.pyplot as plt

# Load dataset
@st.cache_data
def load_data():
    url = "https://raw.githubusercontent.com/mwaskom/seaborn-data/master/iris.csv"
    return pd.read_csv(url)

# Load model
@st.cache_resource
def load_model():
    return joblib.load('naive_bayes_model.pkl')

df = load_data()
model = load_model()

# Sidebar for page selection
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Data Description", "Prediction", "Visualization"])

if page == "Data Description":
    st.title("Iris Dataset Description")
    st.write("""
    The Iris dataset contains 150 samples of iris flowers from three different species (setosa, versicolor, virginica).
    Each sample has four features:
    - sepal length (cm)
    - sepal width (cm)
    - petal length (cm)
    - petal width (cm)
    """)
    st.write("Here is a preview of the data:")
    st.dataframe(df.head())

    st.write("Summary statistics:")
    st.write(df.describe())

    st.write("Species distribution:")
    st.bar_chart(df['species'].value_counts())

elif page == "Prediction":
    st.title("Iris Species Prediction")
    st.write("Enter the features below:")

    sepal_length = st.slider("Sepal length (cm)", float(df['sepal_length'].min()), float(df['sepal_length'].max()), float(df['sepal_length'].mean()))
    sepal_width = st.slider("Sepal width (cm)", float(df['sepal_width'].min()), float(df['sepal_width'].max()), float(df['sepal_width'].mean()))
    petal_length = st.slider("Petal length (cm)", float(df['petal_length'].min()), float(df['petal_length'].max()), float(df['petal_length'].mean()))
    petal_width = st.slider("Petal width (cm)", float(df['petal_width'].min()), float(df['petal_width'].max()), float(df['petal_width'].mean()))

    if st.button("Predict"):
        input_features = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
        prediction = model.predict(input_features)[0]
        # Daftar nama spesies sesuai label angka
        species_names = ['setosa', 'versicolor', 'virginica']

        # KOnversi prediksi angka ke nama spesies
        predicted_species = species_names[int(prediction)]
        st.success(f"Predicted Iris Species: {predicted_species.capitalize()}")

elif page == "Visualization":
    st.title("Data Visualization")

    st.write("Scatter plot of sepal length vs sepal width, colored by species:")
    fig, ax = plt.subplots()
    sns.scatterplot(data=df, x='sepal_length', y='sepal_width', hue='species', ax=ax)
    st.pyplot(fig)

    st.write("Boxplot of petal length by species:")
    fig2, ax2 = plt.subplots()
    sns.boxplot(x='species', y='petal_length', data=df, ax=ax2)
    st.pyplot(fig2)

    st.write("Pairplot of all features:")
    st.write("Generating pairplot, please wait...")
    pairplot_fig = sns.pairplot(df, hue='species')
    st.pyplot(pairplot_fig.fig)
