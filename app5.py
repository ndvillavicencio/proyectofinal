import os
import pandas as pd
import numpy as np
import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from sklearn.impute import KNNImputer
import streamlit.components.v1 as components

# Título de la aplicación
html_code = """
<div style='font-size:40px; color:#000000; text-align:center; background-color:#FFA500; padding:10px; border-radius:10px;'>
    Herramienta de Análisis de Datos Interactiva en Streamlit
</div>
"""
components.html(html_code)



# Sección 1: Carga DATASET
st.header("1. Carga de Dataset")

# Widget para cargar archivos
uploaded_file = st.file_uploader("Elige un archivo", type=["csv", "xlsx", "xls"])

if uploaded_file is not None:
    try:
        if uploaded_file.name.endswith('.csv'):
            df = pd.read_csv(uploaded_file)
        else:
            df = pd.read_excel(uploaded_file)
        
        st.write("Archivo cargado exitosamente!")
        st.write(df.head())
    except Exception as e:
        st.error(f"Error al cargar el archivo: {e}")
else:
    st.info("Por favor, carga un archivo en formato .CSV, .XLSX o .XLS.")


# Sección 2: Análisis Exploratorio de Datos (EDA)
st.header("2. Módulo de EDA")

if uploaded_file is not None:
    # Resumen Estadístico
    st.subheader("Resumen Estadístico")
    st.write(df.describe())

    # Gráficos univariantes
    # Widget para seleccionar una variable y generar gráficos
    st.subheader("Generar Gráficos")
    selected_var = st.selectbox("Selecciona una variable para graficar", df.columns)
    chart_type = st.selectbox("Selecciona el tipo de gráfico", ["Histograma", "Gráfico de Cajas y Bigotes", "Gráfico de Violín", "Gráfico de Barras"])

    if st.button("Generar Gráfico"):
        if chart_type == "Histograma":
            fig = px.histogram(df, x=selected_var)
        elif chart_type == "Gráfico de Cajas y Bigotes":
            fig = px.box(df, y=selected_var)
        elif chart_type == "Gráfico de Violín":
            fig = px.violin(df, y=selected_var)
        elif chart_type == "Gráfico de Barras":
            fig = px.bar(df, x=selected_var)
        
        st.plotly_chart(fig)
    
    



    # Filtrar solo las columnas numéricas
    numeric_df = df.select_dtypes(include=[np.number])

    # Graficar la matriz de correlación de los datos numéricos
    st.subheader("Matriz de Correlación")
    corr_matrix = numeric_df.corr()
    fig = px.imshow(corr_matrix, text_auto=True, aspect="auto", color_continuous_scale='RdBu_r')
    st.plotly_chart(fig)


    # Widget para seleccionar dos variables y generar gráfico de dispersión
    st.subheader("Gráfico de Dispersión")
    x_var = st.selectbox("Selecciona la variable para el eje X", df.columns)
    y_var = st.selectbox("Selecciona la variable para el eje Y", df.columns)

    if st.button("Generar Gráfico de Dispersión"):
        fig = px.scatter(df, x=x_var, y=y_var)
        st.plotly_chart(fig)









    # Manejo de Datos Faltantes
    st.subheader("Manejo de Datos Faltantes")
    
    st.write("Visualización de Datos Faltantes")
    fig, ax = plt.subplots()
    sns.heatmap(df.isnull(), cbar=False, cmap='viridis')
    st.pyplot(fig)
    
    st.write("Imputación de Valores Faltantes")
    numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
    imputation_method = st.selectbox("Selecciona el método de imputación", ["Media", "Mediana", "KNN"])
    
    if imputation_method == "Media":
        df[numeric_columns] = df[numeric_columns].fillna(df[numeric_columns].mean())
    elif imputation_method == "Mediana":
        df[numeric_columns] = df[numeric_columns].fillna(df[numeric_columns].median())
    elif imputation_method == "KNN":
        imputer = KNNImputer(n_neighbors=5)
        df[numeric_columns] = pd.DataFrame(imputer.fit_transform(df[numeric_columns]), columns=numeric_columns)
    
    st.write("Datos después de la imputación:")
    st.write(df.head())

    # Widget para descargar el nuevo archivo con los datos imputados
    st.subheader("Descargar Archivo con Datos Imputados")
    csv = df.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="Descargar CSV",
        data=csv,
        file_name='datos_imputados.csv',
        mime='text/csv',
    )


