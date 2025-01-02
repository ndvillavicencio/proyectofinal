import os
import io
import pandas as pd
import numpy as np
import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from sklearn.impute import KNNImputer
import streamlit.components.v1 as components
from sklearn.linear_model import LinearRegression

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
    
    


    # Gráficos bivariantes
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
    



    # Widget para aplicar la función dropna
    st.subheader("Limpieza de Datos")
    if st.button("Aplicar función dropna"):
        df_cleaned = df.dropna()
        st.write("Datos después de aplicar dropna:")
        st.write(df_cleaned.head())

        # Widget para descargar el nuevo archivo con los datos limpiados
        st.subheader("Descargar Archivo con Datos Limpiados")
        csv_cleaned = df_cleaned.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="Descargar CSV Limpiado",
            data=csv_cleaned,
            file_name='datos_limpiados.csv',
            mime='text/csv',
        )



    # Widget para aplicar la función fillna con método ffill
    st.subheader("Llenado de Datos Faltantes")
    if st.button("Aplicar función fillna (ffill)"):
        df_filled = df.fillna(method='ffill')
        st.write("Datos después de aplicar fillna (ffill):")
        st.write(df_filled.head())

        # Widget para descargar el nuevo archivo con los datos llenados
        st.subheader("Descargar Archivo con Datos Llenados")
        csv_filled = df_filled.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="Descargar CSV Llenado",
            data=csv_filled,
            file_name='datos_llenos.csv',
            mime='text/csv',
        )



    st.subheader("Imputación de Valores Faltantes")
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






# Sección 3: Módulo de Regresiones
st.header("3. Módulo de Regresiones")

if uploaded_file is not None:
    # Filtrar solo las columnas con datos numéricos
    numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()

    # Widget para seleccionar una variable numérica para regresión lineal simple
    st.subheader("Regresión Lineal Simple")
    selected_var = st.selectbox("Selecciona una variable numérica para aplicar regresión lineal", numeric_columns)

    if st.button("Aplicar Regresión Lineal"):
        X = df[[selected_var]].dropna().values.reshape(-1, 1)
        y = df[selected_var].dropna().values

        # Aplicar regresión lineal simple
        model = LinearRegression()
        model.fit(X, y)
        y_pred = model.predict(X)

        # Crear un DataFrame con los datos originales y los datos predichos
        results_df = pd.DataFrame({selected_var: y, 'Predicción': y_pred})

        # Generar gráfico de dispersión con datos originales y predichos
        fig = px.scatter(results_df, x=selected_var, y='Predicción', title=f'Regresión Lineal Simple para {selected_var}')
        fig.add_scatter(x=results_df[selected_var], y=results_df[selected_var], mode='lines', name='Original')
        st.plotly_chart(fig)



# Sección 4: Generación de Informes


def generar_estadisticas(df):
    """
    Genera Estadísticas descriptivas de un dataframe
    """
    return df.describe()

def exportar_excel(df):
    """
    Exporta un dataframe a un archivo excel
    """
    output = io.BytesIO()
    writer = pd.ExcelWriter(output, engine='xlsxwriter')
    df.to_excel(writer, index=False, sheet_name="Estadísticas")
    writer.close()
    processed_data = output.getvalue()
    return processed_data



st.header("4. Generación de Informes")

if uploaded_file is not None:
    if st.button("Ejecutar Análisis Descriptivo al Dataset"):
        # Aplicar la función describe
        stats = df.describe()
        st.dataframe(df.describe())

        # Convertir a XLSX
        output = io.BytesIO()
        writer = pd.ExcelWriter(output, engine='xlsxwriter')
        stats.to_excel(writer, sheet_name='Estadísticas')
        writer.close()
        processed_data = output.getvalue()

        # Descargar el archivo
        st.download_button(
            label="Descargar XLSX con estadísticas",
            data=processed_data,
            file_name='estadisticas_dataset.xlsx',
            mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
        )
