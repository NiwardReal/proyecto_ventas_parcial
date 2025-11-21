import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os

# Titulo de la App
st.title("Tablero de Ventas y Predicciones")
st.markdown("---")

# 1: CARGA DE DATOS
st.sidebar.header("Cargar Datos")
archivo = st.sidebar.file_uploader("Sube tu archivo CSV (ventas.csv)", type=["csv"])

if archivo is not None:
    # Cargar el dataframe
    df = pd.read_csv(archivo, encoding='latin1')
    st.success("Datos cargados exitosamente")
    
    # Mostrar datos
    if st.checkbox("Ver primeros 5 datos"):
        st.write(df.head())

    # 2: GRAFICOS (EDA)
    st.header("1. Analisis Visual")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Ventas por Categoria")
        if 'Category' in df.columns:
            fig, ax = plt.subplots()
            sns.barplot(data=df, x='Category', y='Sales', ax=ax)
            plt.xticks(rotation=45)
            st.pyplot(fig)
        else:
            st.warning("No se encontro la columna 'Category'")

    with col2:
        st.subheader("Distribucion de Ventas")
        fig2, ax2 = plt.subplots()
        sns.histplot(df['Sales'], bins=20, ax=ax2)
        ax2.set_xlim(0, 1000) # Esto es una limitacion solo para que se vea mejor
        st.pyplot(fig2)

    # 3: PREDICCION (ML)
    st.header("2. Prediccion de Ventas Futuras")
    st.write("Usa el modelo entrenado para estimar ventas segun cantidad y descuento")

    # Cargar modelo
    ruta_modelo = os.path.join(os.path.dirname(__file__), '../models/modelo_ventas.pkl')
    
    if os.path.exists(ruta_modelo):
        modelo = joblib.load(ruta_modelo)
        
        # Inputs del usuario
        cantidad = st.number_input("Cantidad de productos", min_value=1, value=1)
        descuento = st.slider("Descuento aplicado", 0.0, 0.8, 0.0)
        
        if st.button("Calcular Prediccion"):
            prediccion = modelo.predict([[cantidad, descuento]])[0]
            st.success(f"Venta estimada: ${prediccion:.2f}")
    else:
        st.error("No se encontro el modelo; ejecute primero 'entrenar_modelo.py'")

else:
    st.info("suba el archivo CSV en el menu lateral para empezar")