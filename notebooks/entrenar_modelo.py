import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import joblib
import os

# 1. Carga de Datos
# Usamos rutas relativas para que funcione en VS Code
ruta_archivo = os.path.join(os.path.dirname(__file__), '../data/ventas.csv')

print(f"Buscando archivo en: {ruta_archivo}")

try:
    df = pd.read_csv(ruta_archivo, encoding='latin1') # encoding latin1 es comun en este dataset
    print("Datos cargados correctamente")
except FileNotFoundError:
    print("No se encuentra ventas.csv en la carpeta data/.")
    exit()

# 2. Limpieza Rapida
# Vamos a predecir 'Sales' (Ventas) usando 'Discount' (Descuento) y 'Quantity' (Cantidad)
# Eliminamos filas que no tengan estos datos
df = df[['Sales', 'Quantity', 'Discount']].dropna()

# 3. Entrenamiento
X = df[['Quantity', 'Discount']]
y = df['Sales']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

modelo = LinearRegression()
modelo.fit(X_train, y_train)

# 4. Evaluacion
score = r2_score(y_test, modelo.predict(X_test))
print(f"📊 Precisión del modelo (R2): {score:.2f}")

# 5. Guardar el modelo
ruta_modelo = os.path.join(os.path.dirname(__file__), '../models/modelo_ventas.pkl')
joblib.dump(modelo, ruta_modelo)
print(f"Modelo guardado en: {ruta_modelo}")