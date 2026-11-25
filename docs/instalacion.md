# Guía de Instalación y Ejecución

## 1. Requisitos del Sistema
* Python 3.11 o superior (probado en Python 3.14).
* Carpetas del dataset ubicadas en el directorio padre (`../Good Quality_Fruits`, `../Bad Quality_Fruits`, `../Mixed Qualit_Fruits`).

## 2. Instalación de Dependencias
Desde la carpeta `Project_Fruit_Classification`:
```bash
pip install -r requirements.txt
```

## 3. Flujo de Trabajo
* **Fase 1 a 5 de CRISP-DM (Cuadernos en `notebooks/`):**
  Abre [`notebooks/01_EDA_Comprension_Datos.ipynb`](../notebooks/01_EDA_Comprension_Datos.ipynb) en VS Code o Jupyter para revisar y ejecutar todas las celdas de exploración, formulación matemática y gráficas.
* **Fase 6 de CRISP-DM (Interfaz gráfica en `src/`):**
  Desde la raíz del repositorio (`Project_Fruit_Classification/`):
  ```bash
  streamlit run src/main.py
  ```
  La app abre en `http://localhost:8501`. No requiere el dataset de Kaggle: solo usa los modelos de `experiments/checkpoints/` y los umbrales de `experiments/results/`.
  * **Subir imagen:** pestaña *Subir imagen* (`.jpg`, `.png`, `.heic`).
  * **Cámara en vivo:** pestaña *Cámara en vivo*; el navegador pedirá permiso para usar la cámara (en macOS también puede pedirlo *Ajustes del Sistema → Privacidad y seguridad → Cámara*).
  * En la barra lateral se elige el modelo (MLP por defecto o Random Forest), el tipo de fruta y la fuente de umbrales de tamaño.
  * Para mejores resultados: una sola fruta, fondo simple y uniforme, buena luz y sin reflejos fuertes.

  > Los checkpoints se guardaron con `scikit-learn==1.7.2`. Con otra versión pueden no cargar o dar advertencias; instalar exactamente esa versión.
* **Pruebas unitarias:**
  ```bash
  python -m unittest discover -s tests
  ```
