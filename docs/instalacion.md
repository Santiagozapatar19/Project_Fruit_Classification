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
  El código en `src/` contiene la lógica de despliegue e inferencia en tiempo real para la interfaz gráfica (Streamlit).
* **Pruebas unitarias:**
  ```bash
  python -m unittest discover -s tests
  ```
