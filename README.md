# Sistema Automático de Clasificación de Calidad y Estimación de Tamaño de Frutas

**Curso:** Algoritmos y Programación III (Semestre 2026-2)  
**Institución:** Universidad Icesi — Facultad de Ingeniería, Diseño y Ciencias Aplicadas  
**Metodología:** CRISP-DM Adaptada a Visión por Computadora y Machine Learning  
**Estado del Proyecto:** Fases 1 a 6 de CRISP-DM implementadas (EDA, preparación, modelado, evaluación y despliegue en Streamlit)

---

## 1. Descripción del Problema y Contexto (Fase 1: Comprensión del Negocio)

En mercados minoristas, plazas de mercado y líneas agroindustriales, la inspección visual manual de frutas frescas según su grado de madurez, defectos superficiales (golpes, manchas fúngicas, podredumbre) y calibre (tamaño) es un proceso lento, subjetivo y propenso a fatiga humana. Esto ocasiona pérdidas económicas, desperdicio alimentario y falta de estandarización comercial.

Este proyecto desarrolla un sistema automático basado en **Visión por Computadora, Machine Learning Tradicional y Deep Learning** capaz de:
1. **Clasificar la calidad del producto** en **3 categorías** (`Buena / Good`, `Regular-Intermedia / Mixed`, `Mala / Bad`) e identificar el tipo de fruta (`Apple`, `Banana`, `Guava`, `Lime/Lemon`, `Orange`, `Pomegranate`).
2. **Estimar el tamaño relativo del ejemplar** en **3 categorías** (`Pequeño / Small`, `Mediano / Medium`, `Grande / Large`) a partir del **diámetro equivalente normalizado en píxeles** ($d_{\text{norm}}$) obtenido mediante segmentación morfológica del contorno de la fruta.
3. **Desplegar las predicciones en una interfaz gráfica interactiva** para imágenes estáticas o captura por cámara en tiempo real.

---

## 2. Estructura del Repositorio (Estándar del Curso)

El repositorio sigue la arquitectura modular definida en [`estructura_proyecto.md`](./estructura_proyecto.md):

```text
📦 Project_Fruit_Classification/
│
├── 📂 docs/                                 # Documentación técnica y metodológica
│   ├── 📜 arquitectura.md                   # Diseño del pipeline CRISP-DM y formulación matemática
│   ├── 📜 instalacion.md                    # Guía de instalación y ejecución reproducible
│   └── 📜 protocolo_recoleccion_fotos.md    # Estándar para las 30-50 fotos propias del grupo
│
├── 📂 data/
│   └── 📂 custom_group_photos/              # Imágenes recolectadas por el grupo + plantilla de anotación
│       └── 📜 annotations_template.csv
│
├── 📂 src/                                  # Código fuente de la Interfaz Gráfica y Despliegue (Fase 6 CRISP-DM)
│   ├── 📂 data/
│   │   └── dataset_loader.py                # Segmentación por saliencia + Otsu y extractor de las 31 características
│   ├── 📂 models/
│   │   └── predictor.py                     # Carga de checkpoints (MLP / Random Forest), calidad y tamaño
│   ├── 📂 utils/
│   │   └── visualization.py                 # Superposición de máscara y contorno para la interfaz
│   └── main.py                              # Aplicación Streamlit (subir imagen o cámara en vivo)
│
├── 📂 notebooks/                            # Cuadernos Jupyter autocontenidos (Fases 1 a 5 CRISP-DM, estilo de clase)
│   └── 📓 01_EDA_Comprension_Datos.ipynb    # Cuaderno completo de EDA (código, tablas, ecuaciones y gráficas)
│
├── 📂 experiments/                          # Artefactos generados por los experimentos
│   ├── 📂 logs/                             # Registros de ejecución
│   ├── 📂 checkpoints/                      # Modelos serializados
│   └── 📂 results/                          # Tablas CSV y figuras vectoriales (PDF/PNG) para el informe IEEE
│       └── 📂 eda_figures/
│
├── 📂 tests/                                # Pruebas unitarias de carga y extracción geométrica
│   └── test_data_pipeline.py
│
├── 📜 requirements.txt                      # Dependencias de Python
├── 📜 .gitignore                            # Exclusión de archivos pesados y temporales
├── 📜 LICENSE                               # Licencia MIT
└── 📜 README.md                             # Descripción general del proyecto
```

---

## 3. Conjuntos de Datos

El sistema integra dos fuentes de datos complementarias:
1. **Base de Datos de Referencia (Kaggle — *Fruit Quality Classification*):** Ubicada en el directorio raíz del espacio de trabajo (`../Good Quality_Fruits`, `../Bad Quality_Fruits`, `../Mixed Qualit_Fruits`), con un total de **19,526 imágenes** distribuidas en 6 frutas (`Apple`, `Banana`, `Guava`, `Lime/Lemon`, `Orange`, `Pomegranate`).
2. **Recolección Propia del Grupo (30–50 imágenes):** Fotografías capturadas bajo el protocolo estandarizado en [`docs/protocolo_recoleccion_fotos.md`](./docs/protocolo_recoleccion_fotos.md) y etiquetadas manualmente en calidad (`Good`, `Medium`, `Bad`) y tamaño (`Small`, `Medium`, `Large`) en [`data/custom_group_photos/annotations_template.csv`](./data/custom_group_photos/annotations_template.csv).

---

## 4. Ejecución Rápida

### 1. Exploración y Entrenamiento en Cuadernos Jupyter (`notebooks/`)
Tal como se trabaja en el curso, todo el desarrollo analítico de **CRISP-DM (Fases 1 a 5: EDA, Preparación de Datos, Entrenamiento de Modelos y Evaluación)** está contenido directamente dentro de los cuadernos de la carpeta [`notebooks/`](./notebooks):
* Abrir y ejecutar [`notebooks/01_EDA_Comprension_Datos.ipynb`](./notebooks/01_EDA_Comprension_Datos.ipynb) en VS Code o Jupyter Lab. El cuaderno es 100% autocontenido: escanea las **19,526 imágenes**, calcula el desbalanceo, realiza la segmentación matemática de tamaño ($d_{\text{norm}}$) y grafica todas las distribuciones directamente en sus celdas (además de guardar copias en `.pdf` vectorial y `.png` en `experiments/results/eda_figures/` para el informe IEEE).

### 2. Interfaz Gráfica (`src/`) — Fase 6: Despliegue
Desde la raíz del repositorio:
```bash
pip install -r requirements.txt
streamlit run src/main.py
```
Se abre en `http://localhost:8501`. La app permite:
* **Entradas:** subir una imagen (`.jpg`, `.png`, `.heic`) o tomar una foto con la cámara (`st.camera_input`).
* **Calidad:** clase `Good / Buena`, `Mixed / Regular` o `Bad / Mala` con sus probabilidades (`predict_proba`). Modelo por defecto: **MLP** (`fase4_mlp.joblib`, ganador de la Fase 5); selector opcional para comparar con **Random Forest**.
* **Tamaño:** categoría `Small / Medium / Large` y valor de $d_{\text{norm}}$, con selector de fruta y de umbrales (recalibrados con fotos propias en la Fase 5 o los de Kaggle).
* **Visualización:** imagen original junto a la máscara y el contorno segmentado, y tabla con las 31 características.

Capturas de la interfaz en [`docs/capturas/`](./docs/capturas).

### Ejecutar Pruebas Unitarias
```bash
python -m unittest discover -s tests
```
