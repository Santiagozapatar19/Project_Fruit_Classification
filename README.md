# Sistema Automático de Clasificación de Calidad y Estimación de Tamaño de Frutas

**Curso:** Algoritmos y Programación III (Semestre 2026-2)  
**Institución:** Universidad Icesi — Facultad de Ingeniería, Diseño y Ciencias Aplicadas  
**Metodología:** CRISP-DM Adaptada a Visión por Computadora y Machine Learning  
**Estado del Proyecto:** En Desarrollo Activo (Fases 1 y 2 completadas, Fase 3 en preparación)

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
│   ├── 📂 data/                             # Funciones de preprocesamiento e inferencia sobre una imagen nueva
│   │   └── dataset_loader.py                # Segmentación Otsu/HSV y extracción de d_norm / color / textura
│   ├── 📂 models/                           # Carga de modelos entrenados para predicción en vivo
│   ├── 📂 utils/                            # Utilidades de visualización para la interfaz
│   └── main.py                              # Aplicación principal de la interfaz gráfica (Streamlit / GUI)
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

### 2. Interfaz Gráfica (`src/`)
La carpeta [`src/`](./src) está dedicada exclusivamente al código de la **interfaz gráfica de usuario (Fase 6: Despliegue)** que cargará los modelos entrenados en los cuadernos para clasificar imágenes o capturas de cámara en tiempo real.

### Ejecutar Pruebas Unitarias
```bash
python -m unittest discover -s tests
```
