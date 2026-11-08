# Protocolo de Recolección y Anotación Manual de Fotografías Propias (Sección 2.2 y 2.3)

Este documento define el estándar técnico y ético para la toma de las **30 a 50 fotografías propias** exigidas en el proyecto final de Algoritmos y Programación III.

---

## 1. Distribución Sugerida para el Grupo (3 Integrantes — 45 Fotos en Total)

Cada integrante selecciona **1 tipo de fruta** (presente en el dataset de referencia) y captura **15 fotografías** cubriendo las **3 categorías de calidad** y los **3 tamaños relativos**:

| Integrante | Fruta Asignada | Calidad Buena (`Good`) | Calidad Regular (`Medium`) | Calidad Mala (`Bad`) | Total |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **Integrante 1** | Banano (`Banana`) | 5 fotos *(Peq, Med, Gra)* | 5 fotos *(Pintón/Manchas leves)* | 5 fotos *(Negro/Sobremaduro)* | **15** |
| **Integrante 2** | Manzana (`Apple`) o Guayaba (`Guava`) | 5 fotos *(Peq, Med, Gra)* | 5 fotos *(Golpe leve/Pecas)* | 5 fotos *(Podredumbre/Hongo)* | **15** |
| **Integrante 3** | Limón (`Lime`) o Naranja (`Orange`) | 5 fotos *(Peq, Med, Gra)* | 5 fotos *(Amarillento/Piel seca)* | 5 fotos *(Moho/Descarte)* | **15** |
| **TOTAL GRUPO** | **3 Frutas** | **15 fotos** | **15 fotos** | **15 fotos** | **45** |

---

## 2. Reglas de Captura Fotográfica (Estándar de Visión por Computadora)

Para garantizar que el algoritmo de segmentación calcule sin sesgos el **diámetro equivalente normalizado** ($d_{\text{norm}}$) y los descriptores de color/textura:

1. **Un solo ejemplar por imagen:** Colocar únicamente una fruta centrada en el encuadre.
2. **Fondo uniforme y de alto contraste:** Usar una cartulina u hoja de papel blanco/gris mate o una superficie lisa sin texturas. **No sostener la fruta con la mano**.
3. **Distancia fija cámara-objeto ($25\text{ a }30\text{ cm}$) y zoom $1\times$:** Mantener la cámara a una altura constante (~28 cm, equivalente al largo de una hoja tamaño carta) en ángulo cenital ($90^\circ$) o ligeramente inclinado ($45^\circ$). Esto asegura que el área en píxeles sea proporcional al tamaño físico real de la fruta.
4. **Iluminación difusa sin flash:** Tomar las fotografías con luz natural indirecta o iluminación interior uniforme, **desactivando el flash** para evitar reflejos especulares blancos y sombras duras adheridas al contorno.
5. **Cumplimiento Ético y Privacidad (Indicador PI1):** No incluir rostros, manos, marcas comerciales visibles ni información personal en el encuadre o en los metadatos EXIF.

---

## 3. Criterios de Etiquetado Manual (`annotations_template.csv`)

Cada imagen debe guardarse en `data/custom_group_photos/` y registrarse en `data/custom_group_photos/annotations_template.csv` con las siguientes columnas:

* `filename`: Nombre del archivo (ej. `banana_good_small_01.jpg`).
* `fruit_type`: `Apple`, `Banana`, `Guava`, `Lime`, `Orange` o `Pomegranate`.
* `quality_label`:
  * `Good`: Fruta fresca, color homogéneo, firme, sin defectos mecánicos ni biológicos.
  * `Medium`: Madurez avanzada o defectos cosméticos leves (pequeñas manchas o magulladuras superficiales $<15\%$ del área), apta para consumo inmediato.
  * `Bad`: Deterioro severo, podredumbre, presencia de micelio/moho o heridas abiertas (descarte).
* `size_label`: `Small` (Pequeño), `Medium` (Mediano) o `Large` (Grande).
* `capture_distance_cm`: Distancia aproximada de captura (ej. `28`).
* `annotator`: Iniciales o nombre del integrante.
