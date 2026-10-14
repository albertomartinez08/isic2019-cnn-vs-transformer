# Guion de la presentación (10 min: 5–6 de exposición + 3–4 de preguntas)

Revisión final: jueves 15 de octubre de 2026, Google Meet, 10 minutos exactos. Conectarse 5 min antes.
Abrir antes: `slides/presentation.pptx`, `paper/main.pdf`, `notebooks/ISIC2019_CNN_vs_Transformer.ipynb`
y la carpeta `results/` por si el profesor pide ver código o logs.

## Diapositiva 1 – Título (15 s)
"Mi proyecto compara tres backbones preentrenados en ImageNet, ResNet-50, EfficientNet-B0 y Swin-T,
sobre ISIC 2019 con un protocolo idéntico, y mide no sólo accuracy sino costo computacional e
interpretabilidad."

## Diapositiva 2 – Problema y objetivo (40 s)
- Clasificación dermatoscópica en 8 diagnósticos (melanoma, nevus, carcinoma basocelular, etc.).
- Pregunta práctica: ¿CNN o Transformer para un prototipo clínico, y a qué precio en latencia y memoria?
- Los resultados publicados no son comparables: cada quien usa otro split, resolución o ensamble.
- Objetivo: dejar todo fijo y comparar en accuracy, balanced accuracy, macro-F1, AUC, por clase,
  parámetros, FLOPs, VRAM, tiempo de entrenamiento, convergencia, latencia, Grad-CAM y un ensamble.

## Diapositiva 3 – Dataset (40 s)
- 25,331 imágenes, 8 clases, muy desbalanceado (NV es la mitad).
- Split 17,727 / 3,844 / 3,760 estratificado por clase y **agrupado por lesión** con el metadato
  `lesion_id`: la misma lesión nunca está en dos particiones. Muchos trabajos no lo hacen y eso
  infla los resultados.
- Preprocesamiento: caché a 256 px, recortes aleatorios a 224, flips, rotación, color jitter.

## Diapositiva 4 – Modelos y protocolo (40 s)
- ResNet-50 (23.5 M), EfficientNet-B0 (4.0 M), Swin-T (27.5 M); todos timm, pesos ImageNet-1k,
  cabeza nueva de 8 clases, fine tuning completo.
- Receta idéntica: batch 32, AdamW lr 1e-4, 1 época de warm-up + coseno, máximo 12 épocas con
  early stopping por macro-F1 de validación, cross-entropy ponderada por 1/√frecuencia con label
  smoothing 0.1, AMP fp16, semilla 42, RTX 4070 Laptop de 8 GB.
- Elegí Swin-T y no ViT-B para que el costo fuera comparable al de ResNet-50.

## Diapositiva 5 – Tabla de resultados (60 s)
- Swin-T: 76.1 % accuracy, 63.9 % balanced accuracy, 64.1 % macro-F1. EfficientNet-B0 57.8 % F1,
  ResNet-50 55.1 %.
- El AUC va al revés: ResNet 0.918 > EfficientNet 0.906 > Swin 0.900. Swin ajusta más fuerte el
  train (loss 1.00 vs 1.42) y da posteriors más afilados: mejores decisiones duras, peor ranking.
- Costo: EfficientNet entrena en 14 min con 1.5 GB; Swin en 23 min con 2.7 GB y 71 ms por imagen,
  2.4 veces la latencia de ResNet (29 ms).
- Ensamble de los tres: 77.8 % accuracy, 66.5 % macro-F1, AUC 0.942.

## Diapositiva 6 – Curvas (30 s)
- Todos convergen dentro del presupuesto gracias al preentrenamiento. Swin ya está en 0.60 de F1 en
  la época 2; las CNN necesitan 5–6 épocas para llegar al 95 % de su mejor valor.
- La pérdida de validación se aplana mientras la de entrenamiento sigue bajando: sobreajuste
  incipiente controlado por early stopping.

## Diapositiva 7 – Accuracy vs costo (30 s)
- "Los FLOPs no son latencia": EfficientNet tiene 10 veces menos FLOPs que ResNet pero es más lento
  en GPU (37 vs 29 ms) porque sus convoluciones depthwise lanzan muchos kernels pequeños que no
  aprovechan los tensor cores. En CPU o móvil sí sería el más rápido.

## Diapositiva 8 – Por clase (40 s)
- Fuertes en NV, BCC y VASC; débiles en AK, SCC y DF (raras o morfológicamente parecidas).
- Dos clusters de error que también confunden a dermatólogos: MEL↔NV y AK↔BKL↔SCC.
- La ponderación de clases subió el recall de las raras (DF 0.60, VASC 0.81) sin hundir NV.
- Swin es el más uniforme: mejor F1 en 7 de 8 clases.

## Diapositiva 9 – Grad-CAM (40 s)
- Las CNN producen un blob centrado en el núcleo pigmentado; Swin produce mapas multifocales que
  siguen el borde irregular y la asimetría, las señales ABCD de los clínicos.
- Fallos informativos: en el BKL los tres miran la zona más oscura y dicen melanoma; en el SCC
  EfficientNet se va al fondo con pelo y escamas. Riesgo conocido en ISIC: segmentar la lesión o
  quitar artefactos en un sistema real.

## Diapositiva 10 – Discusión (30 s)
- Los ensambles de las dos CNN juntas sólo llegan a 61.5 % F1; CNN + Transformer a 66.4 %. Las dos
  familias se equivocan en imágenes distintas: argumento para sistemas híbridos.
- Mis balanced accuracies (0.60–0.68) están en el rango de modelos individuales a 224 px; el ganador
  del reto llegó a 0.63–0.74 con ensambles multirresolución y metadatos, en un test más difícil.

## Diapositiva 11 – Conclusiones (30 s)
- Con preentrenamiento, el Transformer es el más preciso y uniforme en 25k imágenes.
- Cuesta 2.4× la latencia y 1.5× la memoria de ResNet-50; decidir con latencia medida, no con
  parámetros.
- Recomendación: EfficientNet-B0 para CPU/edge, Swin-T con GPU, ensamble para screening offline.
- Futuro: varias semillas, 384–512 px, metadatos, TTA, calibración, preentrenamiento
  auto-supervisado en dermatoscopia, validación externa.
- Todo reproducible: `run_all.ps1` regenera cada número y figura; notebook Colab con ruta rápida
  de 440 MB y modo demo de 3 minutos.

## Preguntas probables y respuestas cortas
- **¿Por qué 12 épocas?** Es un máximo; early stopping con paciencia 4 sobre macro-F1 de validación
  decide. Reporto la mejor época y las épocas al 95 % del mejor F1 como medida de convergencia.
- **¿Por qué macro-F1 y balanced accuracy?** Con NV al 50 %, el accuracy premia al modelo que
  ignora las clases raras; macro-F1 y balanced accuracy pesan igual cada clase.
- **¿Por qué AUC contradice al F1?** AUC evalúa el ranking de probabilidades; F1 la decisión argmax.
  Un modelo más confiado gana decisiones y pierde ranking. Para triage habría que calibrar.
- **¿Por qué no entrenaste desde cero?** Lo que se hace en la práctica y lo que vimos en clase:
  fine tuning desde ImageNet mejora accuracy y tiempo; sí entrené todos los pesos, no sólo inferí.
- **¿Qué cambiarías con más tiempo?** Varias semillas para barras de error, mayor resolución,
  metadatos, TTA y validación en el test oficial de ISIC.
- **¿Fuga de datos?** No: split agrupado por `lesion_id`, verificado con una aserción en
  `prepare_data.py`.
- **¿Cómo mediste latencia?** fp16, batch 1 y 32, 10 iteraciones de calentamiento, 50 medidas,
  `torch.cuda.synchronize`.
