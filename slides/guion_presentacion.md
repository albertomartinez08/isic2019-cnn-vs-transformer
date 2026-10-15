# Guion de la presentación (10 min: 5–6 de exposición + 3–4 de preguntas)

Revisión final: jueves 15 de octubre de 2026, Google Meet (https://meet.google.com/dfd-xwhq-uev), 10 minutos exactos.
Conectarse 5 minutos antes. Tener abiertos: la presentación, el PDF del artículo, el repositorio en el
navegador (https://github.com/albertomartinez08/isic2019-cnn-vs-transformer) y la carpeta `results/`.

Tiempo total hablado: ~5 min 30 s. Las frases entre comillas son lo que se dice; lo demás es apoyo.

---

## 1. Portada (15 s)
"Buenas tardes. Mi proyecto compara tres arquitecturas preentrenadas en ImageNet, ResNet-50,
EfficientNet-B0 y Swin-T, sobre el dataset de dermatoscopia ISIC 2019, con un protocolo idéntico,
y mide no sólo accuracy sino costo computacional e interpretabilidad."

## 2. Problema y objetivo (35 s)
"La tarea es clasificar una imagen dermatoscópica en 8 diagnósticos: melanoma, nevus, carcinoma
basocelular, queratosis actínica, queratosis benigna, dermatofibroma, lesión vascular y carcinoma
espinocelular."
"La pregunta práctica es: ¿conviene una CNN o un Transformer para un prototipo clínico, y a qué
precio en latencia y memoria? En la literatura cada trabajo usa otro split, otra resolución u otro
ensamble, así que no son comparables y casi nadie reporta el costo."
"Mi objetivo fue dejar todo fijo y comparar en accuracy, balanced accuracy, macro-F1, AUC,
parámetros, FLOPs, VRAM, tiempo, convergencia, latencia, Grad-CAM y un ensamble."

## 3. Dataset (35 s)
"ISIC 2019 tiene 25,331 imágenes y está muy desbalanceado: la clase nevus es la mitad."
"Hice un split 70/15/15 estratificado por clase y agrupado por lesión con el metadato lesion_id:
la misma lesión nunca cae en dos particiones. Esto importa porque muchos trabajos no lo hacen y
eso infla los resultados; lo verifico con una aserción en el código."
"Preprocesamiento: caché a 256 píxeles, recortes aleatorios a 224, volteos, rotación y color
jitter en entrenamiento; recorte central en evaluación."

## 4. Modelos y protocolo (35 s)
"ResNet-50 como CNN de referencia, EfficientNet-B0 como CNN eficiente de 4 millones de parámetros,
y Swin-T como Transformer jerárquico. Elegí Swin-T y no ViT-B para que el costo fuera comparable
al de ResNet-50 y la comparación fuera justa."
"La receta es idéntica para los tres: pesos de ImageNet, cabeza nueva de 8 clases, fine tuning
completo, AdamW con learning rate 1e-4, un epoch de warm-up más coseno, máximo 12 épocas con
early stopping sobre macro-F1 de validación, cross-entropy ponderada por el inverso de la raíz de
la frecuencia con label smoothing, precisión mixta, semilla 42, una RTX 4070 de 8 GB."

## 5. Tabla de resultados (60 s)  ← la diapositiva más importante
"En el test, Swin-T es el mejor modelo individual: 76.1 % de accuracy, 63.9 % de balanced
accuracy y 64.1 % de macro-F1. Le siguen EfficientNet-B0 con 57.8 y ResNet-50 con 55.1."
"Dos cosas que vale la pena notar. Primera: el AUC va al revés, ResNet 0.918, EfficientNet 0.906,
Swin 0.900. El AUC mide el ranking de probabilidades y el F1 mide la decisión. Swin ajusta más
fuerte el entrenamiento y da probabilidades más afiladas: mejores decisiones, peor ranking. Para
una herramienta de triage habría que calibrar."
"Segunda: el costo. EfficientNet entrena en 14 minutos con 1.5 GB; Swin en 23 minutos con 2.7 GB y
tarda 71 milisegundos por imagen, 2.4 veces lo que tarda ResNet."
"El ensamble de los tres sube a 77.8 % de accuracy, 66.5 % de macro-F1 y AUC 0.942."

## 6. Curvas de convergencia (25 s)
"Los tres convergen dentro de las 12 épocas gracias al preentrenamiento. Swin ya está en 0.60 de F1
en la época 2; las CNN necesitan 5 o 6 épocas para llegar al 95 % de su mejor valor. La pérdida de
entrenamiento sigue bajando mientras la de validación se aplana: sobreajuste incipiente que el early
stopping mantiene a raya."

## 7. Accuracy contra costo (30 s)
"Esta gráfica es la que le mostraría a quien decide la infraestructura. Los FLOPs no son latencia:
EfficientNet tiene 10 veces menos FLOPs que ResNet y aun así es más lento en GPU, 37 contra 29
milisegundos, porque sus convoluciones depthwise lanzan muchos kernels pequeños que no aprovechan
los tensor cores. En CPU o en un móvil sí sería el más rápido. Conclusión: decidir con latencia
medida en el hardware objetivo, no con el número de parámetros."

## 8. Comportamiento por clase (35 s)
"Los tres modelos son fuertes en nevus, carcinoma basocelular y lesiones vasculares, y débiles en
queratosis actínica, carcinoma espinocelular y dermatofibroma, que son raras y se parecen entre
sí. En la matriz de confusión se ven dos clusters de error que también confunden a los
dermatólogos: melanoma contra nevus, y el grupo de queratosis."
"La ponderación de clases subió el recall de las clases raras sin hundir a nevus. Swin es el más
uniforme: mejor F1 en 7 de 8 clases y el único que pasa de 0.6 en dermatofibroma."

## 9. Grad-CAM (35 s)
"Para interpretabilidad usé Grad-CAM con la clase verdadera como objetivo. Las CNN producen una
mancha compacta centrada en el núcleo pigmentado. Swin produce mapas multifocales que siguen el
borde irregular y la asimetría del pigmento, que son justamente las señales ABCD que usan los
clínicos."
"Los fallos también enseñan: en la queratosis benigna los tres miran la zona más oscura y dicen
melanoma; en el carcinoma espinocelular EfficientNet se va al fondo, donde hay pelo y escamas. En
un sistema real habría que segmentar la lesión o quitar artefactos."

## 10. ¿Son complementarias las familias? (30 s)
"Probé todos los ensambles por votación suave. Las dos CNN juntas sólo llegan a 61.5 % de F1;
EfficientNet más Swin llega a 66.4 %, casi lo mismo que los tres juntos. Es decir, casi toda la
ganancia viene de combinar una CNN con el Transformer: se equivocan en imágenes distintas."
"El precio es 137 milisegundos por imagen: bien para screening offline, no para una herramienta
interactiva."
"Como contexto, mis balanced accuracies de 0.60 a 0.68 están en el rango de modelos individuales a
224 píxeles; el ganador del reto llegó a 0.63–0.74 con ensambles multirresolución y metadatos. Y
como limitación: una sola semilla por modelo, así que diferencias menores a un punto son ruido."

## 11. Conclusiones y recomendación (30 s)
"Tres conclusiones. Uno: con preentrenamiento, el Transformer es el más preciso y el más uniforme
por clase, aun con 25 mil imágenes. Dos: esa precisión cuesta 2.4 veces la latencia y 1.5 veces la
memoria de ResNet, y el modelo más barato en FLOPs no es el más rápido en GPU. Tres: las dos
familias son complementarias y el ensamble da el mejor F1 y el mejor AUC."
"La recomendación: EfficientNet-B0 para CPU o dispositivos edge, Swin-T cuando hay GPU, y el
ensamble para screening offline. Como trabajo futuro: varias semillas, mayor resolución, metadatos,
calibración y validación externa."

## 12. Cierre (10 s)
"Todo es reproducible: el repositorio tiene el código, los logs, las figuras y el notebook de
Colab con una ruta rápida de 442 MB y un modo demo de 3 minutos. Ahí está el QR. Gracias, quedo
atento a sus preguntas."

---

## Preguntas probables y respuestas cortas

- **¿Cuántas épocas entrenaste y por qué?** Máximo 12 con early stopping de paciencia 4 sobre el
  macro-F1 de validación. Reporto la mejor época (12, 8 y 8) y las épocas para llegar al 95 % del
  mejor F1 (6, 5 y 6) como medida de facilidad de convergencia.
- **¿Por qué macro-F1 y balanced accuracy además de accuracy?** Con nevus al 50 %, el accuracy
  premia ignorar las clases raras; macro-F1 y balanced accuracy pesan igual cada clase. Balanced
  accuracy es además la métrica oficial del reto ISIC 2019.
- **¿Por qué el AUC contradice al F1?** AUC evalúa el ranking de probabilidades, F1 la decisión
  argmax. Un modelo más confiado gana decisiones y pierde ranking. Para triage habría que calibrar
  (temperature scaling) y elegir umbrales.
- **¿Entrenaste desde cero?** No, fine tuning completo desde ImageNet, que es lo que se hace en la
  práctica y mejora accuracy y tiempo. Sí entrené todos los pesos de los tres modelos con el
  mismo protocolo; no sólo cargué pesos para inferir.
- **¿Cómo evitaste fuga de datos?** Split agrupado por lesion_id y verificado con una aserción en
  prepare_data.py: cero lesiones compartidas entre particiones.
- **¿Cómo mediste latencia?** fp16, batch 1 y 32, 10 iteraciones de calentamiento, 50 medidas,
  con torch.cuda.synchronize antes y después.
- **¿Por qué Swin-T y no ViT?** Para igualar el costo de ResNet-50 (9 contra 8 GFLOPs, 27 contra
  23 millones de parámetros). ViT-B tiene 86 millones y habría confundido arquitectura con tamaño.
- **¿Por qué pesos 1/√frecuencia y no 1/frecuencia?** El inverso completo sobrecorrige y hunde la
  clase mayoritaria; la raíz es un balanceo suavizado que subió el recall de las raras sin
  perder nevus.
- **¿Qué cambiarías con más tiempo?** Varias semillas para barras de error, 384–512 px, metadatos
  de edad, sexo y sitio, test-time augmentation y validación en el test oficial de ISIC.
- **¿El profesor puede reproducirlo sin bajar 9 GB?** Sí: celda 2a del notebook baja 442 MB desde
  Drive y `--subset 0.1` entrena una prueba en 3 minutos; `summarize.py` regenera las tablas sin
  GPU a partir de los logs guardados.
