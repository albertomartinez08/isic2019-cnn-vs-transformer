# Guion de la presentación (10 min: 5–6 de exposición + 3–4 de preguntas)

Revisión final: jueves 15 de octubre de 2026, Google Meet (https://meet.google.com/dfd-xwhq-uev), 10 minutos exactos.
Conectarse 5 minutos antes. Tener abiertos: la presentación, el PDF del artículo, el repositorio en el
navegador (https://github.com/albertomartinez08/isic2019-cnn-vs-transformer) y la carpeta `results/`.

Tiempo total hablado: 5 min 00 s (presupuesto por diapositiva entre paréntesis; cronometrar la 5).
Las frases entre comillas son lo que se dice. Si vas tarde, salta la 6 y acorta la 10.

---

## 1. Portada (10 s)
"Buenas tardes. Comparo tres arquitecturas preentrenadas, ResNet-50, EfficientNet-B0 y Swin-T,
sobre ISIC 2019 con un protocolo idéntico, midiendo accuracy, costo computacional e
interpretabilidad."

## 2. Problema y objetivo (25 s)
"La tarea: clasificar una imagen dermatoscópica en 8 diagnósticos, entre ellos melanoma."
"La pregunta práctica: ¿CNN o Transformer para un prototipo clínico, y a qué precio en latencia y
memoria? En la literatura no se puede comparar porque cada trabajo usa otro split y otra
resolución, y casi nadie reporta el costo. Mi objetivo fue dejar todo fijo y medir ambas cosas."

## 3. Dataset (25 s)
"25,331 imágenes, muy desbalanceadas: nevus es la mitad. Split 70/15/15 estratificado y agrupado
por lesión, así la misma lesión nunca cae en dos particiones; muchos trabajos no lo hacen y eso
infla resultados. Caché a 256 píxeles, recortes a 224, volteos, rotación y color jitter."

## 4. Modelos y protocolo (30 s)
"Dos CNN, una clásica y una eficiente, y un Transformer jerárquico. Elegí Swin-T y no ViT-B para
que el costo fuera comparable al de ResNet-50."
"Receta idéntica: pesos de ImageNet, fine tuning completo, AdamW, warm-up más coseno, máximo 12
épocas con early stopping sobre macro-F1 de validación, cross-entropy ponderada con label
smoothing, precisión mixta, semilla 42, una RTX 4070 de 8 GB."

## 5. Tabla de resultados (50 s)  ← la más importante
"Swin-T es el mejor individual: 76.1 % de accuracy y 64.1 % de macro-F1; EfficientNet 57.8,
ResNet 55.1."
"Dos observaciones. El AUC va al revés, ResNet 0.918 y Swin 0.900: Swin da probabilidades más
afiladas, mejores decisiones pero peor ranking; para triage habría que calibrar. Y el costo: Swin
tarda 71 milisegundos por imagen, 2.4 veces ResNet, y usa 1.5 veces más memoria; EfficientNet
entrena en 14 minutos con 1.5 GB."
"El ensamble de los tres sube a 77.8 % de accuracy, 66.5 % de F1 y AUC 0.942."

## 6. Curvas (15 s)
"Todos convergen en las 12 épocas. Swin ya está en 0.60 de F1 en la época 2; las CNN necesitan 5 o
6. El early stopping controla el sobreajuste."

## 7. Accuracy contra costo (25 s)
"Los FLOPs no son latencia: EfficientNet tiene 10 veces menos FLOPs que ResNet y aun así es más
lento en GPU, 37 contra 29 milisegundos, porque sus convoluciones depthwise no aprovechan los
tensor cores. En CPU o móvil sí sería el más rápido. Hay que decidir con latencia medida en el
hardware objetivo."

## 8. Por clase (25 s)
"Fuertes en nevus, basocelular y vascular; débiles en queratosis actínica, espinocelular y
dermatofibroma, que son raras y se parecen. Dos confusiones que también tienen los dermatólogos:
melanoma contra nevus y el grupo de queratosis. Swin es el más uniforme: mejor F1 en 7 de 8 clases."

## 9. Grad-CAM (30 s)
"Las CNN miran el núcleo pigmentado; Swin produce mapas multifocales que siguen el borde y la
asimetría, las señales ABCD de los clínicos. Los fallos enseñan: en la queratosis benigna los tres
miran lo más oscuro y dicen melanoma; en el espinocelular EfficientNet se va al pelo y las escamas
del fondo. En producción habría que segmentar la lesión."

## 10. Complementariedad (25 s)
"Las dos CNN juntas sólo llegan a 61.5 % de F1; EfficientNet más Swin a 66.4 %, casi igual que los
tres. La ganancia viene de mezclar familias: se equivocan en imágenes distintas. Cuesta 137
milisegundos por imagen, bien para screening offline. Limitación: una sola semilla, diferencias
menores a un punto son ruido."

## 11. Conclusiones (30 s)
"Uno: con preentrenamiento el Transformer es el más preciso y uniforme aun con 25 mil imágenes.
Dos: cuesta 2.4 veces la latencia y 1.5 la memoria de ResNet, y el más barato en FLOPs no es el
más rápido en GPU. Tres: las familias son complementarias y el ensamble da el mejor F1 y AUC."
"Recomendación: EfficientNet para CPU o edge, Swin-T con GPU, ensamble para screening offline."

## 12. Cierre (10 s)
"Todo es reproducible: código, logs, figuras y notebook de Colab con ruta rápida de 442 MB en el
repositorio del QR. Gracias, quedo atento a sus preguntas."

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
