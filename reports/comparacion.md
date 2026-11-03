| ID | Modelo | Imágenes train | Épocas | Accuracy | F1 macro | Top-3 | Recall enferma | Confianza prom. | Parámetros | Minutos |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| E1 | CNN propia (desde cero) | 70196 | 17 | 95.0% | 95.0% | 99.7% | 98.7% | 95.3% | 427654 | 56.8 |
| C1 | ResNet50 congelada, 100 img/clase | 3800 | 20 | 89.8% | 89.7% | 98.3% | 94.8% | 89.3% | 23665574 | 10.1 |
| C2 | ResNet50 congelada, 500 img/clase | 19000 | 17 | 93.7% | 93.7% | 99.3% | 96.1% | 94.1% | 23665574 | 19.7 |
| E2 | ResNet50 congelada, todas | 70196 | 5 | 94.2% | 94.1% | 99.4% | 96.8% | 95.0% | 23665574 | 17.0 |
| E3 | ResNet50 fine-tuning, Adam | 70196 | 10 | 99.2% | 99.2% | 100.0% | 99.8% | 99.5% | 23665574 | 41.1 |
| E4 | ResNet50 fine-tuning, SGD | 70196 | 10 | 98.7% | 98.7% | 100.0% | 99.3% | 99.1% | 23665574 | 37.3 |
| E5 | MobileNetV2 fine-tuning | 70196 | 15 | 98.5% | 98.5% | 99.8% | 100.0% | 99.0% | 2306662 | 38.0 |
