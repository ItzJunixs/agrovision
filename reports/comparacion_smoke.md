| ID | Modelo | Imágenes train | Épocas | Accuracy | F1 macro | Top-3 | Recall enferma | Confianza prom. | Parámetros | Minutos |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| E1 | CNN propia (desde cero) | 3508 | 1 | 2.2% | 0.2% | 7.6% | 100.0% | 18.5% | 427654 | 0.4 |
| C1 | ResNet50 congelada, 100 img/clase | 190 | 1 | 8.5% | 5.7% | 21.8% | 71.6% | 13.4% | 23665574 | 0.2 |
| C2 | ResNet50 congelada, 500 img/clase | 950 | 1 | 49.2% | 45.4% | 68.3% | 83.8% | 27.9% | 23665574 | 0.3 |
| E2 | ResNet50 congelada, todas | 3508 | 1 | 76.6% | 75.1% | 92.8% | 91.8% | 61.6% | 23665574 | 0.4 |
| E3 | ResNet50 fine-tuning, Adam | 3508 | 1 | 82.0% | 80.9% | 95.3% | 98.4% | 80.5% | 23665574 | 0.5 |
| E4 | ResNet50 fine-tuning, SGD | 3508 | 1 | 84.7% | 83.4% | 96.6% | 93.4% | 87.2% | 23665574 | 0.4 |
| E5 | MobileNetV2 fine-tuning | 3508 | 2 | 78.2% | 76.5% | 93.9% | 95.7% | 70.0% | 2306662 | 0.5 |
