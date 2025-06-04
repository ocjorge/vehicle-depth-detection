# Vehicle Detection with Depth Estimation

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/downloads/)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.x-green.svg)](https://opencv.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-Latest-red.svg)](https://pytorch.org/)
[![YOLO](https://img.shields.io/badge/YOLO-Ultralytics-yellow.svg)](https://ultralytics.com/)
[![MiDaS](https://img.shields.io/badge/MiDaS-Intel-orange.svg)](https://github.com/isl-org/MiDaS)
[![License](https://img.shields.io/badge/License-MIT-lightgrey.svg)](LICENSE)

## 📋 Descripción

Este proyecto combina **detección de objetos vehiculares** con **estimación de profundidad** para analizar videos de tráfico. Utiliza un modelo YOLO personalizado para detectar vehículos y el modelo MiDaS de Intel para calcular mapas de profundidad, proporcionando información espacial completa de la escena.

## ✨ Características

- 🚗 **Detección multi-clase**: Identifica carros, buses, camiones, motocicletas, bicicletas, personas y más
- 📏 **Estimación de profundidad**: Calcula distancias relativas usando MiDaS DPT_Large
- 🎥 **Procesamiento de video**: Analiza videos completos con control de rendimiento
- 🛡️ **Robusto y seguro**: Incluye manejo de errores y límites de tiempo de ejecución
- ⚡ **Optimizado**: Soporte para GPU y opciones de salto de frames

## 🔧 Requisitos

### Dependencias principales
```
torch>=1.9.0
torchvision>=0.10.0
opencv-python>=4.5.0
ultralytics>=8.0.0
Pillow>=8.0.0
numpy>=1.21.0
```

### Hardware recomendado
- **GPU**: NVIDIA con soporte CUDA (recomendado)
- **RAM**: Mínimo 8GB, recomendado 16GB+
- **Almacenamiento**: Espacio suficiente para videos de entrada y salida

## 🚀 Instalación

1. **Clonar el repositorio**
```bash
git clone https://github.com/tu-usuario/vehicle-depth-detection.git
cd vehicle-depth-detection
```

2. **Crear entorno virtual**
```bash
python -m venv venv
source venv/bin/activate  # En Windows: venv\Scripts\activate
```

3. **Instalar dependencias**
```bash
pip install -r requirements.txt
```

4. **Descargar modelo personalizado**
   - Coloca tu modelo YOLO entrenado (`best.pt`) en la carpeta del proyecto
   - Los modelos MiDaS se descargan automáticamente en la primera ejecución

## 📖 Uso

### Configuración básica

1. **Editar rutas en el código**:
```python
VIDEO_INPUT_PATH = 'ruta/a/tu/video.mp4'
OUTPUT_VIDEO_PATH = 'video_procesado.mp4'
model_vehicles = YOLO('ruta/a/tu/modelo.pt')
```

2. **Ejecutar el procesamiento**:
```bash
python main.py
```

### Parámetros configurables

| Parámetro | Descripción | Valor por defecto |
|-----------|-------------|-------------------|
| `CONFIDENCE_THRESHOLD` | Umbral de confianza para detecciones | 0.35 |
| `FRAME_SKIP` | Frames a saltar (0 = procesar todos) | 0 |
| `MAX_EXECUTION_TIME` | Tiempo máximo de ejecución (segundos) | 7200 (2 horas) |

## 🎯 Clases detectadas

El modelo puede identificar las siguientes clases de objetos:

- 🚗 **car** - Automóviles (tamaño real: 1.8m)
- 🛺 **threewheel** - Vehículos de tres ruedas (1.2m)
- 🚌 **bus** - Autobuses (2.5m)
- 🚚 **truck** - Camiones (2.6m)
- 🏍️ **motorbike** - Motocicletas (0.8m)
- 🚐 **van** - Camionetas (2.0m)
- 👤 **person** - Personas (0.5m)
- 🚲 **bicycle** - Bicicletas (0.4m)
- 🐕 **dog** - Perros (0.3m)

## 🔄 Flujo de procesamiento

1. **Carga de modelos**: YOLO personalizado + MiDaS DPT_Large
2. **Lectura de video**: Frame por frame con manejo de errores
3. **Estimación de profundidad**: Conversión RGB → Tensor → Mapa de profundidad
4. **Detección de objetos**: Identificación y localización de vehículos
5. **Visualización**: Dibujo de bounding boxes
6. **Guardado**: Exportación del video procesado

## 📊 Rendimiento

- **Velocidad**: Depende del hardware (GPU recomendada)
- **Precisión**: Ajustable mediante `CONFIDENCE_THRESHOLD`
- **Memoria**: Optimizado para videos de alta resolución
- **Tiempo**: Progreso reportado cada 100 frames

## 🤝 Contribuciones

Las contribuciones son bienvenidas. Por favor:

1. Fork el proyecto
2. Crea una rama para tu feature (`git checkout -b feature/nueva-funcionalidad`)
3. Commit tus cambios (`git commit -m 'Añadir nueva funcionalidad'`)
4. Push a la rama (`git push origin feature/nueva-funcionalidad`)
5. Abre un Pull Request

## 📄 Licencia

Este proyecto está bajo la Licencia MIT. Ver el archivo [LICENSE](LICENSE) para más detalles.

## 🙏 Agradecimientos

- [Ultralytics](https://ultralytics.com/) por YOLO
- [Intel ISL](https://github.com/isl-org/MiDaS) por MiDaS
- [OpenCV](https://opencv.org/) por las herramientas de visión computacional
- [PyTorch](https://pytorch.org/) por el framework de deep learning

## 📞 Contacto

- **Autor**: Tu Nombre
- **Email**: tu.email@ejemplo.com
- **GitHub**: [@tu-usuario](https://github.com/tu-usuario)

---

⭐ **¡No olvides dar una estrella al proyecto si te resulta útil!**
