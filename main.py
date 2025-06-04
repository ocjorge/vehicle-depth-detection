import cv2
import torch
import numpy as np
import time
import json
import csv
import os
from datetime import datetime
from ultralytics import YOLO
from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict, Counter

class VehicleDepthDetector:
    def __init__(self, model_path, video_path, output_dir='output'):
        """
        Inicializar el detector de vehículos con análisis de profundidad
        
        Args:
            model_path (str): Ruta al modelo YOLO entrenado
            video_path (str): Ruta al video de entrada
            output_dir (str): Directorio de salida para resultados
        """
        self.model_path = model_path
        self.video_path = video_path
        self.output_dir = output_dir
        
        # Crear directorio de salida
        os.makedirs(output_dir, exist_ok=True)
        
        # Configuración de parámetros
        self.CONFIDENCE_THRESHOLD = 0.35
        self.MAX_EXECUTION_TIME = 60 * 60 * 2  # 2 horas
        self.MAX_FRAME_ATTEMPTS = 5
        self.FRAME_SKIP = 0
        
        # Tamaños reales de objetos (en metros)
        self.REAL_OBJECT_SIZES_M = {
            'car': 1.8, 'threewheel': 1.2, 'bus': 2.5, 'truck': 2.6,
            'motorbike': 0.8, 'van': 2.0, 'person': 0.5, 'bicycle': 0.4, 'dog': 0.3
        }
        
        # Datos de análisis
        self.detection_data = []
        self.frame_stats = []
        self.processing_times = []
        
        # Cargar modelos
        self._load_models()
        
        # Configurar video
        self._setup_video()
        
        # Configurar logging
        self._setup_logging()
    
    def _load_models(self):
        """Cargar modelos YOLO y MiDaS"""
        print("Cargando modelos...")
        
        # Cargar YOLO
        self.model_vehicles = YOLO(self.model_path)
        
        # Cargar MiDaS
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Usando dispositivo: {self.device}")
        
        self.midas = torch.hub.load("intel-isl/MiDaS", "DPT_Large")
        self.midas.to(self.device)
        self.midas.eval()
        
        midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
        self.transform = midas_transforms.dpt_transform
        
        print("Modelos cargados exitosamente")
    
    def _setup_video(self):
        """Configurar captura y escritura de video"""
        self.cap = cv2.VideoCapture(self.video_path)
        if not self.cap.isOpened():
            raise RuntimeError(f"No se pudo abrir el video: {self.video_path}")
        
        # Propiedades del video
        self.frame_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.frame_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        print(f"Video info: {self.frame_width}x{self.frame_height} a {self.fps:.2f} FPS ({self.total_frames} frames totales)")
        
        # Configurar VideoWriter para video principal
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        self.out_main = cv2.VideoWriter(
            os.path.join(self.output_dir, 'output_detections.mp4'),
            fourcc, self.fps, (self.frame_width, self.frame_height)
        )
        
        # VideoWriter para mapa de profundidad
        self.out_depth = cv2.VideoWriter(
            os.path.join(self.output_dir, 'output_depth.mp4'),
            fourcc, self.fps, (self.frame_width, self.frame_height)
        )
    
    def _setup_logging(self):
        """Configurar archivo de log"""
        self.log_file = os.path.join(self.output_dir, 'processing_log.txt')
        with open(self.log_file, 'w') as f:
            f.write(f"Inicio del procesamiento: {datetime.now()}\n")
            f.write(f"Video: {self.video_path}\n")
            f.write(f"Modelo: {self.model_path}\n")
            f.write(f"Dispositivo: {self.device}\n")
            f.write(f"Resolución: {self.frame_width}x{self.frame_height}\n")
            f.write(f"FPS: {self.fps}\n")
            f.write(f"Frames totales: {self.total_frames}\n")
            f.write("-" * 50 + "\n")
    
    def _log_message(self, message):
        """Escribir mensaje en log y consola"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        log_msg = f"[{timestamp}] {message}"
        print(log_msg)
        
        with open(self.log_file, 'a') as f:
            f.write(log_msg + "\n")
    
    def _calculate_distance(self, bbox, depth_map, object_class):
        """
        Calcular distancia estimada del objeto usando el mapa de profundidad
        
        Args:
            bbox: Coordenadas del bounding box [x1, y1, x2, y2]
            depth_map: Mapa de profundidad del frame
            object_class: Clase del objeto detectado
        
        Returns:
            float: Distancia estimada en metros
        """
        x1, y1, x2, y2 = bbox
        
        # Extraer región de interés del mapa de profundidad
        depth_roi = depth_map[y1:y2, x1:x2]
        
        # Calcular estadísticas de profundidad
        median_depth = np.median(depth_roi)
        mean_depth = np.mean(depth_roi)
        
        # Usar tamaño real del objeto para calibrar distancia
        real_size = self.REAL_OBJECT_SIZES_M.get(object_class, 1.0)
        bbox_height_pixels = y2 - y1
        
        # Estimación simple de distancia (puede mejorarse)
        # Asumiendo que objetos más lejanos tienen valores de profundidad más altos
        estimated_distance = median_depth * 0.1  # Factor de escala empírico
        
        return {
            'estimated_distance': float(estimated_distance),
            'median_depth': float(median_depth),
            'mean_depth': float(mean_depth),
            'bbox_height': bbox_height_pixels
        }
    
    def _process_frame(self, frame, frame_number, timestamp):
        """
        Procesar un frame individual
        
        Args:
            frame: Frame de video
            frame_number: Número del frame
            timestamp: Timestamp en segundos
        
        Returns:
            tuple: (frame_processed, depth_frame, detections)
        """
        frame_start_time = time.time()
        
        # Convertir a RGB para MiDaS
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Inferencia de profundidad
        input_tensor = self.transform(img_rgb).to(self.device)
        if input_tensor.dim() == 3:
            input_tensor = input_tensor.unsqueeze(0)
        
        with torch.no_grad():
            prediction = self.midas(input_tensor)
            prediction = torch.nn.functional.interpolate(
                prediction.unsqueeze(1),
                size=img_rgb.shape[:2],
                mode="bicubic",
                align_corners=False,
            ).squeeze()
        
        depth_map = prediction.cpu().numpy()
        
        # Normalizar mapa de profundidad para visualización
        depth_normalized = cv2.normalize(depth_map, None, 0, 255, cv2.NORM_MINMAX)
        depth_colored = cv2.applyColorMap(depth_normalized.astype(np.uint8), cv2.COLORMAP_PLASMA)
        
        # Detección de objetos
        results = self.model_vehicles.predict(source=frame, conf=self.CONFIDENCE_THRESHOLD, verbose=False)
        
        # Procesar detecciones
        frame_detections = []
        detection_count = 0
        
        if results[0].boxes is not None:
            for box_data in results[0].boxes:
                x1, y1, x2, y2 = map(int, box_data.xyxy[0])
                confidence = float(box_data.conf[0])
                class_id = int(box_data.cls[0])
                class_name = self.model_vehicles.names[class_id]
                
                # Calcular distancia
                distance_info = self._calculate_distance([x1, y1, x2, y2], depth_map, class_name)
                
                # Dibujar bounding box en frame principal
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                
                # Etiqueta con clase, confianza y distancia
                label = f"{class_name}: {confidence:.2f} | {distance_info['estimated_distance']:.1f}m"
                cv2.putText(frame, label, (x1, y1-10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                
                # Dibujar bounding box en mapa de profundidad
                cv2.rectangle(depth_colored, (x1, y1), (x2, y2), (255, 255, 255), 2)
                cv2.putText(depth_colored, f"{class_name}", (x1, y1-10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                
                # Guardar datos de detección
                detection_data = {
                    'frame_number': frame_number,
                    'timestamp': timestamp,
                    'class': class_name,
                    'confidence': confidence,
                    'bbox': [x1, y1, x2, y2],
                    'bbox_center': [(x1+x2)//2, (y1+y2)//2],
                    **distance_info
                }
                
                frame_detections.append(detection_data)
                detection_count += 1
        
        # Agregar información del frame a la esquina
        info_text = f"Frame: {frame_number} | Objetos: {detection_count} | Tiempo: {timestamp:.1f}s"
        cv2.putText(frame, info_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(depth_colored, f"Mapa de Profundidad - Frame: {frame_number}", 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Estadísticas del frame
        processing_time = time.time() - frame_start_time
        frame_stats = {
            'frame_number': frame_number,
            'timestamp': timestamp,
            'detection_count': detection_count,
            'processing_time': processing_time,
            'detections_by_class': dict(Counter([d['class'] for d in frame_detections]))
        }
        
        return frame, depth_colored, frame_detections, frame_stats
    
    def process_video(self):
        """Procesar video completo"""
        self._log_message("Iniciando procesamiento de video...")
        
        start_time = time.time()
        frame_count = 0
        failed_attempts = 0
        
        while self.cap.isOpened():
            # Verificar tiempo máximo
            if (time.time() - start_time) > self.MAX_EXECUTION_TIME:
                self._log_message(f"Tiempo máximo alcanzado ({self.MAX_EXECUTION_TIME/60:.1f} min)")
                break
            
            # Leer frame
            ret, frame = self.cap.read()
            if not ret:
                failed_attempts += 1
                self._log_message(f"Error leyendo frame. Intento {failed_attempts}/{self.MAX_FRAME_ATTEMPTS}")
                
                if failed_attempts >= self.MAX_FRAME_ATTEMPTS:
                    self._log_message("Demasiados intentos fallidos. Terminando...")
                    break
                
                # Reintentar
                self.cap.release()
                self.cap = cv2.VideoCapture(self.video_path)
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_count + 1)
                continue
            
            failed_attempts = 0
            
            # Saltar frames si es necesario
            if self.FRAME_SKIP > 0 and frame_count % (self.FRAME_SKIP + 1) != 0:
                frame_count += 1
                continue
            
            # Procesar frame
            try:
                timestamp = frame_count / self.fps
                
                processed_frame, depth_frame, detections, stats = self._process_frame(
                    frame, frame_count, timestamp
                )
                
                # Guardar datos
                self.detection_data.extend(detections)
                self.frame_stats.append(stats)
                self.processing_times.append(stats['processing_time'])
                
                # Escribir frames
                self.out_main.write(processed_frame)
                self.out_depth.write(depth_frame)
                
                frame_count += 1
                
                # Progreso cada 100 frames
                if frame_count % 100 == 0:
                    elapsed = time.time() - start_time
                    avg_fps = frame_count / elapsed
                    avg_processing_time = np.mean(self.processing_times[-100:])
                    
                    progress_msg = (f"Procesados {frame_count}/{self.total_frames} frames | "
                                  f"Tiempo: {elapsed:.1f}s | Velocidad: {avg_fps:.2f} FPS | "
                                  f"Proc. promedio: {avg_processing_time:.3f}s/frame")
                    self._log_message(progress_msg)
                
            except Exception as e:
                self._log_message(f"Error procesando frame {frame_count}: {str(e)}")
                continue
        
        # Finalizar
        total_time = time.time() - start_time
        self._log_message(f"Procesamiento completado en {total_time:.1f}s")
        self._log_message(f"Frames procesados: {frame_count}")
        self._log_message(f"Detecciones totales: {len(self.detection_data)}")
        
        self._cleanup()
        self._generate_reports()
    
    def _cleanup(self):
        """Liberar recursos"""
        self.cap.release()
        self.out_main.release()
        self.out_depth.release()
        cv2.destroyAllWindows()
    
    def _generate_reports(self):
        """Generar reportes y análisis"""
        self._log_message("Generando reportes...")
        
        # 1. Guardar datos en JSON
        with open(os.path.join(self.output_dir, 'detections.json'), 'w') as f:
            json.dump(self.detection_data, f, indent=2)
        
        # 2. Guardar datos en CSV
        if self.detection_data:
            df_data = []
            for detection in self.detection_data:
                row = {
                    'frame_number': detection['frame_number'],
                    'timestamp': detection['timestamp'],
                    'class': detection['class'],
                    'confidence': detection['confidence'],
                    'x1': detection['bbox'][0],
                    'y1': detection['bbox'][1],
                    'x2': detection['bbox'][2],
                    'y2': detection['bbox'][3],
                    'center_x': detection['bbox_center'][0],
                    'center_y': detection['bbox_center'][1],
                    'estimated_distance': detection['estimated_distance'],
                    'median_depth': detection['median_depth'],
                    'mean_depth': detection['mean_depth'],
                    'bbox_height': detection['bbox_height']
                }
                df_data.append(row)
            
            with open(os.path.join(self.output_dir, 'detections.csv'), 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=df_data[0].keys())
                writer.writeheader()
                writer.writerows(df_data)
        
        # 3. Estadísticas generales
        self._generate_statistics()
        
        # 4. Visualizaciones
        self._generate_visualizations()
        
        self._log_message("Reportes generados exitosamente")
    
    def _generate_statistics(self):
        """Generar estadísticas del análisis"""
        stats = {
            'video_info': {
                'path': self.video_path,
                'resolution': f"{self.frame_width}x{self.frame_height}",
                'fps': self.fps,
                'total_frames': self.total_frames,
                'duration_seconds': self.total_frames / self.fps
            },
            'processing_info': {
                'frames_processed': len(self.frame_stats),
                'total_detections': len(self.detection_data),
                'avg_processing_time_per_frame': np.mean(self.processing_times),
                'total_processing_time': sum(self.processing_times)
            },
            'detection_stats': {}
        }
        
        if self.detection_data:
            # Estadísticas por clase
            class_counts = Counter([d['class'] for d in self.detection_data])
            stats['detection_stats']['by_class'] = dict(class_counts)
            
            # Estadísticas de confianza
            confidences = [d['confidence'] for d in self.detection_data]
            stats['detection_stats']['confidence'] = {
                'mean': np.mean(confidences),
                'std': np.std(confidences),
                'min': np.min(confidences),
                'max': np.max(confidences)
            }
            
            # Estadísticas de distancia
            distances = [d['estimated_distance'] for d in self.detection_data]
            stats['detection_stats']['distance'] = {
                'mean': np.mean(distances),
                'std': np.std(distances),
                'min': np.min(distances),
                'max': np.max(distances)
            }
        
        # Guardar estadísticas
        with open(os.path.join(self.output_dir, 'statistics.json'), 'w') as f:
            json.dump(stats, f, indent=2)
        
        # Reporte de texto
        with open(os.path.join(self.output_dir, 'report.txt'), 'w') as f:
            f.write("REPORTE DE ANÁLISIS DE VIDEO\n")
            f.write("=" * 50 + "\n\n")
            
            f.write("INFORMACIÓN DEL VIDEO:\n")
            f.write(f"Archivo: {stats['video_info']['path']}\n")
            f.write(f"Resolución: {stats['video_info']['resolution']}\n")
            f.write(f"FPS: {stats['video_info']['fps']}\n")
            f.write(f"Duración: {stats['video_info']['duration_seconds']:.1f} segundos\n")
            f.write(f"Frames totales: {stats['video_info']['total_frames']}\n\n")
            
            f.write("PROCESAMIENTO:\n")
            f.write(f"Frames procesados: {stats['processing_info']['frames_processed']}\n")
            f.write(f"Detecciones totales: {stats['processing_info']['total_detections']}\n")
            f.write(f"Tiempo promedio por frame: {stats['processing_info']['avg_processing_time_per_frame']:.3f}s\n")
            f.write(f"Tiempo total de procesamiento: {stats['processing_info']['total_processing_time']:.1f}s\n\n")
            
            if stats['detection_stats']:
                f.write("DETECCIONES POR CLASE:\n")
                for class_name, count in stats['detection_stats']['by_class'].items():
                    f.write(f"  {class_name}: {count}\n")
                
                f.write(f"\nCONFIANZA PROMEDIO: {stats['detection_stats']['confidence']['mean']:.3f}\n")
                f.write(f"DISTANCIA PROMEDIO: {stats['detection_stats']['distance']['mean']:.1f}m\n")
    
    def _generate_visualizations(self):
        """Generar gráficos de análisis"""
        if not self.detection_data:
            return
        
        plt.style.use('default')
        
        # 1. Distribución de clases
        class_counts = Counter([d['class'] for d in self.detection_data])
        
        plt.figure(figsize=(12, 6))
        plt.subplot(2, 2, 1)
        plt.bar(class_counts.keys(), class_counts.values())
        plt.title('Distribución de Detecciones por Clase')
        plt.xlabel('Clase')
        plt.ylabel('Cantidad')
        plt.xticks(rotation=45)
        
        # 2. Histograma de confianza
        plt.subplot(2, 2, 2)
        confidences = [d['confidence'] for d in self.detection_data]
        plt.hist(confidences, bins=20, alpha=0.7)
        plt.title('Distribución de Confianza')
        plt.xlabel('Confianza')
        plt.ylabel('Frecuencia')
        
        # 3. Detecciones por frame
        plt.subplot(2, 2, 3)
        frame_counts = [s['detection_count'] for s in self.frame_stats]
        plt.plot(frame_counts)
        plt.title('Detecciones por Frame')
        plt.xlabel('Número de Frame')
        plt.ylabel('Cantidad de Detecciones')
        
        # 4. Tiempo de procesamiento
        plt.subplot(2, 2, 4)
        plt.plot(self.processing_times)
        plt.title('Tiempo de Procesamiento por Frame')
        plt.xlabel('Número de Frame')
        plt.ylabel('Tiempo (segundos)')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'analysis_charts.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        # 5. Mapa de calor de detecciones
        if len(self.detection_data) > 0:
            plt.figure(figsize=(10, 8))
            
            # Crear grid de posiciones
            x_positions = [d['bbox_center'][0] for d in self.detection_data]
            y_positions = [d['bbox_center'][1] for d in self.detection_data]
            
            plt.hexbin(x_positions, y_positions, gridsize=30, cmap='YlOrRd')
            plt.colorbar(label='Densidad de Detecciones')
            plt.title('Mapa de Calor de Detecciones')
            plt.xlabel('Posición X (píxeles)')
            plt.ylabel('Posición Y (píxeles)')
            plt.gca().invert_yaxis()  # Invertir Y para coincidir con coordenadas de imagen
            
            plt.savefig(os.path.join(self.output_dir, 'detection_heatmap.png'), dpi=300, bbox_inches='tight')
            plt.close()


def main():
    """Función principal"""
    
    # Configuración de rutas
    MODEL_PATH = 'F:/Documents/PycharmProjects/MiDaSDetector/best.pt'
    VIDEO_PATH = 'F:/Documents/PycharmProjects/MiDaSDetector/GH012372_no_audio.mp4'
    OUTPUT_DIR = 'results'
    
    # Verificar que los archivos existen
    if not os.path.exists(MODEL_PATH):
        print(f"Error: No se encontró el modelo en {MODEL_PATH}")
        return
    
    if not os.path.exists(VIDEO_PATH):
        print(f"Error: No se encontró el video en {VIDEO_PATH}")
        return
    
    try:
        # Crear detector
        detector = VehicleDepthDetector(MODEL_PATH, VIDEO_PATH, OUTPUT_DIR)
        
        # Procesar video
        detector.process_video()
        
        print(f"\n✅ Procesamiento completado exitosamente!")
        print(f"📁 Resultados guardados en: {OUTPUT_DIR}/")
        print("\n📄 Archivos generados:")
        print("  - output_detections.mp4    (Video con detecciones)")
        print("  - output_depth.mp4         (Video con mapa de profundidad)")
        print("  - detections.json          (Datos de detecciones)")
        print("  - detections.csv           (Datos en formato CSV)")
        print("  - statistics.json          (Estadísticas del análisis)")
        print("  - report.txt               (Reporte completo)")
        print("  - analysis_charts.png      (Gráficos de análisis)")
        print("  - detection_heatmap.png    (Mapa de calor)")
        print("  - processing_log.txt       (Log del procesamiento)")
        
    except Exception as e:
        print(f"❌ Error durante el procesamiento: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
