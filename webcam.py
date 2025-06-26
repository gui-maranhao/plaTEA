import os
import numpy as np
import cv2
import torch
from utils import EmotionPerceiver, get_face_locations_mediapipe
import datetime
import argparse
import threading
import time
import queue
from speech import start_speech_thread, stop_speech_thread, get_latest_speech_emotion


last_speech_emotion_display = "N/A"
last_mouth_coords = None

# Fila para coletar as emoções faciais
facial_emotion_queue = queue.Queue()

def main():
    global last_speech_emotion_display
    global last_mouth_coords
    
    parser = argparse.ArgumentParser(description='Camera and analytics arguments')
    parser.add_argument('--cameraID', type=int, default=0, help='ID da camera caso for uma webcam')
    parser.add_argument('--saveframe', type=str, default="false", help='Salvar ou nao salvar os frames')
    parser.add_argument('--device', type=str, default="gpu", help='Usar CPU/GPU')
    args = parser.parse_args()
    
    saveframe = args.saveframe
    dev = args.device
    
    print('Dispositivo de processamento:', dev)
    if dev == 'gpu':
        dev = 'cuda'

    video_stream = cv2.VideoCapture(args.cameraID)
    if not video_stream.isOpened():
        print(f"[ERRO] Não foi possível abrir a câmera com ID {args.cameraID}.")
        return

    fer_pipeline = EmotionPerceiver(device=dev)
    prev_face_emotion = ""
    prev_speech_emotion_console = "" 

    start_speech_thread()

    try:
        while True:
            (grabbed, frame) = video_stream.read()
            if not grabbed:
                print(f'[INFO] Skipping due to lack of frames.')
                time.sleep(0.1)
                continue
            
            frame = cv2.resize(frame, (1280, 720))
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            face_locations, mouth_locations = get_face_locations_mediapipe(frame_rgb)

            if face_locations:
                for face_idx, face_location in enumerate(face_locations):
                    (x1_face, y1_face), (x2_face, y2_face) = face_location
                    
                    face_roi = frame_rgb[y1_face:y2_face, x1_face:x2_face, :]
                    
                    if face_roi.shape[0] > 0 and face_roi.shape[1] > 0:
                        emotion, scores = fer_pipeline.predict_emotions(face_roi, logits=True)
                        
                        if prev_face_emotion != emotion:
                            print(f"[FACE] Emoção detectada na face: {emotion}")
                            prev_face_emotion = emotion

                        cv2.rectangle(frame, (x1_face, y1_face), (x2_face, y2_face), (0, 255, 0), 3)
                        cv2.putText(frame, emotion, (x1_face, y1_face - 5),
                                    cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 2)
                        
                        if mouth_locations and face_idx == 0:
                            (x1_mouth, y1_mouth), (x2_mouth, y2_mouth) = mouth_locations[0]
                            last_mouth_coords = ((x1_mouth, y1_mouth), (x2_mouth, y2_mouth)) 
                            
                            cv2.rectangle(frame, (x1_mouth, y1_mouth), (x2_mouth, y2_mouth), (255, 0, 0), 2)
                            
                            text_speech = f"Fala: {last_speech_emotion_display}"
                            text_x = x1_mouth
                            text_y = y2_mouth + 25
                            cv2.putText(frame, text_speech, (text_x, text_y),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
                    else:
                        print("[INFO] Região da face inválida para detecção de emoção facial.")
            elif last_mouth_coords:
                (x1_mouth, y1_mouth), (x2_mouth, y2_mouth) = last_mouth_coords
                cv2.rectangle(frame, (x1_mouth, y1_mouth), (x2_mouth, y2_mouth), (255, 0, 0), 2)
                
                text_speech = f"Fala: {last_speech_emotion_display}"
                text_x = x1_mouth
                text_y = y2_mouth + 25
                cv2.putText(frame, text_speech, (text_x, text_y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

            cv2.imshow('Frame', frame)
            
            # --- Atualização da Emoção da Fala (em segundo plano) ---
            latest_speech_data = get_latest_speech_emotion()
            if latest_speech_data:
                speech_emotion = latest_speech_data['dominant_emotion']
                last_speech_emotion_display = speech_emotion
                
                if prev_speech_emotion_console != speech_emotion:
                    print(f"[FALA] Emoção detectada na fala: {speech_emotion} (Prob: {latest_speech_data['probabilities'][speech_emotion]:.2f})")
                    prev_speech_emotion_console = speech_emotion

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break

            if saveframe == "true":
                current_datetime = datetime.datetime.now()

    except KeyboardInterrupt:
        print("[MAIN] Interrupção pelo usuário (Ctrl+C).")
    except Exception as e:
        print(f"[ERRO] Um erro inesperado ocorreu na função main: {e}")
    finally:
        # --- ENCERRAMENTO ---
        print("Encerrando aplicação...")
        stop_speech_thread()
        video_stream.release()
        cv2.destroyAllWindows()
        print("Aplicação finalizada.")

if __name__ == '__main__':
    main()