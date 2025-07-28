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
import pyvirtualcam

# Variáveis globais
last_speech_emotion_display = "N/A"   # última emoção da fala conhecida
last_mouth_coords = None
last_combined_emotion = "N/A"         # última emoção agregada conhecida

# mapeamento de opostos para detectar ironia
opposite_pairs = [
    ("joy", "sadness"),
    ("happiness", "sadness"),
    ("anger", "joy"),
    ("anger", "happiness"),
    ("fear", "confidence"),
    ("surprise", "neutral")
]

def are_opposites(emotion1, emotion2):
    e1 = emotion1.lower()
    e2 = emotion2.lower()
    for a, b in opposite_pairs:
        if (e1 == a and e2 == b) or (e1 == b and e2 == a):
            return True
    return False

def main():
    global last_speech_emotion_display, last_mouth_coords, last_combined_emotion
    parser = argparse.ArgumentParser(description='Camera and analytics arguments')
    parser.add_argument('--cameraID', type=int, default=0, help='ID da câmera caso for uma webcam')
    parser.add_argument('--saveframe', type=str, default="false", help='Salvar ou não salvar os frames')
    parser.add_argument('--device', type=str, default="gpu", help='Usar CPU/GPU')
    args = parser.parse_args()

    saveframe = args.saveframe.lower() == "true"
    dev = 'cuda' if args.device.lower() == 'gpu' else 'cpu'

    print('Dispositivo de processamento:', dev)

    video_stream = cv2.VideoCapture(args.cameraID)
    if not video_stream.isOpened():
        print(f"[ERRO] Não foi possível abrir a câmera com ID {args.cameraID}.")
        return

    fer_pipeline = EmotionPerceiver(device=dev)
    prev_face_emotion = ""
    prev_speech_emotion_console = ""

    # inicia thread da fala
    start_speech_thread()

    # Configura câmera virtual
    width, height, fps = 1280, 720, 20
    cam = None
    backend_order = ('dshow', 'obs', 'unitycapture', None)
    for backend in backend_order:
        try:
            if backend:
                cam = pyvirtualcam.Camera(width=width, height=height, fps=fps, backend=backend)
            else:
                cam = pyvirtualcam.Camera(width=width, height=height, fps=fps)
            print(f"[INFO] Virtual cam ativada: {cam.device} (backend={backend})")
            break
        except Exception as e:
            print(f"[WARN] pyvirtualcam backend={backend} falhou: {e}")
    else:
        print("[WARN] Nenhum backend funcionou, usando apenas preview local.")

    try:
        while True:
            grabbed, frame = video_stream.read()
            if not grabbed:
                time.sleep(0.1)
                continue

            frame = cv2.resize(frame, (1280, 720))
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            face_locations, mouth_locations = get_face_locations_mediapipe(frame_rgb)

            latest_speech_data = get_latest_speech_emotion()

            if face_locations:
                for face_idx, face_location in enumerate(face_locations):
                    (x1_face, y1_face), (x2_face, y2_face) = face_location
                    face_roi = frame_rgb[y1_face:y2_face, x1_face:x2_face, :]

                    if face_roi.shape[0] > 0 and face_roi.shape[1] > 0:
                        face_emotion, face_scores = fer_pipeline.predict_emotions(face_roi, logits=True)
                        if prev_face_emotion != face_emotion:
                            print(f"[FACE] Emoção detectada na face: {face_emotion}")
                            prev_face_emotion = face_emotion

                        cv2.rectangle(frame, (x1_face, y1_face), (x2_face, y2_face), (0, 255, 0), 3)
                        cv2.putText(frame, face_emotion, (x1_face, y1_face - 5),
                                    cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 2)

                        if latest_speech_data:
                            speech_probs = latest_speech_data['probabilities']
                            speech_dom = max(speech_probs, key=speech_probs.get)
                            last_speech_emotion_display = speech_dom
                            combined_probs = {}
                            for label in face_scores.keys():
                                pf = face_scores.get(label, 0.0)
                                ps = speech_probs.get(label, 0.0)
                                combined_probs[label] = (0.3 * pf) + (0.7 * ps)
                            if are_opposites(speech_dom, face_emotion):
                                if speech_probs[speech_dom] > 0.6 and face_scores.get(face_emotion, 0) > 0.6:
                                    last_combined_emotion = "Irony"
                                else:
                                    last_combined_emotion = max(combined_probs, key=combined_probs.get)
                            else:
                                last_combined_emotion = max(combined_probs, key=combined_probs.get)
                            if prev_speech_emotion_console != speech_dom:
                                print(f"[FALA] Emoção detectada na fala: {speech_dom} (Prob: {speech_probs[speech_dom]:.2f})")
                                prev_speech_emotion_console = speech_dom

                        cv2.rectangle(frame,
                                      (x1_face, y1_face),
                                      (x2_face, y1_face + 40),
                                      (0, 255, 255), -1)
                        cv2.putText(frame, f"Agr: {last_combined_emotion}",
                                    (x1_face + 5, y1_face + 30),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)

                        if mouth_locations and face_idx == 0:
                            (x1_mouth, y1_mouth), (x2_mouth, y2_mouth) = mouth_locations[0]
                            last_mouth_coords = ((x1_mouth, y1_mouth), (x2_mouth, y2_mouth))
                            cv2.rectangle(frame, (x1_mouth, y1_mouth), (x2_mouth, y2_mouth), (255, 0, 0), 2)
                            cv2.putText(frame, f"Fala: {last_speech_emotion_display}",
                                        (x1_mouth, y2_mouth + 25),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

            elif last_mouth_coords:
                (x1_mouth, y1_mouth), (x2_mouth, y2_mouth) = last_mouth_coords
                cv2.rectangle(frame, (x1_mouth, y1_mouth), (x2_mouth, y2_mouth), (255, 0, 0), 2)
                cv2.putText(frame, f"Fala: {last_speech_emotion_display}",
                            (x1_mouth, y2_mouth + 25),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

            # envia para virtual cam e mostra preview
            if cam:
                try:
                    cam.send(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                    cam.sleep_until_next_frame()
                except Exception as e:
                    print(f"[WARN] falha ao enviar para virtual cam: {e}")
            cv2.imshow('Preview', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            if saveframe:
                ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S%f")
                cv2.imwrite(f"frame_{ts}.png", frame)

    except KeyboardInterrupt:
        print("[MAIN] Interrupção pelo usuário (Ctrl+C).")
    except Exception as e:
        print(f"[ERRO] Um erro inesperado ocorreu na função main: {e}")
    finally:
        print("Encerrando aplicação...")
        stop_speech_thread()
        video_stream.release()
        cv2.destroyAllWindows()
        if cam:
            cam.close()
        print("Aplicação finalizada.")

if __name__ == '__main__':
    main()
