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
import csv
from speech import start_speech_thread, stop_speech_thread, get_latest_speech_emotion
import pyvirtualcam
from collections import Counter

# Variáveis globais
last_speech_emotion_display = "N/A"
last_mouth_coords = None
last_combined_emotion = None  # None = ainda não calculado
debug_mode = False  # Modo debug ON/OFF

# Coleta temporizada
collecting = False
collected_face = []
collected_speech = []
collected_agg = []

# Controle para saber se já vimos essa fala (timestamp)
last_speech_ts = None

# Guarda última face detectada para uso quando chega fala mas não há face no mesmo frame
last_face_emotion_global = None
last_face_scores_global = None

# mapeamento de opostos para detectar ironia
opposite_pairs = [
    ("joy", "sadness"),
    ("happiness", "sadness"),
    ("anger", "joy"),
    ("anger", "happiness"),
    ("fear", "confidence"),
    ("surprise", "neutral")
]

emotion_color_map = {
    'anger': (0, 0, 255),
    'sadness': (255, 0, 0),
    'disgust': (0, 128, 0),
    'fear': (128, 0, 128),
    'surprise': (0, 165, 255),
    'happiness': (0, 255, 255),
    'contempt': (0, 255, 0),
    'neutral': (192, 192, 192),
    'irony': (203, 182, 255),
}

def are_opposites(emotion1, emotion2):
    e1 = (emotion1 or "").lower()
    e2 = (emotion2 or "").lower()
    return any((e1 == a and e2 == b) or (e1 == b and e2 == a) for a, b in opposite_pairs)

def salvar_csv(face_list, speech_list, agg_list):
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"emocoes_coleta_{ts}.csv"
    with open(filename, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["Frame", "Face", "Fala", "Agregado"])
        for i in range(len(agg_list)):
            face = face_list[i] if i < len(face_list) else ""
            speech = speech_list[i] if i < len(speech_list) else ""
            agg = agg_list[i] if i < len(agg_list) else ""
            writer.writerow([i+1, face, speech, agg])
    print(f"[INFO] CSV salvo: {filename}")

def resumo_emocoes(lista, nome):
    if not lista:
        print(f"[RESUMO] Nenhum dado coletado para {nome}.")
        return
    contagem = Counter(lista)
    total = len(lista)
    print(f"\n[RESUMO] {nome}:")
    for emo, count in contagem.most_common():
        perc = (count / total) * 100
        print(f"  {emo}: {count} ({perc:.1f}%)")
    mais_freq = contagem.most_common(1)[0][0]
    print(f"  Mais frequente: {mais_freq}")

def main():
    global last_speech_emotion_display, last_mouth_coords, last_combined_emotion
    global debug_mode, collecting, collected_face, collected_speech, collected_agg
    global last_speech_ts, last_face_emotion_global, last_face_scores_global

    parser = argparse.ArgumentParser(description='Camera and analytics arguments')
    parser.add_argument('--cameraID', type=int, default=0)
    parser.add_argument('--saveframe', type=str, default="false")
    parser.add_argument('--device', type=str, default="gpu")
    args = parser.parse_args()

    saveframe = args.saveframe.lower() == "true"
    dev = 'cuda' if args.device.lower() == 'gpu' else 'cpu'
    print('Dispositivo:', dev)

    video_stream = cv2.VideoCapture(args.cameraID)
    if not video_stream.isOpened():
        print(f"[ERRO] Não foi possível abrir a câmera {args.cameraID}.")
        return

    fer_pipeline = EmotionPerceiver(device=dev)
    prev_face_emotion = ""
    prev_speech_emotion_console = ""

    start_speech_thread()

    width, height, fps = 1280, 720, 20
    cam = None
    for backend in ('dshow', 'obs', 'unitycapture', None):
        try:
            cam = pyvirtualcam.Camera(width=width, height=height, fps=fps, backend=backend) if backend else pyvirtualcam.Camera(width=width, height=height, fps=fps)
            print(f"[INFO] Virtual cam: {cam.device} (backend={backend})")
            break
        except Exception:
            pass

    try:
        while True:
            grabbed, frame = video_stream.read()
            if not grabbed:
                time.sleep(0.1)
                continue

            frame = cv2.resize(frame, (1280, 720))
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            face_locations, mouth_locations = get_face_locations_mediapipe(frame_rgb)

            # pega última fala (se houver). retorna None se não houver novas entradas na fila
            latest_speech_data = get_latest_speech_emotion()

            # Detecta se é UMA NOVA fala (com base no timestamp)
            new_speech_arrived = False
            speech_probs = None
            speech_dom = None
            if latest_speech_data:
                ts = latest_speech_data.get('timestamp', None)
                if ts != last_speech_ts:
                    new_speech_arrived = True
                    last_speech_ts = ts
                    speech_probs = latest_speech_data.get('probabilities', {})
                    if speech_probs:
                        speech_dom = max(speech_probs, key=speech_probs.get)
                        last_speech_emotion_display = speech_dom
                        if prev_speech_emotion_console != speech_dom:
                            print(f"[FALA] Emoção detectada na fala: {speech_dom} (Prob: {speech_probs.get(speech_dom,0):.2f})")
                            prev_speech_emotion_console = speech_dom
            # se não veio nova fala, speech_probs e speech_dom continuam None (não atualizamos agregado)

            # reset do agregado por face (mas só atualizamos realmente quando nova fala chega)
            # NOTA: last_combined_emotion é mantido entre frames até chegar nova fala
            if face_locations:
                for face_idx, face_location in enumerate(face_locations):
                    (x1_face, y1_face), (x2_face, y2_face) = face_location

                    # --- calcula emoção da face sempre (para exibir) ---
                    face_emotion, face_scores = None, {}
                    face_roi = frame_rgb[y1_face:y2_face, x1_face:x2_face, :]
                    if face_roi.size > 0:
                        face_emotion, face_scores = fer_pipeline.predict_emotions(face_roi, logits=True)
                        # atualiza últimos valores de face para usar caso a fala chegue sem face no mesmo frame
                        last_face_emotion_global = face_emotion
                        last_face_scores_global = face_scores
                        if prev_face_emotion != face_emotion:
                            prev_face_emotion = face_emotion
                            # print(f"[FACE] Emoção detectada na face: {face_emotion}")

                    # --- Se chegou nova fala, recalcula a agregada agora (com face atual se houver, senão com última face conhecida) ---
                    combined_updated_this_frame = False
                    if new_speech_arrived:
                        # tenta usar face_scores (do face_roi atual); caso contrário, usa últimos armazenados
                        use_face_scores = face_scores if face_scores else last_face_scores_global
                        use_face_emotion = face_emotion if face_emotion else last_face_emotion_global

                        if use_face_scores and speech_probs:
                            labels = set(list(use_face_scores.keys()) + list(speech_probs.keys()))
                            combined_probs = {label: (0.35 * use_face_scores.get(label, 0.0) + 0.65 * speech_probs.get(label, 0.0))
                                              for label in labels}
                            if are_opposites(speech_dom, use_face_emotion):
                                if speech_probs.get(speech_dom, 0) > 0.6 and (use_face_scores.get(use_face_emotion, 0) > 0.6):
                                    last_combined_emotion = "Irony"
                                else:
                                    last_combined_emotion = max(combined_probs, key=combined_probs.get)
                            else:
                                last_combined_emotion = max(combined_probs, key=combined_probs.get)
                        elif use_face_emotion:
                            # fala nova existe mas não conseguimos scores da face -> fallback para face
                            last_combined_emotion = use_face_emotion
                        elif speech_dom:
                            # fala nova mas sem face conhecida -> fallback para fala
                            last_combined_emotion = speech_dom
                        else:
                            last_combined_emotion = None

                        combined_updated_this_frame = True

                    # --- Inicialização da agregada (somente se ainda for None) ---
                    # isso faz com que no primeiro frame com face a agregada não seja None,
                    # mas NÃO é considerada "atualização por fala".
                    if last_combined_emotion is None and last_face_emotion_global:
                        last_combined_emotion = last_face_emotion_global
                        # NOTA: não marca combined_updated_this_frame = True

                    # --- Desenhos em modo debug (face + fala) ---
                    if debug_mode:
                        # face rect e label
                        cv2.rectangle(frame, (x1_face, y1_face), (x2_face, y2_face), (0, 255, 0), 3)
                        cv2.putText(frame, face_emotion or "?", (x1_face, y1_face - 5),
                                    cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 2)
                        # mouth / fala box (apenas para a face 0)
                        if mouth_locations and face_idx == 0:
                            (x1_mouth, y1_mouth), (x2_mouth, y2_mouth) = mouth_locations[0]
                            last_mouth_coords = ((x1_mouth, y1_mouth), (x2_mouth, y2_mouth))
                            cv2.rectangle(frame, (x1_mouth, y1_mouth), (x2_mouth, y2_mouth), (255, 0, 0), 2)
                            cv2.putText(frame, f"Fala: {last_speech_emotion_display}", (x1_mouth, y2_mouth + 25),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

                    # --- rect do agregado: APARECE SEMPRE ---
                    agg_text = last_combined_emotion if last_combined_emotion else "N/A"
                    if last_combined_emotion:
                        color = emotion_color_map.get(last_combined_emotion.lower(), (255, 255, 255))
                    else:
                        color = emotion_color_map.get('neutral', (192, 192, 192))
                    cv2.rectangle(frame, (x1_face, y1_face), (x2_face, y1_face + 40), color, -1)
                    cv2.putText(frame, f"Agr: {agg_text}", (x1_face + 5, y1_face + 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)

                    # --- gravação: só se coletando E se a agregada foi atualizada por UMA fala NOVA neste frame ---
                    if collecting and combined_updated_this_frame:
                        collected_face.append(face_emotion or "N/A")
                        collected_speech.append(last_speech_emotion_display if last_speech_emotion_display else "N/A")
                        collected_agg.append(last_combined_emotion if last_combined_emotion else "N/A")

            else:
                # sem face no frame atual: desenha boca do último frame (se existir) apenas em debug
                if last_mouth_coords and debug_mode:
                    (x1_mouth, y1_mouth), (x2_mouth, y2_mouth) = last_mouth_coords
                    cv2.rectangle(frame, (x1_mouth, y1_mouth), (x2_mouth, y2_mouth), (255, 0, 0), 2)
                    cv2.putText(frame, f"Fala: {last_speech_emotion_display}", (x1_mouth, y2_mouth + 25),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

            # envia para a virtual cam (inverte depois)
            output_frame = cv2.flip(frame, 1)

            if cam:
                try:
                    cam.send(cv2.cvtColor(output_frame, cv2.COLOR_BGR2RGB))
                    cam.sleep_until_next_frame()
                except Exception:
                    pass

            cv2.imshow('Preview', output_frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('3'):
                debug_mode = not debug_mode
                print(f"[DEBUG] Modo debug: {'ON' if debug_mode else 'OFF'}")
            elif key == ord('8'):
                collecting = True
                collected_face.clear()
                collected_speech.clear()
                collected_agg.clear()
                print("[INFO] Coleta iniciada! (pressione 9 para finalizar)")
            elif key == ord('9'):
                collecting = False
                print("[INFO] Coleta finalizada!")
                resumo_emocoes(collected_face, "Face")
                resumo_emocoes(collected_speech, "Fala")
                resumo_emocoes(collected_agg, "Agregado")
                salvar_csv(collected_face, collected_speech, collected_agg)

            if saveframe:
                ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S%f")
                cv2.imwrite(f"frame_{ts}.png", output_frame)

    finally:
        stop_speech_thread()
        video_stream.release()
        cv2.destroyAllWindows()
        if cam:
            cam.close()
        print("Aplicação finalizada.")

if __name__ == '__main__':
    main()
