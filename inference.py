import os
import numpy as np
import cv2
import torch
from utils import EmotionPerceiver, get_face_locations_mediapipe


def main():
    video_stream = cv2.VideoCapture('input.mp4')
    fer_pipeline = EmotionPerceiver(device='cuda')
    fourcc = cv2.VideoWriter_fourcc('M', 'J', 'P', 'G')
    out = cv2.VideoWriter('output.mp4', fourcc, 30.0, (640,  480))

    while True:
        (grabbed, frame) = video_stream.read()
        if not grabbed:
            print(f'[INFO] Stopping execution due to lack of frames.')
            break

        # Frame é a imagem capturada pela câmera
        frame = cv2.resize(frame, (1280, 720))
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        face_locations = get_face_locations_mediapipe(frame_rgb)

        if face_locations is None:
            continue

        for face_location in face_locations:
            (x1, y1), (x2, y2) = face_location  # Cada face detectada
            face = frame_rgb[y1:y2, x1:x2, :]

            emotion, scores = fer_pipeline.predict_emotions(
                face, logits=True)  # Cada emoção detectada
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 3)
            cv2.putText(frame, emotion, (x1, y1-5),
                        cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 2)

        cv2.imshow('Frame', frame)
        out.write(frame)
        cv2.waitKey(1)
    video_stream.release()
    out.release()


if __name__ == '__main__':
    main()
