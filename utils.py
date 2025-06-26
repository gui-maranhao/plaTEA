'''
mediapipe_utils.py
Created on 2023 02 22 12:28:05
Description: File containing utils methods for processing mediapipe faces and processing FER

Author: Will <wlc2@cin.ufpe.br>
'''
from typing import Union, Tuple
import math
import mediapipe as mp
from torchvision import transforms
import torch
import numpy as np
from PIL import Image
import cv2 # Adicionado para uso do cv2 em casos onde a imagem pode ser processada


# Inicializa o detector de face do MediaPipe para bounding boxes
mp_face_detection = mp.solutions.face_detection

# Inicializa o detector de Face Mesh do MediaPipe para landmarks
mp_face_mesh = mp.solutions.face_mesh
# Usamos max_num_faces=1 para focar em uma única pessoa, o que simplifica a lógica
face_mesh_detector = mp_face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=1, # Detecta no máximo 1 face para simplificar a lógica de boca
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# Estes são alguns dos índices de landmarks para a boca no MediaPipe Face Mesh.
# Lista de índices de landmarks da boca (lábios externos e internos)
# Fonte para referências visuais: https://github.com/ManuelTS/augmentedFaceMeshIndices/blob/master/MediaPipe_Face_Mesh_Simplified_Keypoint_Mapping.pdf
LIP_LANDMARKS_INDICES = [
    # Lábios externos
    61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, # pontos superiores do lábio
    409, 271, 137, 150, 153, 144, 376, 303, 302, 301, 300, 299, # pontos inferiores do lábio
    # Alguns pontos internos para ajudar no bounding box, se necessário
    78, 191, 80, 81, 82, 13, 312, 311, 310, 415, 309
]


def _normalized_to_pixel_coordinates(
        normalized_x: float, normalized_y: float, image_width: int,
        image_height: int) -> Union[None, Tuple[int, int]]:
    """Converts normalized value pair to pixel coordinates."""

    # Checks if the float value is between 0 and 1.
    def is_valid_normalized_value(value: float) -> bool:
        return (value > 0 or math.isclose(0, value)) and (value < 1 or
                                                          math.isclose(1, value))

    if not (is_valid_normalized_value(normalized_x) and
            is_valid_normalized_value(normalized_y)):
        return None
    x_px = min(math.floor(normalized_x * image_width), image_width - 1)
    y_px = min(math.floor(normalized_y * image_height), image_height - 1)
    return x_px, y_px


def get_face_locations_mediapipe(image):
    image_rows, image_cols, _ = image.shape
    face_locs = []
    mouth_locs = []

    # Processa a imagem com o Face Detection para a bounding box da face
    with mp_face_detection.FaceDetection(
            model_selection=1, min_detection_confidence=0.5) as face_detection:
        results_face_detection = face_detection.process(image)
    
    # Processa a imagem com o Face Mesh para os landmarks da boca
    results_face_mesh = face_mesh_detector.process(image) # Usa o detector globalmente instanciado

    # --- Processamento da Detecção da Face ---
    if results_face_detection.detections:
        for dets in results_face_detection.detections:
            relative_bounding_box = dets.location_data.relative_bounding_box
            rect_start_point = _normalized_to_pixel_coordinates(
                relative_bounding_box.xmin, relative_bounding_box.ymin, image_cols,
                image_rows)
            rect_end_point = _normalized_to_pixel_coordinates(
                relative_bounding_box.xmin + relative_bounding_box.width,
                relative_bounding_box.ymin + relative_bounding_box.height, image_cols,
                image_rows)
            if rect_start_point and rect_end_point: # Garante que os pontos são válidos
                face_locs.append([rect_start_point, rect_end_point])

    # --- Processamento da Detecção da Boca (apenas para a primeira face) ---
    if results_face_mesh.multi_face_landmarks:
        # Foca na primeira face detectada pelo Face Mesh
        face_landmarks = results_face_mesh.multi_face_landmarks[0]
        
        mouth_points_coords = []
        for idx in LIP_LANDMARKS_INDICES:
            # Verifica se o índice existe para evitar erros
            if idx < len(face_landmarks.landmark):
                lm = face_landmarks.landmark[idx]
                x_px, y_px = _normalized_to_pixel_coordinates(lm.x, lm.y, image_cols, image_rows)
                if x_px is not None and y_px is not None:
                    mouth_points_coords.append((x_px, y_px))
        
        if mouth_points_coords:
            # Calcula o bounding box mínimo e máximo para os pontos da boca
            min_x = min([p[0] for p in mouth_points_coords])
            max_x = max([p[0] for p in mouth_points_coords])
            min_y = min([p[1] for p in mouth_points_coords])
            max_y = max([p[1] for p in mouth_points_coords])

            # Adiciona uma pequena margem (padding) para o retângulo da boca
            padding_x = int((max_x - min_x) * 0.1) # 10% de padding horizontal
            padding_y = int((max_y - min_y) * 0.2) # 20% de padding vertical (boca é mais larga que alta)

            mouth_x1 = max(0, min_x - padding_x)
            mouth_y1 = max(0, min_y - padding_y)
            mouth_x2 = min(image_cols, max_x + padding_x)
            mouth_y2 = min(image_rows, max_y + padding_y)

            mouth_locs.append(((mouth_x1, mouth_y1), (mouth_x2, mouth_y2)))

    # Retorna ambas as listas: localizações das faces e localizações das bocas
    return face_locs, mouth_locs


class EmotionPerceiver:
    def __init__(self, device='cpu'):

        self.device = device
        self.idx_to_class = {
            0: 'Anger', 1: 'Contempt', 2: 'Disgust', 3: 'Fear', 4: 'Happiness', 5: 'Neutral', 6: 'Sadness', 7: 'Surprise'
        }
        self.img_size = 224
        self.test_transforms = transforms.Compose(
            [
                transforms.Resize((self.img_size, self.img_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
            ]
        )
        try:
            model = torch.load('enet_b0_8_best_afew.pt', map_location=torch.device(device)) # Adicionado map_location
        except FileNotFoundError:
            print("[ERRO] Modelo 'enet_b0_8_best_afew.pt' não encontrado. Certifique-se de que está no diretório correto.")
            # Você pode sair ou carregar um modelo dummy para continuar o desenvolvimento
            exit()
        except Exception as e:
            print(f"[ERRO] Falha ao carregar o modelo PyTorch: {e}")
            exit()


        if isinstance(model.classifier, torch.nn.Sequential):
            self.classifier_weights = model.classifier[0].weight.cpu().data.numpy()
            self.classifier_bias = model.classifier[0].bias.cpu().data.numpy()
        else:
            self.classifier_weights = model.classifier.weight.cpu().data.numpy()
            self.classifier_bias = model.classifier.bias.cpu().data.numpy()
        
        model.classifier = torch.nn.Identity()
        model = model.to(self.device)
        self.model = model.eval()

    def get_probab(self, features):
        x = np.dot(features, np.transpose(
            self.classifier_weights))+self.classifier_bias
        return x

    def extract_features(self, face_img):
        # Converte para PIL Image para que transforms funcione
        img_tensor = self.test_transforms(Image.fromarray(face_img))
        img_tensor.unsqueeze_(0)
        features = self.model(img_tensor.to(self.device))
        features = features.data.cpu().numpy()
        return features

    def predict_emotions(self, face_img, logits=True):
        # Verifica se a imagem da face não está vazia
        if face_img.shape[0] == 0 or face_img.shape[1] == 0:
            # Retorna uma emoção padrão e scores vazios se a imagem for inválida
            return "Neutral", {label: 0.0 for label in self.idx_to_class.values()}
            
        features = self.extract_features(face_img)
        scores_raw = self.get_probab(features)[0] # scores_raw para não confundir com scores finais

        if not logits:
            e_x = np.exp(scores_raw - np.max(scores_raw)) # Subtrai o máximo para estabilidade numérica
            scores = e_x / e_x.sum()
        else:
            scores = scores_raw # Se logits, retorna os scores brutos

        pred_idx = np.argmax(scores_raw) # Usa os scores brutos para a predição
        
        # Converte scores para um dicionário, se for o caso
        scores_dict = {self.idx_to_class[i]: score for i, score in enumerate(scores)}
        
        return self.idx_to_class[pred_idx], scores_dict

    def extract_multi_features(self, face_img_list):
        imgs = [self.test_transforms(Image.fromarray(face_img))
                for face_img in face_img_list]
        features = self.model(torch.stack(imgs, dim=0).to(self.device))
        features = features.data.cpu().numpy()
        return features

    def predict_multi_emotions(self, face_img_list, logits=True):
        features = self.extract_multi_features(face_img_list)
        scores_raw = self.get_probab(features)
        preds = np.argmax(scores_raw, axis=1) # Predições baseadas nos scores brutos

        if not logits:
            e_x = np.exp(scores_raw - np.max(scores_raw, axis=1)[:, np.newaxis])
            scores = e_x / e_x.sum(axis=1)[:, None]
        else:
            scores = scores_raw

        # Retorna uma lista de emoções e os scores correspondentes
        return [self.idx_to_class[pred] for pred in preds], scores