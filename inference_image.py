import cv2
import torch
from utils import EmotionPerceiver, get_face_locations_mediapipe

def analisar_emocoes_em_imagem(caminho_imagem):
    if torch.backends.mps.is_available():
        dev = 'mps'
        print("Dispositivo de processamento: Apple GPU (MPS)")
    elif torch.cuda.is_available():
        dev = 'cuda'
        print("Dispositivo de processamento: NVIDIA GPU (CUDA)")
    else:
        dev = 'cpu'
        print("Dispositivo de processamento: CPU")

    fer_pipeline = EmotionPerceiver(device=dev)
    
    imagem = cv2.imread(caminho_imagem)
    if imagem is None:
        print(f"[ERRO] Não foi possível carregar a imagem em: {caminho_imagem}")
        return []
    
    imagem_rgb = cv2.cvtColor(imagem, cv2.COLOR_BGR2RGB)
    face_locations, _ = get_face_locations_mediapipe(imagem_rgb)
    
    resultados = []
    
    if face_locations:
        print(f"[INFO] Encontrada(s) {len(face_locations)} face(s). Processando...")
        for face_location in face_locations:
            (x1, y1), (x2, y2) = face_location
            face_roi = imagem_rgb[y1:y2, x1:x2]
            if face_roi.size > 0:
                emocao, _ = fer_pipeline.predict_emotions(face_roi, logits=True)
                resultados.append((face_location, emocao))
    else:
        print("[INFO] Nenhuma face foi detectada na imagem.")
        
    return resultados

if __name__ == '__main__':
    CAMINHO_DA_IMAGEM = "input_test.jpeg"
    dados_das_emocoes = analisar_emocoes_em_imagem(CAMINHO_DA_IMAGEM)

    print("\n--- RESULTADO FINAL ---")
    print(dados_das_emocoes)

    if dados_das_emocoes:
        imagem_visualizacao = cv2.imread(CAMINHO_DA_IMAGEM)
        for bbox, label in dados_das_emocoes:
            (x1, y1), (x2, y2) = bbox
            cv2.rectangle(imagem_visualizacao, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(imagem_visualizacao, label, (x1, y1 - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        
        cv2.imwrite("resultado_emocoes.jpg", imagem_visualizacao)
        print("\n[INFO] Imagem com os resultados foi salva como 'resultado_emocoes.jpg'")
        # Para exibir em uma janela, descomente as linhas abaixo:
        # cv2.imshow("Resultados", imagem_visualizacao)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()