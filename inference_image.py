import cv2
import torch
from utils import EmotionPerceiver, get_face_locations_mediapipe

################ Meet selenium Integration ######################################
from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException, NoSuchElementException
import time

MEET_SCREENSHOT_PATH = 'meet_grid_view.png'
##################################################################################

################ Meet Selenium Integration ########################################
def options():
    """Configura as opções do Chrome para a sessão do Selenium."""
    opt = Options()
    opt.add_argument("--disable-infobars")
    opt.add_argument("start-maximized")
    opt.add_argument("--disable-extensions")
    # Passa o argumento 1 para permitir e 2 para bloquear
    opt.add_experimental_option("prefs", {
        "profile.default_content_setting_values.media_stream_mic": 2, 
        "profile.default_content_setting_values.media_stream_camera": 2,
        "profile.default_content_setting_values.geolocation": 2, 
        "profile.default_content_setting_values.notifications": 2 
    })
    # Opções para evitar a detecção de automação pelo Google
    opt.add_argument("--disable-blink-features=AutomationControlled")
    opt.add_experimental_option("excludeSwitches", ["enable-automation"])
    opt.add_experimental_option('useAutomationExtension', False)
    return opt

def screenshot(driver):
    """
    Ajusta o layout do Google Meet para Mosaico com o máximo de participantes,
    garante que nenhuma tela esteja fixada e tira uma captura de tela.
    """
    driver.save_screenshot(MEET_SCREENSHOT_PATH)
    print(f"Captura de tela salva com sucesso em: {MEET_SCREENSHOT_PATH}")

def waiting():
    """Espera pela confirmação do usuário para prosseguir."""
    x = input("Já está na sala? (y/n): ")
    while x.lower()!= "y":
        x = input("Já está na sala? (y/n): ")

###############################################################################

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

def main():
    CAMINHO_DA_IMAGEM = MEET_SCREENSHOT_PATH
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

if __name__ == '__main__':
    link = "https://meet.google.com/trm-pwki-kwy"
    driver = webdriver.Chrome(options=options())
    driver.get(link)

    waiting()

    # Chama a nova função para ajustar o layout e tirar a foto
    while True:
        screenshot(driver)
        main()
        time.sleep(60)
