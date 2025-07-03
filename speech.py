import speech_recognition as sr
from transformers import pipeline
import threading
import queue
import time
import collections
import json # Para lidar com a saída do Vosk

from vosk import Model, KaldiRecognizer
import pyaudio 

# Carrega o modelo Vosk (fora da função para carregar apenas uma vez)
# Certifique-se de que o caminho para o modelo Vosk está correto!
VOSK_MODEL_PATH = "vosk-model-small-en-us-0.15" # Atualize este caminho
try:
    vosk_model = Model(VOSK_MODEL_PATH)
except Exception as e:
    print(f"[ERRO] Falha ao carregar o modelo Vosk em {VOSK_MODEL_PATH}: {e}")
    print("Certifique-se de que baixou e descompactou o modelo Vosk corretamente.")
    # Se não puder carregar, o programa não deve continuar sem reconhecimento de fala
    exit() 

# --- Fim das Adições para Vosk ---

# Inicializa o reconhecedor (usaremos com Vosk agora)
# recognizer = sr.Recognizer() # Não será usado diretamente com stream do PyAudio

# Inicializa o classificador de emoções de texto globalmente
text_emotion_classifier = pipeline(
    "text-classification",
    model="j-hartmann/emotion-english-distilroberta-base",
    return_all_scores=True
)

speech_emotion_queue = queue.Queue()
stop_speech_event = threading.Event()
speech_buffer = collections.deque(maxlen=3) 

def speech_recognition_thread_func():
    print("[SPEECH] Thread de reconhecimento de fala iniciada.")

    # Configurações do PyAudio para stream
    CHUNK = 8192 # Tamanho do chunk de áudio para processar
    FORMAT = pyaudio.paInt16
    CHANNELS = 1
    RATE = 16000 # Vosk geralmente prefere 16kHz

    p = pyaudio.PyAudio()
    stream = p.open(format=FORMAT,
                    channels=CHANNELS,
                    rate=RATE,
                    input=True,
                    frames_per_buffer=CHUNK)

    # Inicializa o reconhecedor Vosk para a stream
    vosk_recognizer = KaldiRecognizer(vosk_model, RATE)

    print("[SPEECH] Microfone ajustado e ouvindo para reconhecimento offline...")

    while not stop_speech_event.is_set():
        try:
            data = stream.read(CHUNK, exception_on_overflow=False)
            # Verifica se há fala no chunk de áudio
            if vosk_recognizer.AcceptWaveform(data):
                result_json = vosk_recognizer.Result()
                result = json.loads(result_json)
                text = result.get('text', '').strip()

                if text: # Apenas se houver texto reconhecido
                    print(f"[SPEECH] Você disse: {text}")
                    speech_buffer.append(text)

                    # Classifica a emoção do texto
                    predictions = text_emotion_classifier(text)[0]
                    sorted_preds = sorted(predictions, key=lambda x: x['score'], reverse=True)
                    top_emotion = sorted_preds[0]

                    speech_emotion_queue.put({
                        "timestamp": time.time(),
                        "dominant_emotion": top_emotion['label'],
                        "probabilities": {p['label']: p['score'] for p in predictions}
                    })
            # else:
            #     # Se não houver fala aceita, pode pegar o partial result se desejar
            #     # partial_text = vosk_recognizer.PartialResult()
            #     # print(f"Partial: {json.loads(partial_text)['partial']}")
            #     pass

        except Exception as e:
            print(f"[SPEECH] Erro inesperado na thread de fala: {e}")
            break 

    # Garante que a stream é fechada ao sair do loop
    stream.stop_stream()
    stream.close()
    p.terminate()
    print("[SPEECH] Thread de reconhecimento de fala finalizada.")

# --- Funções para Gerenciamento (permanecem as mesmas) ---
def start_speech_thread():
    """Inicia a thread de reconhecimento de fala."""
    global speech_thread
    stop_speech_event.clear() 
    speech_thread = threading.Thread(target=speech_recognition_thread_func)
    speech_thread.start()

def stop_speech_thread():
    """Sinaliza a thread de reconhecimento de fala para parar."""
    print("[SPEECH] Sinalizando thread de fala para parar...")
    stop_speech_event.set()
    if 'speech_thread' in globals() and speech_thread.is_alive():
        speech_thread.join(timeout=5)
        if speech_thread.is_alive():
            print("[SPEECH] Aviso: Thread de fala pode não ter terminado graciosamente.")

def get_latest_speech_emotion():
    """Pega a última emoção de fala da fila (não bloqueante)."""
    latest_emotion = None
    while not speech_emotion_queue.empty():
        latest_emotion = speech_emotion_queue.get_nowait()
    return latest_emotion

def get_speech_buffer_text():
    """Retorna o texto atual no buffer de fala."""
    return " ".join(speech_buffer)

if __name__ == '__main__':
    start_speech_thread()
    try:
        while True:
            latest_speech_emo = get_latest_speech_emotion()
            if latest_speech_emo:
                print(f"[MAIN] Emoção da fala recebida: {latest_speech_emo['dominant_emotion']} em {latest_speech_emo['timestamp']:.2f}")
            time.sleep(1)
    except KeyboardInterrupt:
        print("[MAIN] Interrupção no programa principal.")
    finally:
        stop_speech_thread()
        print("[MAIN] Programa principal finalizado.")
