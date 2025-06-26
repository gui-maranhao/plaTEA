import speech_recognition as sr
from transformers import pipeline
import threading
import queue
import time
import collections

recognizer = sr.Recognizer()

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
    with sr.Microphone() as source:
        recognizer.adjust_for_ambient_noise(source) 
        print("[SPEECH] Microfone ajustado para ruído ambiente. Ouvindo...")

        while not stop_speech_event.is_set():
            try:
                audio = recognizer.listen(source, timeout=5, phrase_time_limit=10) 
                
                text = recognizer.recognize_google(audio, language="en-US")
                print(f"[SPEECH] Você disse: {text}")
                
                speech_buffer.append(text)

                predictions = text_emotion_classifier(text)[0]
                sorted_preds = sorted(predictions, key=lambda x: x['score'], reverse=True)
                top_emotion = sorted_preds[0]
                
                speech_emotion_queue.put({
                    "timestamp": time.time(),
                    "dominant_emotion": top_emotion['label'],
                    "probabilities": {p['label']: p['score'] for p in predictions}
                })

            except sr.WaitTimeoutError:
                # print("[SPEECH] Sem fala detectada por um tempo.")
                pass # Nenhuma fala detectada, continua o loop
            except sr.UnknownValueError:
                print("[SPEECH] Não foi possível entender o áudio.")
            except sr.RequestError as e:
                print(f"[SPEECH] Erro no serviço de reconhecimento: {e}")
            except Exception as e:
                print(f"[SPEECH] Erro inesperado na thread de fala: {e}")
                break # Sai do loop em caso de erro grave

    print("[SPEECH] Thread de reconhecimento de fala finalizada.")

# --- Funções para Gerenciamento ---
def start_speech_thread():
    """Inicia a thread de reconhecimento de fala."""
    global speech_thread
    stop_speech_event.clear() # Garante que o evento de parada esteja limpo
    speech_thread = threading.Thread(target=speech_recognition_thread_func)
    speech_thread.start()

def stop_speech_thread():
    """Sinaliza a thread de reconhecimento de fala para parar."""
    print("[SPEECH] Sinalizando thread de fala para parar...")
    stop_speech_event.set()
    if 'speech_thread' in globals() and speech_thread.is_alive():
        speech_thread.join(timeout=5) # Espera a thread terminar
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