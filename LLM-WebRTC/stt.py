import speech_recognition as sr
import logging
import sys
import time
import threading

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger("STTProcessor")

class STTProcessor:
    @staticmethod
    def list_audio_devices():
        """Print and return the list of available microphones."""
        logger.info("\n" + "="*50)
        logger.info("🎙️ MICROPHONE SCANNER")
        logger.info("="*50)
        mic_list = sr.Microphone.list_microphone_names()
        for i, name in enumerate(mic_list):
            logger.info(f"   [{i}] {name}")
        logger.info("="*50)
        return mic_list

    def __init__(self, mic_index=None, language="es-ES", wake_words=None):
        self.recognizer = sr.Recognizer()
        self.recognizer.energy_threshold = 400 
        self.recognizer.dynamic_energy_threshold = True
        self.recognizer.pause_threshold = 0.8 
        
        self.mic_index = mic_index
        self.language = language
        self.wake_words = wake_words if wake_words else ["jonay", "yonay", "jonai", "yonai", "honay", "jhonay", "unai", "jona", "jonah"]
        
    def calibrate_mic(self, source):
        logger.info("\n🎤 [AUDIO] Calibrating environment (stay silent for 2 seconds)...")
        self.recognizer.adjust_for_ambient_noise(source, duration=2.0)
        logger.info(f"🎤 [AUDIO] Calibration ready (Threshold: {self.recognizer.energy_threshold:.0f})")

    def listen_and_transcribe(self, source):
        def animate():
            chars = ['⠋', '⠙', '⠹', '⠸', '⠼', '⠴', '⠦', '⠧', '⠇', '⠏']
            i = 0
            while getattr(self, "is_listening", False):
                sys.stdout.write(f'\r   🔴 RECORDING... {chars[i]}')
                sys.stdout.flush()
                time.sleep(0.1)
                i = (i + 1) % len(chars)

        self.is_listening = True
        anim_thread = threading.Thread(target=animate)
        anim_thread.start()
        
        try:
            audio = self.recognizer.listen(source, timeout=1.5, phrase_time_limit=8)
            self.is_listening = False
            
            anim_thread.join()
            sys.stdout.write('\r' + ' ' * 60 + '\r')
            sys.stdout.write('   ⏳ Transcribing...\n')
            sys.stdout.flush()
            
            text = self.recognizer.recognize_google(audio, language=self.language)
            return text.lower()
            
        except sr.UnknownValueError:
            self.is_listening = False
            anim_thread.join()
            sys.stdout.write('\r' + ' ' * 60 + '\r')
            return ""
        
        except sr.WaitTimeoutError:
            self.is_listening = False
            anim_thread.join()
            sys.stdout.write('\r' + ' ' * 60 + '\r')
            return ""
        
        except Exception as e:
            self.is_listening = False
            anim_thread.join()
            sys.stdout.write('\r' + ' ' * 60 + '\r')
            logger.error(f"   ❌ Microphone Error: {e}")
            return ""
            
    def is_wake_word(self, text):
        return any(wake_word in text for wake_word in self.wake_words)