import time
import json
import logging
import threading
import queue
import speech_recognition as sr

from dogbot.webrtc_connection import WebRTCConnection
from dogbot.go2_motion import Go2Motion, SportActions

from llm_brain import ALLOWED_ACTIONS, LLMGo2Brain
from stt import STTProcessor
from metrics_tracker import MetricsTracker

logging.basicConfig(level=logging.ERROR, format='%(message)s', force=True)
logger = logging.getLogger("Go2-Threaded")
logger.setLevel(logging.INFO)
logger.propagate = False

logging.getLogger("STTProcessor").setLevel(logging.INFO)

if not logger.handlers:
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(logging.Formatter('%(message)s'))
    logger.addHandler(console_handler)

class Go2ThreadedController:
    def __init__(self, host="192.168.1.168", mic_index=0):
        self.host = host
        self.mic_index = mic_index
        self.brain = LLMGo2Brain()
        self.stt_processor = STTProcessor(mic_index=self.mic_index)
        
        self.metrics = MetricsTracker()
        self.text_queue = queue.Queue()
        self.action_queue = queue.Queue()
        self.stop_event = threading.Event()

    def voice_listener_worker(self):
        """Thread 1: STT with passive safety metrics."""
        try:
            with sr.Microphone(device_index=self.mic_index) as source:
                
                self.stt_processor.calibrate_mic(source)
                
                while not self.stop_event.is_set():
                    try:
                        t0_stt = time.time()
                        
                        text = self.stt_processor.listen_and_transcribe(source)
                        
                        if text:
                            stt_latency = time.time() - t0_stt
                            if self.stt_processor.is_wake_word(text):
                                logger.info(f"🎙️ '{text}' (Wake-word OK)")
                                self.text_queue.put((text, stt_latency))
                            else:
                                self.metrics.ignored_no_wake += 1
                                logger.info(f"   😴 Ignored: '{text}' (No wake-word)")
                    except Exception as e:
                        logger.error(f"❌ Error STT Loop: {e}")
                        time.sleep(1)
        except Exception as e:
            logger.error(f"❌ Failed to open microphone: {e}")

    def brain_worker(self):
        """Thread 2: LLM with structural metrics."""
        while not self.stop_event.is_set():
            try:
                text, stt_latency = self.text_queue.get(timeout=1)
                t0_llm = time.time()
                
                self.metrics.total_llm_calls += 1
                
                actions, description = self.brain.process(text)
                
                llm_latency = time.time() - t0_llm
                
                if actions is not None:
                    self.metrics.valid_json_count += 1
                    if not actions:
                        self.metrics.llm_no_action_decisions += 1
                    else:
                        self.action_queue.put((actions, text))
                else:
                    self.metrics.invalid_json_count += 1
                
                logger.info(f"📊 Latency: {stt_latency + llm_latency:.2f}s")
                self.text_queue.task_done()
            except queue.Empty:
                continue

    def run(self):
        """Main thread: execution and autocorrections."""
        logger.info(f"🔌 Connecting to {self.host}...")
        conn = WebRTCConnection(host=self.host)
        conn.connect()
        
        if not conn.is_connected:
            logger.error("❌ Connection failed.")
            return
            
        motion = Go2Motion(conn)
        logger.info("✅ Jonay ready.")

        threading.Thread(target=self.voice_listener_worker, daemon=True).start()
        threading.Thread(target=self.brain_worker, daemon=True).start()

        try:
            while not self.stop_event.is_set():
                try:
                    commands, raw_text = self.action_queue.get(timeout=0.5)
                    
                    self.metrics.session_history.append({"text": raw_text, "actions": commands})

                    for cmd in commands:
                        ctype = str(cmd.get("type", "")).lower()
                        
                        if ctype.upper() in ALLOWED_ACTIONS:
                            self.metrics.autocorrect_triggers += 1
                            cmd["value"] = ctype.upper()
                            ctype = "action"
                        
                        if ctype == "action":
                            val = str(cmd.get("value", "")).upper()
                            if val in ALLOWED_ACTIONS:
                                motion.execute_behavior(getattr(SportActions, val))
                                time.sleep(0.3)
                        
                        elif ctype == "move":
                            p = cmd.get("params", {})
                            dur = float(p.get("duration", 2.0))
                            
                            if dur < 1.0:
                                self.metrics.autocorrect_triggers += 1
                                dur = 1.0

                            motion.execute_behavior(SportActions.BALANCE_STAND)
                            time.sleep(0.5)
                            motion.move(x=float(p.get("x", 0)), 
                                        y=float(p.get("y", 0)), 
                                        yaw=float(p.get("yaw", 0)), 
                                        duration=dur)
                            time.sleep(dur + 0.3)
                            
                    self.action_queue.task_done()
                except queue.Empty:
                    continue
        except KeyboardInterrupt:
            self.stop_event.set()
        finally:
            self.metrics.print_summary()
            conn.disconnect()

if __name__ == "__main__":
    MIC_INDEX = None
    STTProcessor.list_audio_devices()
    if MIC_INDEX is None:
        sel = input("🎤 Microphone (Enter for 0): ")
        idx = int(sel) if sel.strip() else 0
    else:
        idx = MIC_INDEX
    Go2ThreadedController("192.168.1.2", mic_index=idx).run()