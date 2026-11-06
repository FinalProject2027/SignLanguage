import pyttsx3
import threading
import queue
import time

class AsynchronousTTSEngine:
    def __init__(self, voice_speed: int = 175, volume: float = 1.0):
        """Initializes the thread-safe queue and starts the daemon audio worker[cite: 5]."""
        self.queue = queue.Queue()
        self.voice_speed = voice_speed
        self.volume = volume
        self.is_running = True
        
        self.worker_thread = threading.Thread(target=self._tts_worker_loop, daemon=True)
        
    def start_worker(self) -> None:
        """Launches the background daemon thread[cite: 5]."""
        self.worker_thread.start()

    def _tts_worker_loop(self) -> None:
        """Continuous consumer loop that plays sound without blocking the main thread[cite: 5]."""
        # Initialize pyttsx3 completely inside the thread for Windows COM safety[cite: 5]
        engine = pyttsx3.init()
        engine.setProperty('rate', self.voice_speed)
        engine.setProperty('volume', self.volume)
        
        while self.is_running:
            try:
                # Block for up to 0.5s waiting for text. Allows graceful shutdown.
                text, metadata = self.queue.get(timeout=0.5)
                
                engine.say(text)
                engine.runAndWait()
                
                self.queue.task_done()
            except queue.Empty:
                continue

    def speak_async(self, text: str, metadata: dict = None) -> None:
        """Non-blocking public method to enqueue text for synthesis[cite: 5]."""
        if text:
            # Provide an emergency flush to prevent stale speech output[cite: 5]
            with self.queue.mutex:
                self.queue.queue.clear()
            self.queue.put((text, metadata or {}))

    def stop_worker(self) -> None:
        """Gracefully halts worker thread[cite: 5]."""
        self.is_running = False
        if self.worker_thread.is_alive():
            self.worker_thread.join(timeout=2.0)