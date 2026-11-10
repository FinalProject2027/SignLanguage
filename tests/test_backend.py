import time
import sys
from pathlib import Path

# Add the project root to the path so it can find the src folder
sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.backend.processor import TextPostProcessor
from src.backend.tts_engine import AsynchronousTTSEngine
from src.backend.telemetry import TelemetryCollector

def run_integration_test():
    print("--- Starting Backend Integration Test ---")
    
    processor = TextPostProcessor()
    tts = AsynchronousTTSEngine(voice_speed=150)
    telemetry = TelemetryCollector("test_telemetry.csv")
    
    tts.start_worker()
    
    # Use perf_counter to sync with Member 4's boundary detector
    t_capture = time.perf_counter()
    
    # 1. Simulate Member 2's model output
    raw_output = "[SOS] THANK_YOU THANK_YOU PLEASE [EOS]"
    t_infer = time.perf_counter()
    
    # 2. Member 3 Processor smooths it
    smoothed = processor.process_sequence(raw_output)
    t_smooth = time.perf_counter()
    
    print(f"Raw Output: {raw_output}")
    print(f"Smoothed Text: {smoothed}")
    
    # 3. Member 3 TTS plays it asynchronously
    t_synth = time.perf_counter()
    tts.speak_async(smoothed)
    
    time.sleep(0.05) 
    t_playback = time.perf_counter()
    
    # 4. Log Telemetry
    timestamps = {
        "t_capture": t_capture,
        "t_infer": t_infer,
        "t_smooth": t_smooth,
        "t_synth": t_synth,
        "t_playback": t_playback
    }
    telemetry.record_event("test_01", timestamps, smoothed)
    
    print("Telemetry Logged. Awaiting audio completion...")
    time.sleep(2) 
    tts.stop_worker()
    
    stats = telemetry.generate_summary_statistics()
    print(f"Test Complete. Metrics: {stats}")

if __name__ == "__main__":
    run_integration_test()