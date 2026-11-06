import csv
import os
import statistics

class TelemetryCollector:
    def __init__(self, log_filepath: str = "benchmark_telemetry.csv"):
        """Initializes benchmarking storage and CSV writers[cite: 5]."""
        self.log_filepath = log_filepath
        self.data = []
        
        if not os.path.exists(self.log_filepath):
            with open(self.log_filepath, mode='w', newline='') as file:
                writer = csv.writer(file)
                writer.writerow([
                    "phrase_id", "smoothed_sentence", "t_capture", 
                    "t_infer", "t_smooth", "t_synth", "t_playback",
                    "L_inf_ms", "L_tts_ms", "L_post_ms"
                ])

    def record_event(self, phrase_id: str, timestamps: dict, text: str) -> None:
        """Calculates latency deltas and writes structured rows to disk[cite: 5]."""
        # Calculate derived metrics in milliseconds[cite: 5]
        l_inf = (timestamps.get("t_infer", 0) - timestamps.get("t_capture", 0)) * 1000
        l_tts = (timestamps.get("t_playback", 0) - timestamps.get("t_smooth", 0)) * 1000
        l_post = (timestamps.get("t_playback", 0) - timestamps.get("t_capture", 0)) * 1000

        row = [
            phrase_id, text,
            timestamps.get("t_capture", 0), timestamps.get("t_infer", 0),
            timestamps.get("t_smooth", 0), timestamps.get("t_synth", 0),
            timestamps.get("t_playback", 0),
            round(l_inf, 2), round(l_tts, 2), round(l_post, 2)
        ]
        self.data.append(row)

        with open(self.log_filepath, mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(row)

    def generate_summary_statistics(self) -> dict:
        """Computes mean latencies for immediate inclusion in the IEEE paper results table[cite: 5]."""
        if not self.data:
            return {}
        
        l_infs = [row[7] for row in self.data]
        l_posts = [row[9] for row in self.data]

        return {
            "avg_inference_ms": round(statistics.mean(l_infs), 2),
            "avg_post_latency_ms": round(statistics.mean(l_posts), 2),
            "max_post_latency_ms": round(max(l_posts), 2)
        }