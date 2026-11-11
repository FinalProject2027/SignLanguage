import base64
import threading
import time

import streamlit as st
import cv2
import mediapipe as mp
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.core.base_options import BaseOptions
from mediapipe.tasks.python.vision.core.vision_task_running_mode import (
    VisionTaskRunningMode,
)
from src.backend.realtime_pipeline import RealtimePipeline

if __package__:
    from .live_camera import (
        DEFAULT_BUFFER_SIZE,
        MAX_BUFFER_SIZE,
        MODEL_PATH,
        LandmarkBuffer,
        extract_frame_landmarks,
    )
else:
    from live_camera import (
        DEFAULT_BUFFER_SIZE,
        MAX_BUFFER_SIZE,
        MODEL_PATH,
        LandmarkBuffer,
        extract_frame_landmarks,
    )


class LiveCameraStream:
    def __init__(self, capture, detector, buffer_capacity, predict_fn=None):
        self.capture = capture
        self.detector = detector
        self.landmark_buffer = LandmarkBuffer(buffer_capacity)
        self.pipeline = RealtimePipeline(predict_fn, buffer_size=buffer_capacity)
        self.stop_event = threading.Event()
        self.lock = threading.Lock()
        self.latest_frame = None
        self.latest_image = None
        self.latest_prediction = None
        self.processing_times = (0.0, 0.0, 0.0)
        self.error = None
        self.thread = threading.Thread(target=self.process_frames, daemon=True)

    def start(self):
        self.thread.start()

    def stop(self):
        self.stop_event.set()
        if self.thread.is_alive():
            self.thread.join()

    def process_frames(self):
        try:
            while not self.stop_event.is_set():
                success, frame = self.capture.read()
                if not success:
                    if not self.stop_event.is_set():
                        self.error = "Could not read a frame from the webcam."
                    break

                started = time.perf_counter()
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                landmarks = extract_frame_landmarks(
                    frame,
                    self.detector,
                    rgb_frame,
                )
                landmark_time_ms = (time.perf_counter() - started) * 1000

                event = self.pipeline.push(landmarks)
                started = time.perf_counter()
                success, encoded_frame = cv2.imencode(".jpg", frame)
                jpeg_time_ms = (time.perf_counter() - started) * 1000
                if not success:
                    raise RuntimeError("Could not encode the webcam frame.")

                started = time.perf_counter()
                image_data = base64.b64encode(encoded_frame).decode("ascii")
                base64_time_ms = (time.perf_counter() - started) * 1000

                with self.lock:
                    self.landmark_buffer.append(landmarks)
                    self.latest_frame = rgb_frame
                    self.latest_image = image_data
                    if event is not None:
                        self.latest_prediction = event["text"]
                    self.processing_times = (
                        landmark_time_ms,
                        jpeg_time_ms,
                        base64_time_ms,
                    )
        except Exception as error:
            self.error = str(error)
        finally:
            try:
                self.capture.release()
            finally:
                self.detector.close()


def release_resources(capture, detector):
    try:
        if capture is not None:
            capture.release()
    finally:
        if detector is not None:
            detector.close()


def stop_camera():
    stream = st.session_state.camera_stream
    try:
        if stream is not None:
            stream.stop()
    except Exception as error:
        st.session_state.camera_error = (
            f"Error while releasing camera resources: {error}"
        )
    finally:
        st.session_state.camera_stream = None
        st.session_state.camera_running = False


def start_camera(predict_fn=None):
    capture = None
    detector = None
    try:
        if not MODEL_PATH.is_file():
            raise FileNotFoundError(f"MediaPipe model not found: {MODEL_PATH}")

        capture = cv2.VideoCapture(0)
        if not capture.isOpened():
            raise RuntimeError("Could not open webcam at camera index 0.")

        options = vision.HolisticLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(MODEL_PATH)),
            running_mode=VisionTaskRunningMode.IMAGE,
        )
        detector = vision.HolisticLandmarker.create_from_options(options)
        stream = LiveCameraStream(
            capture,
            detector,
            st.session_state.buffer_capacity,
            predict_fn,
        )
        stream.start()
        st.session_state.camera_stream = stream
        st.session_state.camera_running = True
        st.session_state.camera_error = None
    except Exception as error:
        try:
            release_resources(capture, detector)
        finally:
            st.session_state.camera_stream = None
            st.session_state.camera_running = False
            st.session_state.camera_error = (
                f"Could not start the camera: {error}"
            )


st.set_page_config(page_title="Live Sign Language Camera")
st.title("Live Sign Language Camera")
st.write("Capture webcam frames and build a synchronized landmark window.")

if "camera_running" not in st.session_state:
    st.session_state.camera_running = False
    st.session_state.camera_stream = None
    st.session_state.camera_error = None

if "last_displayed_frame" not in st.session_state:
    st.session_state.last_displayed_frame = None
    st.session_state.last_displayed_image = None

buffer_capacity = st.slider(
    "Landmark window size",
    min_value=1,
    max_value=MAX_BUFFER_SIZE,
    value=DEFAULT_BUFFER_SIZE,
    disabled=st.session_state.camera_running,
    key="buffer_capacity",
)
start_button, stop_button = st.columns(2)
start_button.button(
    "Start Camera",
    disabled=st.session_state.camera_running,
    on_click=start_camera,
)
stop_button.button(
    "Stop Camera",
    disabled=not st.session_state.camera_running,
    on_click=stop_camera,
)

if st.session_state.camera_error:
    st.error(st.session_state.camera_error)
    st.session_state.camera_error = None


@st.fragment(run_every=0.3)
def show_live_camera():
    if not st.session_state.camera_running:
        st.session_state.last_displayed_frame = None
        st.session_state.last_displayed_image = None
        st.info("Select Start Camera to begin.")
        return

    stream = st.session_state.camera_stream
    with stream.lock:
        frame = stream.latest_frame
        image_data = stream.latest_image
        error = stream.error
        frame_count = len(stream.landmark_buffer)
        buffer_capacity = stream.landmark_buffer.frames.maxlen
        prediction = stream.latest_prediction
        landmark_time_ms, jpeg_time_ms, base64_time_ms = stream.processing_times

    if error or not stream.thread.is_alive():
        st.session_state.camera_error = (
            f"Live camera processing failed: {error}"
            if error
            else "Camera has stopped."
        )
        stop_camera()
        st.rerun()

    if frame is not None and image_data is not None:
        if stream.pipeline.predict_fn is None:
            st.info(
                "Predictions disabled: no trained checkpoint or "
                "predict_fn is configured."
            )
        elif prediction is not None:
            st.write(f"Prediction: {prediction}")

        if frame is not st.session_state.last_displayed_frame:
            st.session_state.last_displayed_frame = frame
            st.session_state.last_displayed_image = image_data
        image_data = st.session_state.last_displayed_image
        st.html(
            f'<img src="data:image/jpeg;base64,{image_data}" '
            'alt="Live camera feed" style="width: 100%; height: auto;">'
        )
        st.caption(
            f"Buffered frames: {frame_count}/{buffer_capacity} | "
            f"Landmark extraction: {landmark_time_ms:.2f} ms | "
            f"JPEG encoding: {jpeg_time_ms:.2f} ms | "
            f"Base64 conversion: {base64_time_ms:.2f} ms"
        )


show_live_camera()
