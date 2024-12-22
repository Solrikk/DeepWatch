import torch
import whisper
import tempfile
import os
import cv2
import numpy as np
import sys
import mediapipe as mp
import logging
from typing import List, Dict, Tuple
import pyaudio
import threading
import wave
from datetime import datetime

sys.path.append('yolov5')
from models.common import DetectMultiBackend
from utils.torch_utils import select_device
from utils.general import non_max_suppression
from utils.augmentations import letterbox

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

whisper_model = whisper.load_model("base")
device = select_device('cpu')
yolo_model = DetectMultiBackend('yolov5l.pt', device=device)
imgsz = [640, 640]

mp_pose = mp.solutions.pose
mp_face_detection = mp.solutions.face_detection
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

LEFT_ARM_CONNECTIONS = [
    (mp_pose.PoseLandmark.LEFT_SHOULDER, mp_pose.PoseLandmark.LEFT_ELBOW),
    (mp_pose.PoseLandmark.LEFT_ELBOW, mp_pose.PoseLandmark.LEFT_WRIST)
]
RIGHT_ARM_CONNECTIONS = [
    (mp_pose.PoseLandmark.RIGHT_SHOULDER, mp_pose.PoseLandmark.RIGHT_ELBOW),
    (mp_pose.PoseLandmark.RIGHT_ELBOW, mp_pose.PoseLandmark.RIGHT_WRIST)
]

def caesar_cipher_encrypt(text: str, shift: int = 3) -> str:
    encrypted = ""
    for char in text:
        if char.isupper():
            encrypted += chr((ord(char) + shift - 65) % 26 + 65)
        elif char.islower():
            encrypted += chr((ord(char) + shift - 97) % 26 + 97)
        elif char.isdigit():
            encrypted += chr((ord(char) + shift - 48) % 10 + 48)
        else:
            encrypted += char
    return encrypted

def overlay_text(frame: np.ndarray, text: str, position: Tuple[int, int], font_scale: float = 0.4, color: Tuple[int, int, int] = (255, 255, 255), thickness: int = 1) -> None:
    font = cv2.FONT_HERSHEY_SIMPLEX
    lines = text.split('\n')
    x, y = position
    line_height = 15
    for i, line in enumerate(lines):
        text_size, _ = cv2.getTextSize(line, font, font_scale, thickness)
        text_w, text_h = text_size
        overlay = frame.copy()
        cv2.rectangle(overlay, (x, y - text_h - 5 + i*line_height), (x + text_w + 5, y + 5 + i*line_height), (0, 0, 0), -1)
        alpha = 0.6
        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
        cv2.putText(frame, line, (x, y + i*line_height), font, font_scale, color, thickness, cv2.LINE_AA)

def transcribe_audio(audio_file_path: str) -> List[Dict]:
    result = whisper_model.transcribe(audio_file_path)
    segments = result.get('segments', [])
    return segments

def detect_and_capture_faces(frame: np.ndarray, person_boxes: List[Tuple[int, int, int, int]], face_detector) -> Tuple[np.ndarray, List[np.ndarray]]:
    face_images = []
    for (x1, y1, x2, y2) in person_boxes:
        person_roi = frame[y1:y2, x1:x2]
        if person_roi.size == 0:
            continue
        rgb_roi = cv2.cvtColor(person_roi, cv2.COLOR_BGR2RGB)
        results = face_detector.process(rgb_roi)
        if results.detections:
            for detection in results.detections:
                bboxC = detection.location_data.relative_bounding_box
                ih, iw, _ = person_roi.shape
                fx1 = int(bboxC.xmin * iw) + x1
                fy1 = int(bboxC.ymin * ih) + y1
                fx2 = int((bboxC.xmin + bboxC.width) * iw) + x1
                fy2 = int((bboxC.ymin + bboxC.height) * ih) + y1
                face_width = fx2 - fx1
                face_height = fy2 - fy1
                if face_width < 30 or face_height < 30:
                    continue
                fx1 = max(0, min(fx1, frame.shape[1] - 1))
                fy1 = max(0, min(fy1, frame.shape[0] - 1))
                fx2 = max(0, min(fx2, frame.shape[1] - 1))
                fy2 = max(0, min(fy2, frame.shape[0] - 1))
                if fx2 <= fx1 or fy2 <= fy1:
                    continue
                face_img = frame[fy1:fy2, fx1:fx2]
                if face_img.size == 0:
                    continue
                face_images.append(face_img)
    return frame, face_images

def detect_pose_on_frame_custom(frame: np.ndarray, results_pose) -> np.ndarray:
    if not results_pose or not results_pose.pose_landmarks:
        return frame
    h, w, _ = frame.shape
    landmarks = results_pose.pose_landmarks.landmark
    def get_coords(landmark):
        return int(landmark.x * w), int(landmark.y * h)
    for idx, lm in enumerate(landmarks):
        x, y = get_coords(lm)
        cv2.circle(frame, (x, y), 4, (200, 200, 200), -1)
    c = (100, 100, 100)
    for conn in mp_pose.POSE_CONNECTIONS:
        if conn in LEFT_ARM_CONNECTIONS or conn in RIGHT_ARM_CONNECTIONS:
            continue
        s, e = conn
        start = landmarks[s]
        end = landmarks[e]
        x1, y1 = get_coords(start)
        x2, y2 = get_coords(end)
        cv2.line(frame, (x1, y1), (x2, y2), c, 2)
    c1 = (255, 50, 50)
    c2 = (50, 255, 50)
    t = 5
    for (s, e) in LEFT_ARM_CONNECTIONS:
        start = landmarks[s]
        end = landmarks[e]
        x1, y1 = get_coords(start)
        x2, y2 = get_coords(end)
        cv2.line(frame, (x1, y1), (x2, y2), c1, t)
    for (s, e) in RIGHT_ARM_CONNECTIONS:
        start = landmarks[s]
        end = landmarks[e]
        x1, y1 = get_coords(start)
        x2, y2 = get_coords(end)
        cv2.line(frame, (x1, y1), (x2, y2), c2, t)
    return frame

def draw_hands_on_frame(frame: np.ndarray, results_hands) -> np.ndarray:
    if results_hands.multi_hand_landmarks:
        for hand_landmarks in results_hands.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS, mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2, circle_radius=3), mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2))
    return frame

def add_subtitles(frame: np.ndarray, segments: List[Dict], current_time_ms: float) -> None:
    subtitle_text = None
    for segment in segments:
        start_time = segment['start'] * 1000
        end_time = segment['end'] * 1000
        text = segment['text']
        if start_time <= current_time_ms <= end_time:
            subtitle_text = text.strip()
            break
    if subtitle_text:
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 1
        thickness = 2
        (text_w, text_h), _ = cv2.getTextSize(subtitle_text, font, font_scale, thickness)
        x = 50
        y = frame.shape[0] - 50
        overlay = frame.copy()
        cv2.rectangle(overlay, (x - 10, y - text_h - 10), (x + text_w + 10, y + 10), (0, 0, 0), -1)
        alpha = 0.5
        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
        cv2.putText(frame, subtitle_text, (x, y), font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)

def save_faces(face_images: List[np.ndarray], output_dir: str, frame_count: int) -> None:
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    for i, face in enumerate(face_images):
        face_filename = os.path.join(output_dir, f"frame_{frame_count}_face_{i}.png")
        cv2.imwrite(face_filename, face)

def place_detected_faces(frame: np.ndarray, face_images: List[np.ndarray], width: int, height: int, side: str = 'right') -> None:
    for idx, face in enumerate(face_images):
        face = cv2.resize(face, (100, 100))
        if side == 'right':
            x_offset = width - 120
        else:
            x_offset = 20
        y_offset = 10 + (110 * idx)
        if y_offset + 100 <= height:
            frame[y_offset:y_offset + 100, x_offset:x_offset + 100] = face

def detect_objects_on_frame(frame: np.ndarray) -> List[Tuple[str, Tuple[int, int, int, int]]]:
    img = letterbox(frame, imgsz, stride=32, auto=True)[0]
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype(np.float32) / 255.0
    img = np.transpose(img, (2, 0, 1))
    img = np.expand_dims(img, 0)
    img = torch.from_numpy(img).to(device)
    pred = yolo_model(img)
    pred = non_max_suppression(pred)
    detections = []
    if pred and pred[0] is not None and len(pred[0]):
        h0, w0 = frame.shape[:2]
        h, w = imgsz
        gain = min(w / w0, h / h0)
        pad_w = (w - w0 * gain) / 2
        pad_h = (h - h0 * gain) / 2
        for *box, conf, cls in pred[0]:
            cls_name = yolo_model.names[int(cls)].lower()
            if cls_name == 'person':
                x1, y1, x2, y2 = box
                x1 = (x1 - pad_w) / gain
                y1 = (y1 - pad_h) / gain
                x2 = (x2 - pad_w) / gain
                y2 = (y2 - pad_h) / gain
                x1, y1, x2, y2 = map(int, (x1, y1, x2, y2))
                x1 = max(0, min(x1, w0 - 1))
                y1 = max(0, min(y1, h0 - 1))
                x2 = max(0, min(x2, w0 - 1))
                y2 = max(0, min(y2, h0 - 1))
                if x2 <= x1 or y2 <= y1:
                    continue
                detections.append((cls_name, (x1, y1, x2, y2)))
    return detections

def audio_transcription(audio_buffer, segments, lock, channels, rate):
    buffer = bytearray()
    frames_per_segment = rate
    sample_width = 2
    while True:
        with lock:
            if audio_buffer:
                buffer.extend(audio_buffer.pop(0))
        if len(buffer) >= frames_per_segment * sample_width * channels:
            segment_data = buffer[:frames_per_segment * sample_width * channels]
            buffer = buffer[frames_per_segment * sample_width * channels:]
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_audio:
                tmp_audio_path = tmp_audio.name
                with wave.open(tmp_audio, 'wb') as wf:
                    wf.setnchannels(channels)
                    wf.setsampwidth(sample_width)
                    wf.setframerate(rate)
                    wf.writeframes(segment_data)
            try:
                new_segments = transcribe_audio(tmp_audio_path)
                with lock:
                    segments.extend(new_segments)
                logging.info(f"Транскрибировано {len(new_segments)} сегментов.")
            except Exception as e:
                logging.error(f'Ошибка транскрипции аудио: {str(e)}')
            finally:
                os.unlink(tmp_audio_path)

def start_audio_capture(audio_buffer, lock, sample_format, channels, rate, chunk):
    def callback(in_data, frame_count, time_info, status):
        with lock:
            audio_buffer.append(in_data)
        return (None, pyaudio.paContinue)
    p = pyaudio.PyAudio()
    stream = p.open(format=sample_format, channels=channels, rate=rate, input=True, frames_per_buffer=chunk, stream_callback=callback)
    stream.start_stream()
    return p, stream

def process_webcam(segments: List[Dict], pose, face_detector, hands):
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise IOError("Не удалось открыть вебкамеру")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 640)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 480)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter('output.mp4', fourcc, fps, (width, height))
    face_output_dir = 'extracted_faces'
    frame_count = 0
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame_count += 1
            objects_found = detect_objects_on_frame(frame)
            person_boxes = [bbox for cls, bbox in objects_found if cls == 'person']
            frame, face_images = detect_and_capture_faces(frame, person_boxes, face_detector)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results_pose = pose.process(rgb_frame)
            frame = detect_pose_on_frame_custom(frame, results_pose)
            results_hands = hands.process(rgb_frame)
            frame = draw_hands_on_frame(frame, results_hands)
            current_time_ms = cap.get(cv2.CAP_PROP_POS_MSEC)
            add_subtitles(frame, segments, current_time_ms)
            if face_images:
                save_faces(face_images, face_output_dir, frame_count)
                place_detected_faces(frame, face_images, width, height, side='right')
            technical_data = ["f/2.8","1/200s","ISO 100","50mm"]
            for idx, data in enumerate(technical_data):
                overlay_text(frame, data, (width - 200, height - 20 - idx*20), font_scale=0.5, color=(0, 255, 255), thickness=1)
            cv2.imshow('Webcam Feed', frame)
            out.write(frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        cap.release()
        out.release()
        cv2.destroyAllWindows()

def main():
    try:
        sample_format = pyaudio.paInt16
        channels = 1
        rate = 16000
        chunk = 1024
        audio_buffer = []
        segments = []
        lock = threading.Lock()
        audio_thread = threading.Thread(target=audio_transcription, args=(audio_buffer, segments, lock, channels, rate), daemon=True)
        audio_thread.start()
        p, stream = start_audio_capture(audio_buffer, lock, sample_format, channels, rate, chunk)
        with mp_pose.Pose(min_detection_confidence=0.5) as pose, mp_face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.5) as face_detector, mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5, min_tracking_confidence=0.5) as hands:
            process_webcam(segments, pose, face_detector, hands)
    except Exception as e:
        logging.error(f'Не удалось обработать видео: {str(e)}')
    finally:
        try:
            stream.stop_stream()
            stream.close()
            p.terminate()
        except:
            pass
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
