import torch
import os
import cv2
import numpy as np
import sys
import mediapipe as mp
import logging
from typing import List, Dict, Tuple

sys.path.append('yolov5')
from models.common import DetectMultiBackend
from utils.torch_utils import select_device
from utils.general import non_max_suppression
from utils.augmentations import letterbox

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

device = select_device('cpu')
yolo_model = DetectMultiBackend('yolov5l.pt', device=device)
imgsz = [640, 640]

mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

def detect_and_capture_faces(frame: np.ndarray) -> Tuple[np.ndarray, List[np.ndarray]]:
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5)
    face_images = []
    for (x, y, w, h) in faces:
        face_img = frame[y:y + h, x:x + w]
        face_images.append(face_img)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
    return frame, face_images

def detect_pose_on_frame(frame: np.ndarray, pose) -> np.ndarray:
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results_pose = pose.process(rgb_frame)
    if results_pose.pose_landmarks:
        mp_drawing.draw_landmarks(
            frame,
            results_pose.pose_landmarks,
            mp_pose.POSE_CONNECTIONS,
            mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=2),
            mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2))
    return frame

def save_faces(face_images: List[np.ndarray], output_dir: str) -> None:
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    for i, face in enumerate(face_images):
        face_filename = os.path.join(output_dir, f"face_{i}.png")
        cv2.imwrite(face_filename, face)

def place_detected_faces(frame: np.ndarray, face_images: List[np.ndarray], width: int, height: int) -> None:
    for idx, face in enumerate(face_images):
        face = cv2.resize(face, (100, 100))
        x_offset = width - 120
        y_offset = 10 + (110 * idx)
        if y_offset + 100 <= height:
            frame[y_offset:y_offset + 100, x_offset:x_offset + 100] = face

def load_reference_objects(object_dir: str) -> Dict[str, np.ndarray]:
    ref_objects = {}
    if os.path.exists(object_dir):
        for fname in os.listdir(object_dir):
            if fname.lower().endswith(('.png', '.jpg', '.jpeg')):
                class_name = os.path.splitext(fname)[0]
                obj_img = cv2.imread(os.path.join(object_dir, fname))
                if obj_img is not None:
                    ref_objects[class_name.lower()] = obj_img
    return ref_objects

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
        h, w = img.shape[2:]
        gain = min(w / w0, h / h0)
        pad_w = (w - w0 * gain) / 2
        pad_h = (h - h0 * gain) / 2
        for *box, conf, cls in pred[0]:
            x1, y1, x2, y2 = box
            x1 = (x1 - pad_w) / gain
            y1 = (y1 - pad_h) / gain
            x2 = (x2 - pad_w) / gain
            y2 = (y2 - pad_h) / gain
            x1, y1, x2, y2 = map(int, (x1, y1, x2, y2))
            cls_name = yolo_model.names[int(cls)].lower()
            detections.append((cls_name, (x1, y1, x2, y2)))
    return detections

def place_reference_objects(frame: np.ndarray, objects_images: List[np.ndarray], width: int, height: int) -> None:
    for idx, obj_img in enumerate(objects_images):
        obj_img = cv2.resize(obj_img, (100, 100))
        x_offset = 20
        y_offset = 10 + (110 * idx)
        if y_offset + 100 <= height:
            frame[y_offset:y_offset + 100, x_offset:x_offset + 100] = obj_img

def process_image(image_path: str) -> str:
    frame = cv2.imread(image_path)
    if frame is None:
        raise IOError(f"Не удалось прочитать изображение: {image_path}")
    height, width = frame.shape[:2]
    face_output_dir = 'extracted_faces'
    object_output_dir = 'extracted_object'
    reference_objects = load_reference_objects(object_output_dir)
    with mp_pose.Pose(min_detection_confidence=0.5) as pose:
        frame, face_images = detect_and_capture_faces(frame)
        frame = detect_pose_on_frame(frame, pose)
        if face_images:
            save_faces(face_images, face_output_dir)
        place_detected_faces(frame, face_images, width, height)
        objects_found = detect_objects_on_frame(frame)
        matched_objects_images = []
        for cls_name, (x1, y1, x2, y2) in objects_found:
            color = (255, 0, 0)
            if cls_name in reference_objects:
                color = (0, 255, 0)
                matched_objects_images.append(reference_objects[cls_name])
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, cls_name, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        if matched_objects_images:
            place_reference_objects(frame, matched_objects_images, width, height)
    output_path = 'output_image.png'
    cv2.imwrite(output_path, frame)
    return output_path

def main():
    image_path = 'input_image.png'
    try:
        output_path = process_image(image_path)
        logging.info(f"Обработанное изображение сохранено в {output_path}")
    except Exception as e:
        logging.error(f'Не удалось обработать изображение: {str(e)}')

if __name__ == "__main__":
    main()
