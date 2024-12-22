![Logo](https://github.com/Solrikk/DeepWatch/blob/main/assets/photo/photo_2024-12-13_07-09-19.jpg)

<div align="center">
  <h4>
    <a href="https://github.com/Solrikk/DeepWatch/blob/main/README.md">⭐English⭐</a> |
    <a href="https://github.com/Solrikk/DeepWatch/blob/main/docs/readme/README_JP.md">日本語</a> |
    <a href="https://github.com/Solrikk/DeepWatch/blob/main/docs/readme/README_RU.md">Русский</a> |
    <a href="https://github.com/Solrikk/DeepWatch/blob/main/docs/readme/README_FR.md">Français</a> |
    <a href="https://github.com/Solrikk/DeepWatch/blob/main/docs/readme/README_GE.md">Deutsch</a> |
    <a href="https://github.com/Solrikk/DeepWatch/blob/main/docs/readme/README_AR.md">العربية</a> |
    <a href="https://github.com/Solrikk/DeepWatch/blob/main/docs/readme/README_ES.md">Español</a> |
    <a href="https://github.com/Solrikk/DeepWatch/blob/main/docs/readme/README_KR.md">한국어</a> |
    <a href="https://github.com/Solrikk/DeepWatch/blob/main/docs/readme/README_TR.md">Türkçe</a> |
    <a href="https://github.com/Solrikk/DeepWatch/blob/main/docs/readme/README_CN.md">中文</a>
  </h4>
</div>

-----------------

# DeepWatch

**Description:** This application combines computer vision and speech recognition capabilities for enhanced live video analysis. With it, you can simultaneously:

- **Track Body Position:** The built-in pose recognition module detects key points and draws a skeleton, aiding in movement analysis.
- **Detailed Display of Hands and Fingers:** The [Mediapipe Hands](https://google.github.io/mediapipe/solutions/hands) system ensures accurate recognition of palms and fingers.
- **Real-Time Speech Recognition:** Integration with [Whisper](https://github.com/openai/whisper) allows instant audio transcription and subtitle overlay.
- **Optionally Identify Objects:** [YOLOv5](https://github.com/ultralytics/yolov5) recognizes various objects in the frame, automatically processing their coordinates.
- **Save Highlighted Faces (If Necessary):** For additional analysis, faces can be extracted from the video stream.

## Main Functionality

1. **Computer Vision and Motion Analysis**  
   Utilizes [OpenCV](https://opencv.org/) and [Mediapipe Pose](https://google.github.io/mediapipe/solutions/pose) to determine the pose and key points of the human body.

2. **Real-Time and Subtitles**  
   Uses [PyAudio](https://people.csail.mit.edu/hubert/pyaudio/) to capture the audio stream, which is processed by [Whisper](https://github.com/openai/whisper). The results are transcribed into a convenient subtitle format.

3. **Modular Detection Approach**  
   Includes a flexible system for recognizing faces, hands, objects, and further interacting with them. Easily expandable for any tasks.

## Technologies

- **Python:** The primary development language.
- **PyTorch:** The foundation for neural networks, providing integration with YOLO and Whisper models.
- **OpenCV:** Reading and processing the video stream, basic computer vision functions.
- **Mediapipe:** A set of solutions for recognizing poses, hands, and other key body points.
- **PyAudio:** Real-time audio capture and output.

> **Combine computer vision and speech recognition to see and hear the world in a single stream!**
