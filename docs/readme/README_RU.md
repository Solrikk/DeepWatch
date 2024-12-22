![Logo](https://github.com/Solrikk/DeepWatch/blob/main/assets/photo/photo_2024-12-13_07-09-19.jpg)

<div align="center">
  <h4>
    <a href="https://github.com/Solrikk/DeepWatch/blob/main/README.md">English</a> |
    <a href="https://github.com/Solrikk/DeepWatch/blob/main/docs/readme/README_JP.md">日本語</a> |
    <a href="https://github.com/Solrikk/DeepWatch/blob/main/docs/readme/README_RU.md">⭐Русский⭐</a> |
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

**Описание:** это приложение сочетает в себе возможности компьютерного зрения и распознавания речи для расширенного анализа живого видео. С его помощью вы можете одновременно:

- **Отслеживать положение тела:** встроенный модуль распознавания позы выявляет ключевые точки и рисует скелет, помогая анализировать движения.
- **Детально отображать руки и пальцы:** система [Mediapipe Hands](https://google.github.io/mediapipe/solutions/hands) обеспечивает точное распознавание кистей и пальцев.
- **Распознавать речь в реальном времени:** интеграция с [Whisper](https://github.com/openai/whisper) позволяет мгновенно транскрибировать аудио и накладывать субтитры.
- **Опционально идентифицировать объекты:** [YOLOv5](https://github.com/ultralytics/yolov5) распознаёт различные объекты в кадре, автоматически обрабатывая их координаты.
- **Сохранять выделенные лица (при необходимости):** для дополнительного анализа фотографии лиц могут извлекаться из видеопотока.

## Основной функционал

1. **Компьютерное зрение и анализ движения**  
   Использует [OpenCV](https://opencv.org/) и [Mediapipe Pose](https://google.github.io/mediapipe/solutions/pose) для определения позы и ключевых точек человеческого тела.

2. **Реальное время и субтитры**  
   С помощью [PyAudio](https://people.csail.mit.edu/hubert/pyaudio/) захватывает аудиопоток, который обрабатывается [Whisper](https://github.com/openai/whisper). Результаты транскрибируются в удобном формате субтитров.

3. **Модульный подход к детектированию**  
   Включает в себя гибкую систему для распознавания лиц, рук, объектов и дальнейшего взаимодействия с ними. Легко расширяется под любые задачи.

## Технологии

- **Python**: основной язык разработки.
- **PyTorch**: база для нейросетей, обеспечивает интеграцию с моделями YOLO и Whisper.
- **OpenCV**: чтение и обработка видеопотока, базовые функции компьютерного зрения.
- **Mediapipe**: набор решений для распознавания поз, рук и других ключевых точек тела.
- **PyAudio**: захват и вывод звука в реальном времени.

> **Совместите компьютерное зрение и распознавание речи, чтобы видеть и слышать мир в едином потоке!**

