import os
import cv2
import numpy as np
from flask import Flask, render_template, Response, request, jsonify
from tflite_runtime.interpreter import Interpreter
from thermal_camera import ThermalCamera

app = Flask(__name__)

# COCO Model
coco_model_path = 'COCO/detect.tflite'
coco_labelmap_path = 'COCO/coco_labels.txt'

with open(coco_labelmap_path, 'r') as f:
    coco_labels = [line.strip() for line in f.readlines()]
if coco_labels[0] == '???':
    del coco_labels[0]

coco_interpreter = Interpreter(model_path=coco_model_path)
coco_interpreter.allocate_tensors()

coco_input_details = coco_interpreter.get_input_details()
coco_output_details = coco_interpreter.get_output_details()
coco_height = coco_input_details[0]['shape'][1]
coco_width = coco_input_details[0]['shape'][2]
coco_floating_model = (coco_input_details[0]['dtype'] == np.float32)
coco_input_mean = 127.5
coco_input_std = 127.5

def detect_objects_coco(frame):
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame_resized = cv2.resize(frame_rgb, (coco_width, coco_height))
    input_data = np.expand_dims(frame_resized, axis=0)
    if coco_floating_model:
        input_data = (np.float32(input_data) - coco_input_mean) / coco_input_std
    coco_interpreter.set_tensor(coco_input_details[0]['index'], input_data)
    coco_interpreter.invoke()
    boxes = coco_interpreter.get_tensor(coco_output_details[0]['index'])[0]
    classes = coco_interpreter.get_tensor(coco_output_details[1]['index'])[0]
    scores = coco_interpreter.get_tensor(coco_output_details[2]['index'])[0]
    return boxes, classes, scores

def gen_frames_coco():
    cap = cv2.VideoCapture('sample-videos/bottle-detection.mp4')
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        boxes, classes, scores = detect_objects_coco(frame)
        imW, imH = frame.shape[1], frame.shape[0]
        for i in range(len(scores)):
            if scores[i] > 0.5:
                ymin = int(max(1, (boxes[i][0] * imH)))
                xmin = int(max(1, (boxes[i][1] * imW)))
                ymax = int(min(imH, (boxes[i][2] * imH)))
                xmax = int(min(imW, (boxes[i][3] * imW)))
                cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (10, 255, 0), 4)
                object_name = coco_labels[int(classes[i])]
                label = f'{object_name}: {int(scores[i]*100)}%'
                label_size, base_line = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
                label_ymin = max(ymin, label_size[1] + 10)
                cv2.rectangle(frame, (xmin, label_ymin - label_size[1] - 10), (xmin + label_size[0], label_ymin + base_line - 10), (255, 255, 255), cv2.FILLED)
                cv2.putText(frame, label, (xmin, label_ymin - 7), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
        _, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

# MLX90640 Model
mlx_model_path = 'MLX90640/detect.tflite'
mlx_labelmap_path = 'MLX90640/labelmap.txt'

with open(mlx_labelmap_path, 'r') as f:
    mlx_labels = [line.strip() for line in f.readlines()]
if mlx_labels[0] == '???':
    del mlx_labels[0]

mlx_interpreter = Interpreter(model_path=mlx_model_path)
mlx_interpreter.allocate_tensors()

mlx_input_details = mlx_interpreter.get_input_details()
mlx_output_details = mlx_interpreter.get_output_details()
mlx_height = mlx_input_details[0]['shape'][1]
mlx_width = mlx_input_details[0]['shape'][2]
mlx_floating_model = (mlx_input_details[0]['dtype'] == np.float32)
mlx_input_mean = 127.5
mlx_input_std = 127.5

camera = ThermalCamera()

def detect_objects_ml(frame):
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame_resized = cv2.resize(frame_rgb, (mlx_width, mlx_height))
    input_data = np.expand_dims(frame_resized, axis=0)
    if mlx_floating_model:
        input_data = (np.float32(input_data) - mlx_input_mean) / mlx_input_std
    mlx_interpreter.set_tensor(mlx_input_details[0]['index'], input_data)
    mlx_interpreter.invoke()
    boxes = mlx_interpreter.get_tensor(mlx_output_details[1]['index'])[0]
    classes = mlx_interpreter.get_tensor(mlx_output_details[3]['index'])[0]
    scores = mlx_interpreter.get_tensor(mlx_output_details[0]['index'])[0]
    return boxes, classes, scores

def gen_frames_ml():
    while True:
        frame = camera.get_frame()
        frame = camera.apply_colormap(frame)
        if camera.apply_filter:
            frame = cv2.GaussianBlur(frame, (5, 5), 0)

        boxes, classes, scores = detect_objects_ml(frame)
        imW, imH = frame.shape[1], frame.shape[0]
        for i in range(len(scores)):
            if scores[i] > 0.5:
                ymin = int(max(1, (boxes[i][0] * imH)))
                xmin = int(max(1, (boxes[i][1] * imW)))
                ymax = int(min(imH, (boxes[i][2] * imH)))
                xmax = int(min(imW, (boxes[i][3] * imW)))
                cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (10, 255, 0), 2)
                object_name = mlx_labels[int(classes[i])]
                label = f'{object_name}: {int(scores[i]*100)}%'
                label_size, base_line = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
                label_ymin = max(ymin, label_size[1] + 10)
                cv2.rectangle(frame, (xmin, label_ymin - label_size[1] - 10), (xmin + label_size[0], label_ymin + base_line - 10), (255, 255, 255), cv2.FILLED)
                cv2.putText(frame, label, (xmin, label_ymin - 7), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    colormaps = ['inferno', 'magma', 'plasma', 'viridis', 'cividis', 'jet', 'nipy_spectral']
    camera.generate_legend()
    return render_template('index.html', colormaps=colormaps, current_colormap=camera.colormap)

@app.route('/video_feed_coco')
def video_feed_coco():
    return Response(gen_frames_coco(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/video_feed_ml')
def video_feed_ml():
    return Response(gen_frames_ml(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/change_colormap', methods=['POST'])
def change_colormap():
    data = request.get_json()
    colormap = data['colormap']
    camera.set_colormap(colormap)
    return jsonify(success=True)

@app.route('/filter', methods=['GET'])
def toggle_filter():
    camera.toggle_filter()
    return ("Filtering Toggled")

@app.route('/temperature', methods=['POST'])
def get_temperature():
    x = float(request.form['x'])
    y = float(request.form['y'])
    frame = camera.get_frame()

    image_width, image_height = 320, 240
    sensor_width, sensor_height = 32, 24

    sensor_x = int((x / image_width) * sensor_width)
    sensor_y = int((y / image_height) * sensor_height)

    if sensor_x >= sensor_width:
        sensor_x = sensor_width - 1
    if sensor_y >= sensor_height:
        sensor_y = sensor_height - 1

    temperature = frame[sensor_y, sensor_x]
    return jsonify(temperature=temperature)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000, debug=True)
