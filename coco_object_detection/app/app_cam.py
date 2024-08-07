from flask import Flask, render_template, Response
import cv2
import numpy as np
import tensorflow as tf

app = Flask(__name__)

model_path = 'ssd_mobilenet_v2_coco_2018_03_29/frozen_inference_graph.pb'
label_path = 'app/coco_labels.txt'

with open(label_path, 'r') as f:
    labels = [line.strip() for line in f.readlines()]

detection_graph = tf.Graph()
with detection_graph.as_default():
    od_graph_def = tf.GraphDef()
    with tf.gfile.GFile(model_path, 'rb') as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def, name='')

    sess = tf.Session(graph=detection_graph)

def load_image_into_numpy_array(image):
    (im_width, im_height) = image.size
    return np.array(image).reshape((im_height, im_width, 3)).astype(np.uint8)

def detect_objects(image_np, sess, detection_graph):
    image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
    detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
    detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
    detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')
    num_detections = detection_graph.get_tensor_by_name('num_detections:0')

    image_np_expanded = np.expand_dims(image_np, axis=0)
    (boxes, scores, classes, num) = sess.run(
        [detection_boxes, detection_scores, detection_classes, num_detections],
        feed_dict={image_tensor: image_np_expanded})

    return np.squeeze(boxes), np.squeeze(scores), np.squeeze(classes).astype(np.int32)

def detect_and_draw(frame):
    image_np = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    boxes, scores, classes = detect_objects(image_np, sess, detection_graph)
    for i in range(len(scores)):
        if scores[i] > 0.5:
            box = boxes[i]
            ymin, xmin, ymax, xmax = box
            (left, right, top, bottom) = (xmin * frame.shape[1], xmax * frame.shape[1],
                                          ymin * frame.shape[0], ymax * frame.shape[0])
            p1 = (int(left), int(top))
            p2 = (int(right), int(bottom))
            cv2.rectangle(frame, p1, p2, (77, 255, 9), 3, 1)
            class_name = labels[classes[i] - 1]
            label = '{}: {:.2f}'.format(class_name, scores[i])
            cv2.putText(frame, label, (p1[0], p1[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (77, 255, 9), 2)
    return frame

def generate_frames():
    cap = cv2.VideoCapture(0)
    while True:
        success, frame = cap.read()
        if not success:
            break
        else:
            frame = detect_and_draw(frame)
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000)
