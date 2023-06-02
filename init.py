# git clone https://github.com/ultralytics/yolov5 /pip install ultralytics /git clone https://github.com/nwojke/deep_sort.git
# cd yolov5
# pip install -r requirements.txt
# mingw-w64 설치

'''
    yolov8 test
yolo predict model=yolov8n.pt source='https://ultralytics.com/images/bus.jpg'


from ultralytics import YOLO
# Load a model
model = YOLO("yolov8n.yaml")  # build a new model from scratch
model = YOLO("yolov8n.pt")  # load a pretrained model (recommended for training)

# Use the model
model.train(data="coco128.yaml", epochs=3)  # train the model
metrics = model.val()  # evaluate model performance on the validation set
results = model("https://ultralytics.com/images/bus.jpg")  # predict on an image
path = model.export(format="onnx")  # export the model to ONNX format
'''

'''
video test
import cv2

capture = cv2.VideoCapture("v1.mp4")

while cv2.waitKey(33) < 0:
    if capture.get(cv2.CAP_PROP_POS_FRAMES) == capture.get(cv2.CAP_PROP_FRAME_COUNT):
        capture.set(cv2.CAP_PROP_POS_FRAMES, 0)

    ret, frame = capture.read()
    cv2.imshow("VideoFrame", frame)

capture.release()
cv2.destroyAllWindows()
'''


if __name__ == '__main__' :
    from ultralytics import YOLO

    # Load a model
    model = YOLO("yolov8m.yaml")  # build a new model from scratch
    model = YOLO("yolov8m.pt")  # load a pretrained model (recommended for training)

    # Use the model
    model.train(data="coco128.yaml", epochs=3)  # train the model
    metrics = model.val()  # evaluate model performance on the validation set
    results = model("https://ultralytics.com/images/bus.jpg")  # predict on an image
    path = model.export(format="onnx")  # export the model to ONNX format