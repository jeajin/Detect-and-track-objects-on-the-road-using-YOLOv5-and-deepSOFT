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
    import cv2
    import os
    from ultralytics import YOLO


    # Load a model
    model = YOLO("yolov8m.yaml")  # build a new model from scratch
    model = YOLO("yolov8m.pt")  # load a pretrained model (recommended for training)

    # Use the model
    model.train(data="./datasets/car/test.yaml", epochs=10)  # train the model
    metrics = model.val()  # evaluate model performance on the validation set

    ##################

    video_paths = os.getcwd() + "/datasets/car/v3.mp4"
    cap = cv2.VideoCapture(video_paths)

    while cap.isOpened():
        success, frame = cap.read()

        if success:
            results = model(frame)

            annotated_frame = results[0].plot()

            cv2.imshow("YOLOv8 inference", annotated_frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
        else:
            break
    cap.release()
    cv2.destroyAllWindows()
    # results = model("https://ultralytics.com/images/bus.jpg")  # predict on an image
    # results = model(os.getcwd()+"/datasets/car/testImage/testImage_000000.jpg")  # predict on an image

    path = model.export(format="onnx")  # export the model to ONNX format




'''
import cv2
import os
from ultralytics import  YOLO

model = YOLO("yolov8n.pt")

video_paths = os.getcwd() + "/datasets/car/v3.mp4"
cap = cv2.VideoCapture(video_paths)

while cap.isOpened():
    success, frame = cap.read()

    if success:
        results = model(frame)

        annotated_frame = results[0].plot()

        cv2.imshow("YOLOv8 inference",annotated_frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    else:
        break
cap.release()
cv2.destroyAllWindows()
'''




