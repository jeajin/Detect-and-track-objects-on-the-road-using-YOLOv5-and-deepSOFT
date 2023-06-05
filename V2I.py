# video to images
import cv2
import os
import glob
from PIL import Image

def v2i(video_paths, video_names, n ):
    for path, vn in zip(video_paths, video_names):

        # Read the video from specified path
        cam = cv2.VideoCapture(path)

        try:

            # creating a folder named data
            if not os.path.exists(vn):
                os.makedirs(vn)

        # if not created then raise error
        except OSError:
            print('Error: Creating directory of data')

        # frame
        currentframe = 0
        while (True):

            # reading from frame
            ret, frame = cam.read()
            if currentframe % n == 0:
                if ret:
                    # if video is still left continue creating images
                    name = "./" + vn + "/" + vn + "_%06d.jpg" % currentframe
                    if currentframe % 100 == 0:
                        print('Creating... ' + name)

                    # writing the extracted images /Image.open(frame).resize()
                    cv2.imwrite(name, frame)

                    # increasing counter so that it will
                    # show how many frames are created
                    currentframe += 1
                else:
                    print('Creating... ' + name)
                    break
            else:
                currentframe += 1
        cam.release()
    # Release all space and windows once done

    cv2.destroyAllWindows()

def show_video(name):

    capture = cv2.VideoCapture(name)

    while cv2.waitKey(33) < 0:
        if capture.get(cv2.CAP_PROP_POS_FRAMES) == capture.get(cv2.CAP_PROP_FRAME_COUNT):
            capture.set(cv2.CAP_PROP_POS_FRAMES, 0)

        ret, frame = capture.read()
        cv2.imshow("VideoFrame", frame)

    capture.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':

    os.chdir("./datasets/car")
    paths = os.getcwd()
    video_paths = glob.glob(paths + '/v3.mp4')
    video_names = []
    for video in os.listdir():
        if '.mp4' in video:
            print(video)
            video_names.append("testImage")

    v2i(video_paths, video_names, 30)

    '''
    os.chdir("./datasets/car")
    paths = os.getcwd()
    video_paths = glob.glob(paths + '/*.mp4')
    video_names = []
    for video in os.listdir():
        if '.mp4' in video:
            print(video)
            video_names.append(video[:-4])

            
    v2i(video_paths, video_names)
    '''


    #show_video(video_names[2]+'.mp4')
