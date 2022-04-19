import cv2
import os

def find_image(label_images, order ):
    for path in label_images:
        if path.split("_iter_")[-1].split(".jpg")[0] == str(order*10):
            return path
    return None

img_root = '../results/scene0164_03/0/'
label = 24

images_path = os.listdir(img_root)
fps = 5    #保存视频的FPS，可以适当调整

label_images = []

for path in images_path:
    if path.split("label_")[-1].split("_iter")[0] == str(label):
        label_images.append(path)


size = (1132, 228)

#可以用(*'DVIX')或(*'X264'),如果都不行先装ffmepg: sudo apt-get install ffmepg
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
videoWriter = cv2.VideoWriter(f'./results/video_demo/demo_{label}.mp4', fourcc, fps, size)#最后一个是保存图片的尺寸



for order in range(len(label_images)):
    path = find_image(label_images, order)
    if path == None:
        break

    frame = cv2.imread(img_root+path)
    cv2.imwrite(img_root + 'sample.png', frame)
    videoWriter.write(frame)
videoWriter.release()