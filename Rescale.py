import cv2
import shutil
import os
import argparse
import sys
import glob

def change_res(path, width, height):
    fps = int(round((cv2.VideoCapture(path)).get(cv2.CAP_PROP_FPS)))
    os.rename(path, path[:-4] + '_old_' + '.avi')
    out = cv2.VideoWriter(path, cv2.VideoWriter_fourcc('X','V','I','D'), fps, (width, height))

    path = path[:-4] + '_old_' + '.avi'

    cap = cv2.VideoCapture(path)

    while cap.isOpened():
        sucess, frame = cap.read()
        if sucess == False:
            break
     
        resize = cv2.resize(frame, (width, height), cv2.INTER_CUBIC)
        out.write(resize)

    cap.release()
    out.release()
    shutil.move(path, 'trash.mp4')
    cv2.destroyAllWindows()
print('Ok')
folderPath = 'c:\\Users\\ufxds\\Desktop\\DeepWorkOut\\Videos\\'
paths = glob.glob(folderPath + '*.mp4')
print(len(paths))
for path in paths:
    change_res(path, 640, 360)