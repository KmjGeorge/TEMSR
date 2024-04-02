import os
import cv2
import pandas as pd
import re
import tiffile

scale = 2048 / 62.76


def load_data(pic_path, label_path):
    pattern = re.compile(r'(\d\d)\t(\d*.\d*)\t(\d*.\d*)')
    atom_list = []
    # img = cv2.imread(pic_path, 0)
    img = tiffile.imread(pic_path)
    with open(label_path, 'r') as f:
        content = f.readlines()
        for c in content:
            try:
                rlt = re.match(pattern=pattern, string=c)
                cls, x, y = rlt.group(0).split('\t')
                atom_list.append(Atom(cls, x, y))
            except:
                pass
    return img, atom_list


class Atom:
    def __init__(self, label, x, y):
        self.label = label
        self.x = int(round(float(x) * scale))
        self.y = int(round(float(y) * scale))


if __name__ == '__main__':
    img, atoms = load_data('D:\Datasets\Sim ReSe2\\New\ReSe2_2.3_1.2_0_0\image\\2.3_1.2_0_0_35.0_0.6_1052.tif',
                           'D:\Datasets\Sim ReSe2\\New\ReSe2_2.3_1.2_0_0\\2.3-1.2.xyz')
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    print(img.shape)
    print(len(atoms))
    for atom in atoms:
        clr = [0, 255, 0] if atom.label == '75' else [0, 0, 255]
        cv2.circle(img, center=(atom.x, atom.y), color=clr, radius=5, thickness=3)
    # cv2.imshow('img', img)
    # cv2.waitKey(0)
    cv2.imwrite('D:\Datasets\Sim ReSe2\\New\ReSe2_2.3_1.2_0_0\\imgwithlabel.png', img)
