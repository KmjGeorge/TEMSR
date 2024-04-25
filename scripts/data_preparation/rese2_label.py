import os
import cv2
import pandas as pd
import re
import tiffile
import pandas as pd

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
        if label == '34':
            self.label = '0'
            self.size = 25
        else:
            self.label = '1'
            self.size = 40
        self.x = int(round(float(x) * scale))
        self.y = int(round(float(y) * scale))


def make_yololabel(img, atom_list, save_path):
    size_x = img.shape[1]
    size_y = img.shape[0]
    label_cls = []
    label_x = []
    label_y = []
    label_box_x = []
    label_box_y = []
    for atom in atom_list:
        label_cls.append(atom.label)
        label_x.append(atom.x / size_x)
        label_y.append(atom.y / size_y)
        label_box_x.append(atom.size / size_x)
        label_box_y.append(atom.size / size_y)
    with open(os.path.join(save_path, '1.txt'), 'w') as f:
        for i in range(len(label_cls)):
            f.write(str(label_cls[i]) + ' ' + str(label_x[i]) + ' ' + str(label_y[i]) + ' ' + str(
                label_box_x[i]) + ' ' + str(label_box_y[i]) + '\n')
    cv2.imwrite(os.path.join(save_path, '1.png'), img)


def make_crops_with_label(img_path, label_path, stride, ):
    pass


if __name__ == '__main__':
    path = 'G:\datasets\Sim ReSe2\\New\ReSe2_3.1_2.4_0_0'
    img, atoms = load_data(os.path.join(path, 'image', '3.1_2.4_0_0_35.0_0.6_1099.tif'),
                           os.path.join(path, '3.1-2.4.xyz'))
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    make_yololabel(img, atoms, save_path=path)

    '''
    print(img.shape)
    print(len(atoms))
    for atom in atoms:
        clr = [0, 255, 0] if atom.label == '75' else [0, 0, 255]
        cv2.circle(img, center=(atom.x, atom.y), color=clr, radius=5, thickness=3)
    # cv2.imshow('img', img)
    # cv2.waitKey(0)
    cv2.imwrite('G:\Datasets\Sim ReSe2\\New\ReSe2_2.3_1.2_0_0\\imgwithlabel.png', img)
    '''
