import os
path = 'D:/Datasets/TEMPatch for SR/LQ/Val'
filelist = os.listdir(path)
for filename in filelist:
    os.rename(os.path.join(path, filename), os.path.join(path, filename.replace('_lq', '')))