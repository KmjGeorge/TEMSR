import tiffile as tf

# gt = tf.imread('')
lq = tf.imread('G:\datasets\STEM ReSe2\ReSe2\paired\offset\\2219_GT_x19y2.tif')
print(lq.shape)
print(lq.dtype)
print((lq[:, :, 0] == lq[:, :, 1]).all())
print((lq[:, :, 1] == lq[:, :, 2]).all())
