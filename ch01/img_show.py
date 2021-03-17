# coding: utf-8
import os
os.chdir(os.path.dirname(os.path.abspath(__file__)))
import matplotlib.pyplot as plt
from matplotlib.image import imread

img = imread('../dataset/cactus.png') # 이미지 읽어오기
plt.imshow(img)

plt.show()
