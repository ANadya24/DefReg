import os
from read_tiff import read_tiff
from glob import glob
import cv2
from tqdm import tqdm

PATH = './data/'
output_path = './nopermute_dataset/masks/'

if not os.path.exists(output_path):
    os.mkdir(output_path)

filenames = glob(PATH + 'Seq*_mask.tif')
for file in filenames:
    name = file.split('/')[-1].split('.')[0]
    images, num = read_tiff(file)
    # for i, im in enumerate(tqdm(images[:-1])):
    #     for j, im2 in enumerate(images[i+1:]):
    #         cv2.imwrite(output_path + f'{name}-{i}{j}_1.jpg', im)
    #         cv2.imwrite(output_path +f'{name}-{i}{j}_2.jpg', im2)
    # print(f"Done {file}\n")
    for i, im in enumerate(tqdm(images[:-1:2])):
        im2 = images[i+1]
        cv2.imwrite(output_path + f'{name}-{i}_1.jpg', im)
        cv2.imwrite(output_path +f'{name}-{i}_2.jpg', im2)
    print(f"Done {file}\n")


