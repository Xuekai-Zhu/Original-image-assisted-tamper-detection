from PIL import Image
import numpy as np
import shutil
import cv2
import os


f2 = os.listdir(path)
def dHash(image):
    image_new = image
    # 计算均值
    avreage = np.mean(image_new)
    hash = []
    # 每行前一个像素大于后一个像素为1，相反为0，生成哈希
    for i in range(8):
        for j in range(8):
            if image[i, j] > image[i, j + 1]:
                hash.append(1)
            else:
                hash.append(0)
    return hash


def Hamming_distance(hash1, hash2):
    num = 0
    for index in range(len(hash1)):
        if hash1[index] != hash2[index]:
            num += 1
    return num


#匹配文件夹路径，输出文件夹路径A，输出文件夹路径B
def Dhashmatch(match_path,path_a,path_b):
	path=match_path
	flist = os.listdir(path)
	for match_dir in flist:
	    f = os.listdir(path + '/' + match_dir)
	    im1 = max(f)
	    image1 = Image.open(path + '/' + match_dir + '/' + im1)
	    h, w = image1.size[0], image1.size[1]
	    image1 = np.array(image1.resize((9, 8), Image.ANTIALIAS).convert('L'), 'f')
	    similar = 0
	    s = im1
	    for i in f:
		if (i == im1):
		    continue

		image2 = Image.open(path + '/' + match_dir + '/' + i)
		img_np = np.array(image2)
		m_h, m_w, _ = img_np.shape

		##去除size不一致和全白匹配
		img_np = np.array(image2)
		img_thr = cv2.cvtColor(img_np, cv2.COLOR_BGR2GRAY)
		all_area = h * w
		num = np.sum(img_thr == 255)
		p = num / all_area
		if p > 0.99 or m_h != h or m_w != w:
		    continue

		image2 = np.array(image2.resize((9, 8), Image.ANTIALIAS).convert('L'), 'f')
		hash1 = dHash(image1)
		hash2 = dHash(image2)
		dist = Hamming_distance(hash1, hash2)
		similarity = 1 - dist * 1.0 / 64
		if ((similarity > similar) and (similarity > 0.8)):
		    s = i
		    similar = similarity
	    if(s!=im1)
		    source_a = os.path.join(path, match_dir, im1)
		    target_a = os.path.join(path_a, match_dir + '.jpg')
		    shutil.copy(source_a, target_a)

		    source_b = os.path.join(path, match_dir, s)
		    target_b = os.path.join(path_b, match_dir + '.jpg')
		    shutil.copy(source_b, target_b)
	    else:
		    source_a = os.path.join(path, match_dir, im1)
		    target_a = os.path.join(splicing_path, match_dir + '.jpg')
		    shutil.copy(source_a, target_a)





