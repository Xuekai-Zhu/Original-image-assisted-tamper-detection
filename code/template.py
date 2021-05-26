import cv2
import numpy as np
from matplotlib import pyplot as plt
import os

#模板匹配算法参数(匹配路径,输出路径，匹配对象，匹配模板)
def template(path,out,datalist,temp):  
    for i in datalist:
        img_1 = path + '/' + i
        dirs=out+i[:-4]
        if  os.path.exists(dirs):
            continue
        os.mkdir(dirs)
        template = cv2.imread(img_1, cv2.IMREAD_GRAYSCALE)
        template2 = cv2.imread(img_1)
        for j in temp:
            try:

                img_2 = y + '/' + j
                img_rgb = cv2.imread(img_2, cv2.IMREAD_COLOR)
                img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2GRAY)

                h, w = template.shape[:2]

                # 归一化平方差匹配
                res = cv2.matchTemplate(img_gray, template, cv2.TM_CCOEFF_NORMED)


                threshold = 0.6


                min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
                if max_val <= threshold:
                    continue


                img_out = out + i[:-4]+'/' + i.split('.')[0]  + '+' + j
                cv2.imwrite(img_out, img_rgb[max_loc[1]: max_loc[1] + h, max_loc[0]:max_loc[0] + w, :])
            except:
                continue
        cv2.imwrite(dirs+'/'+i,template2)

