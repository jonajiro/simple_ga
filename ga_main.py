import numpy as np
import sys
import cv2
import math
import matplotlib
from matplotlib import pyplot as plt
import random
import time
from PIL import Image
import os
import glob

from ga import ga
from ga import ind


def main():


    if not os.path.exists('data'):
        os.mkdir('data')
    if not os.path.exists('gen_data'):
        os.mkdir('gen_data')

    ga_1 = ga()
    ga_1.gen_num = 100
    ga_1.ind_num = 50

    ga_1.gene_num = 2

    ga_1.gene_max[0] = 5
    ga_1.gene_min[0] = -5
    ga_1.gene_max[1] = 5
    ga_1.gene_min[1] = -5

    ga_1.alp = 0.5
    ga_1.mu = 0.1

    ga_1.th_dis = 0.1
    ga_1.th_clust_num = 2
    ga_1.cyclegen_clust = 20

    save_data = np.zeros([ga_1.gen_num+1,5])

    ga_1.ga_init()
    save_data[0,0] = 0
    save_data[0,1] = min(ga_1.ind_c.f)
    save_data[0,2] = max(ga_1.ind_c.c) + 1

    ga_1.ga_clustering()
    for ita in range(ga_1.gen_num):
        if ita % ga_1.cyclegen_clust == 0 and ita != 0:
            ga_1.ga_clustering()
        data_plot(ga_1,ita+1)
        ga_1.ga_shuffle()
        ga_1.ga_cross()
        save_data[ita+1,0] = ita + 1
        save_data[ita+1,1] = min(ga_1.ind_c.f)
        save_data[ita+1,2] = max(ga_1.ind_c.c) + 1

        save_data000 = np.zeros([ga_1.ind_num,ga_1.gene_num+2])
        for i in range(ga_1.ind_num):
            save_data000[i,0] = ga_1.ind_c.f[i]
            save_data000[i,1] = ga_1.ind_c.c[i]
            for j in range(ga_1.gene_num):
                save_data000[i,j+2] = ga_1.ind_c.g[i,j]
        np.savetxt('./gen_data/gen{number:04}.csv'.format(number=ita), save_data000, delimiter=",", fmt="%.5f")

        print(min(ga_1.ind_c.f))
    print("end")
    ga_1.ga_eval()
    print(ga_1.ind_c.g)
    print(ga_1.ind_c.c)
    video_output(ga_1)

    np.savetxt("fit.csv", save_data, delimiter=",", fmt="%.5f")

def video_output(ga_i):
    #ビデオ作成
    # encoder(for mp4)
    # fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
    fourcc = cv2.VideoWriter_fourcc('h', '2', '6', '4')
    # fourcc = cv2.VideoWriter_fourcc(*'XVID')
    # output file name, encoder, fps, size(fit to image size)

    video = cv2.VideoWriter('video.mp4',fourcc, 20.0, (640, 480))
    # video = cv2.VideoWriter('video.avi',fourcc, 20.0, (img.shape[1], img.shape[0]))

    if not video.isOpened():
        print("can't be opened")
        sys.exit()

    for i in range(0, ga_i.gen_num+1):
        # hoge0000.png, hoge0001.png,..., hoge0090.png
        img = cv2.imread("./data/" +'./fit%04d.png' % i)
        # can't read image, escape
        if img is None:
            print("can't read")
            break
        # add
        video.write(img)
        print(i)

    video.release()
    print('written')

def data_plot(ga_i,num):
    ga_i.ga_eval()
    f_cross_index = np.argsort(ga_i.ind_c.f)
    x = np.arange(ga_i.gene_min[0], ga_i.gene_max[0], (ga_i.gene_max[0]-ga_i.gene_min[0])/100) #x軸の描画範囲の生成。0から10まで0.05刻み。
    y = np.arange(ga_i.gene_min[1], ga_i.gene_max[1], (ga_i.gene_max[1]-ga_i.gene_min[1])/100) #y軸の描画範囲の生成。0から10まで0.05刻み。

    X, Y = np.meshgrid(x, y)
    Z = np.zeros([len(X),len(Y)])
    for i in range(len(X)):
        for j in range(len(Y)):
            Z[i,j] = ga_i.func([X[i,j],Y[i,j]])

    # 等高線図の生成。
    cont=plt.contour(X,Y,Z,  5, vmin=-1,vmax=1, colors=['black'])
    # cont.clabel(fmt='%1.1f', fontsize=14)


    plt.xlabel('X', fontsize=24)
    plt.ylabel('Y', fontsize=24)


    plt.pcolormesh(X,Y,Z, cmap='cool') #カラー等高線図
    pp=plt.colorbar (orientation="vertical") # カラーバーの表示
    pp.set_label("Fitness",  fontsize=24)

    N=128
    f = 8
    omg = [i*f*2*np.pi/N for i in range(N)]
    sig = [np.sin(i) for i in omg]

    colorlist = ["r", "g", "b", "c", "m", "y", "k", "w"]

    for i in range(ga_i.ind_num):
        c_num = ga_i.ind_c.c[i]
        for j in range(int(max(ga_i.ind_c.c) + 1)):
            if c_num > (len(colorlist) - 1):
                c_num = c_num - (len(colorlist) - 1)
            else:
                break
        plt.scatter(ga_i.ind_c.g[i,0], ga_i.ind_c.g[i,1], color=colorlist[c_num])
    plt.xlim(ga_i.gene_min[0], ga_i.gene_max[0])
    plt.ylim(ga_i.gene_min[1], ga_i.gene_max[1])
    plt.title('step{number:04}'.format(number=num))
    plt.savefig("./data/" +'fit{number:04}.png'.format(number=num)) # この行を追記
    plt.clf()


if __name__ == "__main__":
    main()
