import numpy as np
import math
import random
from ind import ind
class ga:
    ind_num = 100
    gen_num = 100
    gene_num = 2
    gene_max = 20*np.ones(gene_num)
    gene_min = -20*np.ones(gene_num)
    alp = 0.5
    mu = 0.1
    cyclegen_clust = 80
    th_dis = 0.1
    th_clust_num = 3

    ind_c = ind()

    def ga_init(self):
        self.ind_c.ind_init(self.ind_num,self.gene_num)
        # self.gene.gene_init(self.ind_num,self.gene_num)
        for i in range(self.ind_num):
            for j in range(self.gene_num):
                self.ind_c.g[i,j] = (self.gene_max[j] - self.gene_min[j]) * random.random() + self.gene_min[j]
        self.ga_eval()

    def ga_eval(self):
        for i in range(self.ind_num):
            self.ind_c.f[i] = self.func(self.ind_c.g[i,:])

    def ga_eval_clust(self,clust_num):
        f_in = 0
        clust_f = 0
        clust_ind = 0
        for i in range(self.ind_num):
            if self.ind_c.c[i] == clust_num:
                if f_in == 0:
                    f_in = 1
                    clust_f = self.ind_c.f[i]
                    clust_ind = i
                else:
                    if clust_f > self.ind_c.f[i]:
                        clust_f = self.ind_c.f[i]
                        clust_ind = i

        return f_in , clust_f , clust_ind

    def ga_cross(self):

        for i in range(int(self.ind_num/2)):
            gene_cross = np.zeros([4,self.gene_num])
            f_cross = np.zeros(4)

            for j in range(self.gene_num):
                gene_cross[0,j] = self.ind_c.g[i,j]
            f_cross[0] = self.ind_c.f[i]
            for j in range(self.gene_num):
                gene_cross[1,j] = self.ind_c.g[i*2+1,j]
            f_cross[1] = self.ind_c.f[i*2+1]

            for j in range(self.gene_num):
                if self.mu > random.random():
                    gene_cross[2,j] = (self.gene_max[j] - self.gene_min[j]) * random.random() + self.gene_min[j]
                else:
                    if (gene_cross[0,j] - gene_cross[1,j]) > 0:
                        ugene = gene_cross[0,j] + (gene_cross[0,j] - gene_cross[1,j]) * self.alp
                        lgene = gene_cross[1,j] - (gene_cross[0,j] - gene_cross[1,j]) * self.alp
                        gene_cross[2,j] = (ugene - lgene) * random.random() + lgene
                        if gene_cross[2,j] > self.gene_max[j]:
                            gene_cross[2,j] = self.gene_max[j]
                        if gene_cross[2,j] < self.gene_min[j]:
                            gene_cross[2,j] = self.gene_min[j]
                    else:
                        ugene = gene_cross[1,j] + (gene_cross[1,j] - gene_cross[0,j]) * self.alp
                        lgene = gene_cross[0,j] - (gene_cross[1,j] - gene_cross[0,j]) * self.alp
                        gene_cross[2,j] = (ugene - lgene) * random.random() + lgene
                        if gene_cross[2,j] > self.gene_max[j]:
                            gene_cross[2,j] = self.gene_max[j]
                        if gene_cross[2,j] < self.gene_min[j]:
                            gene_cross[2,j] = self.gene_min[j]
            f_cross[2] = self.func(gene_cross[2,:])

            for j in range(self.gene_num):
                if self.mu > random.random():
                    gene_cross[3,j] = (self.gene_max[j] - self.gene_min[j]) * random.random() + self.gene_min[j]
                else:
                    if (gene_cross[0,j] - gene_cross[1,j]) > 0:
                        ugene = gene_cross[0,j] + (gene_cross[0,j] - gene_cross[1,j]) * self.alp
                        lgene = gene_cross[1,j] - (gene_cross[0,j] - gene_cross[1,j]) * self.alp
                        gene_cross[3,j] = (ugene - lgene) * random.random() + lgene
                        if gene_cross[3,j] > self.gene_max[j]:
                            gene_cross[3,j] = self.gene_max[j]
                        if gene_cross[3,j] < self.gene_min[j]:
                            gene_cross[3,j] = self.gene_min[j]
                    else:
                        ugene = gene_cross[1,j] + (gene_cross[1,j] - gene_cross[0,j]) * self.alp
                        lgene = gene_cross[0,j] - (gene_cross[1,j] - gene_cross[0,j]) * self.alp
                        gene_cross[3,j] = (ugene - lgene) * random.random() + lgene
                        if gene_cross[3,j] > self.gene_max[j]:
                            gene_cross[3,j] = self.gene_max[j]
                        if gene_cross[3,j] < self.gene_min[j]:
                            gene_cross[3,j] = self.gene_min[j]
            f_cross[3] = self.func(gene_cross[3,:])

            f_cross_index = np.argsort(f_cross)
            for j in range(self.gene_num):
                 self.ind_c.g[i,j] = gene_cross[f_cross_index[0],j]
                 self.ind_c.f[i] = f_cross[f_cross_index[0]]
            for j in range(self.gene_num):
                 self.ind_c.g[i*2+1,j] = gene_cross[f_cross_index[1],j]
                 self.ind_c.f[i*2+1] = f_cross[f_cross_index[1]]



    def ga_shuffle(self):
        clust_num = max(self.ind_c.c) + 1
        clust_num_num = np.zeros(clust_num,dtype = int)

        for i in range(clust_num):
            for j in range(int(self.ind_num/2)):
                if self.ind_c.c[j] == i:
                    clust_num_num[i] = clust_num_num[i] + 1
        # print(clust_num_num)
        cnt0 = 0
        for i in range(clust_num):
            if clust_num_num[i] > 1:
                for j in range(clust_num_num[i]):
                    s_num = int((clust_num_num[i] - 1) * random.random()) + j
                    if s_num > clust_num_num[i]:
                        s_num = s_num - clust_num_num[i]
                    for k in range(self.gene_num):
                        swap_gene = self.ind_c.g[j + cnt0,k]
                        self.ind_c.g[j + cnt0,k] = self.ind_c.g[s_num + cnt0,k]
                        self.ind_c.g[s_num + cnt0,k] = swap_gene

                    swap_f = self.ind_c.f[j + cnt0]
                    self.ind_c.f[j + cnt0] = self.ind_c.f[s_num + cnt0]
                    self.ind_c.f[s_num + cnt0] = swap_f
            cnt0 = cnt0 + clust_num_num[i]

        cnt0 = int(self.ind_num/2)-1
        for i in range(clust_num):
            if clust_num_num[i] > 1:
                for j in range(clust_num_num[i]):
                    s_num = int((clust_num_num[i] - 1) * random.random()) + j
                    if s_num > clust_num_num[i]:
                        s_num = s_num - clust_num_num[i]
                    for k in range(self.gene_num):
                        swap_gene = self.ind_c.g[j + cnt0,k]
                        self.ind_c.g[j + cnt0,k] = self.ind_c.g[s_num + cnt0,k]
                        self.ind_c.g[s_num + cnt0,k] = swap_gene

                    swap_f = self.ind_c.f[j + cnt0]
                    self.ind_c.f[j + cnt0] = self.ind_c.f[s_num + cnt0]
                    self.ind_c.f[s_num + cnt0] = swap_f
            cnt0 = cnt0 + clust_num_num[i]

    def ga_clustering(self):
        #非類似度は遺伝子距離で算出

        #クラスタ振り直し
        for i in range(self.ind_num):
            self.ind_c.c[i] = i
        clust_num = max(self.ind_c.c) + 1

        clust_ind = self.ind_c.c.copy()
        gene_clust = self.ind_c.g.copy()
        dis_min = 0

        while 1:
            dis_clust = np.zeros([clust_num,clust_num])
            mrg_clust = np.zeros(2,dtype=int)
            dis_min = 0

            for i in range(clust_num):
                for j in range(clust_num):
                    if i == j:
                        break
                    else:
                        for k in range(self.gene_num):
                            dis_clust[i,j] = dis_clust[i,j] + (gene_clust[j,k] - gene_clust[i,k])**2
                        if i == 1 and j == 0:
                            dis_min = dis_clust[i,j]

                            mrg_clust[0] = i
                            mrg_clust[1] = j

                        else:
                            if dis_min > dis_clust[i,j]:
                                dis_min = dis_clust[i,j]
                                mrg_clust[0] = i
                                mrg_clust[1] = j

            if dis_min > self.th_dis:
                #併合するクラスタ間の非類似度が閾値を超えたときクラスタリング終了
                break

            cnt0 = 0
            for i in range(self.ind_num):
                #併合個体に最終クラスタ番号+1を割り振る
                if clust_ind[i] == mrg_clust[0] or clust_ind[i] == mrg_clust[1]:
                    clust_ind[i] = clust_num
                    cnt0 = cnt0 + 1

            #併合個体以外のクラスタ番号を更新する
            #併合個体のクラスタを末尾に移動させ、その他クラスタを小さい順に並べる
            clust_ind_index = np.argsort(clust_ind)
            cnt1 = clust_ind[clust_ind_index[0]]
            cnt2 = 0
            for i in range(self.ind_num - cnt0):
                if clust_ind[clust_ind_index[i]] == cnt1:
                    clust_ind[clust_ind_index[i]] = cnt2
                else:
                    cnt1 = clust_ind[clust_ind_index[i]]
                    cnt2 = cnt2 + 1
                    clust_ind[clust_ind_index[i]] = cnt2

            for i in range(cnt0):
                #併合個体に最終クラスタ番号-1を割り振る
                clust_ind[clust_ind_index[i + self.ind_num - cnt0]] = clust_num - 2

            #クラスタ数を減算する
            clust_num = clust_num - 1

            if clust_num < self.th_clust_num:
                #クラスタ数が規定値を下回った時クラスタリング終了
                break

            #クラスタ遺伝子を更新する
            gene_clust = np.zeros([clust_num,self.gene_num])

            cnt1 = 0
            for i in range(clust_num):
                cnt1 = 0
                for j in range(self.ind_num):
                    if i == clust_ind[j]:
                        for k in range(self.gene_num):
                            gene_clust[i,k] = gene_clust[i,k] + self.ind_c.g[j,k]
                        cnt1 = cnt1 + 1
                for j in range(self.gene_num):
                    gene_clust[i,j] = gene_clust[i,j]/cnt1


        #個体数が奇数のクラスタを検知
        clust_num = int(max(clust_ind)+1)
        ind_num_clust = np.zeros(clust_num)
        clust_ind_last = 0

        f_odd = 0
        clust_ind_odd = 0
        clust_ind_last_odd = 0

        for i in range(clust_num):
            for j in range(self.ind_num):
                if i == clust_ind[j]:
                    ind_num_clust[i] = ind_num_clust[i] + 1
                    clust_ind_last = j
            if (ind_num_clust[i] % 2) != 0:#奇数の時
                if f_odd == 1:#奇数で２個目の時
                    if ind_num_clust[i] > ind_num_clust[clust_ind_odd]:
                        clust_ind[clust_ind_last] = clust_ind_odd
                    else:
                        clust_ind[clust_ind_last_odd] = i
                    f_odd = 0
                else:
                    f_odd = 1
                    clust_ind_odd = i
                    clust_ind_last_odd = clust_ind_last

        # print(clust_ind)
        #クラスタ番号を更新する
        clust_ind_index = np.argsort(clust_ind)
        cnt1 = clust_ind[clust_ind_index[0]]
        cnt2 = 0
        for i in range(self.ind_num):
            if clust_ind[clust_ind_index[i]] == cnt1:
                clust_ind[clust_ind_index[i]] = cnt2
            else:
                cnt1 = clust_ind[clust_ind_index[i]]
                cnt2 = cnt2 + 1
                clust_ind[clust_ind_index[i]] = cnt2

        clust_num = int(max(clust_ind)+1)
        ind_num_clust = np.zeros(clust_num,dtype=int)
        gene_buf = self.ind_c.g.copy()
        f_buf = self.ind_c.f.copy()
        cnt0 = 0
        for i in range(clust_num):
            for j in range(self.ind_num):
                if i == clust_ind[j]:
                    for k in range(self.gene_num):
                        self.ind_c.g[cnt0,k] = gene_buf[j,k]
                    self.ind_c.f[cnt0] = f_buf[j]
                    self.ind_c.c[cnt0] = i
                    cnt0 = cnt0 + 1
                    ind_num_clust[i] = ind_num_clust[i] + 1


        gene_buf = self.ind_c.g.copy()
        f_buf = self.ind_c.f.copy()
        cnt0 = 0
        cnt1 = 0
        for i in range(clust_num):
            for j in range(int(ind_num_clust[i]/2)):
                for k in range(self.gene_num):
                    self.ind_c.g[cnt1,k] = gene_buf[j+cnt0,k]
                self.ind_c.f[cnt1] = f_buf[j+cnt0]
                self.ind_c.c[cnt1] = i

                for k in range(self.gene_num):
                    self.ind_c.g[cnt1 + int(self.ind_num/2),k] = gene_buf[j+int(ind_num_clust[i]/2)+cnt0,k]
                self.ind_c.f[cnt1 + int(self.ind_num/2)] = f_buf[j+int(ind_num_clust[i]/2)+cnt0]
                self.ind_c.c[cnt1 + int(self.ind_num/2)] = i
                cnt1 = cnt1 + 1
            cnt0 = cnt0 + ind_num_clust[i]

    def func(self,gene):#<function name> (大域解X,Y) 大域解適応度 (遺伝子上下限値 X,Y)
        #Booth function (1,3) 0 (-10~10)
        # f = (gene[0] + 2*gene[1] - 7)**2 + (2*gene[0] + gene[1] - 5)**2

        # Ackley function (0,0) 0 (-32.768, 32.768)
        # f = -20*np.exp(-0.2*np.sqrt(0.5*(gene[0]**2+gene[1]**2)))-np.exp(0.5*(np.cos(2*np.pi*gene[0])+np.cos(2*np.pi*gene[1])))+np.e+20

        # Beale function (3,0.5) 0 (-4.5, 4.5)
        # f = (1.5 - gene[0] + gene[0]*gene[1])**2 + (2.25 - gene[0] + gene[0] * (gene[1]**2))**2 + (2.625 - gene[0] + gene[0] * (gene[1]**3))**2

        # Drop-Wave function(0,0) -1 (-5.12, 5.12)
        # f = -(1 + np.cos(12 * np.sqrt(gene[0]**2 + gene[1]**2)))/((0.5 * (gene[0]**2 + gene[1]**2)) + 2)

        # Himmelblau's function (3, 2 -3.78, -3.28 -2.81, 3.13 3.58, -1.85) 0 (-5~5)
        f = (gene[0]**2 + gene[1] -11)**2 + (gene[0] + gene[1]**2 - 7)**2

        # Rosenbrock function (banana function)(1,1 ) 0 (-5, 10)
        # f = 100*(gene[1] - gene[0]**2)**2 - (gene[0] - 1)**2

        # Eggholder function (512,404.2319...pi) -959.6407 (-512~512)
        # f = -(gene[1] + 47)*np.sin(np.sqrt(abs(gene[1] + gene[0]/2 + 47))) - gene[0]*np.sin(np.sqrt(abs(gene[0] - (gene[1] + 47))))

        # #Bukin function N.6 (-10,1) 0 (-15~-5,-3~3)
        # f = 100 * np.sqrt(abs(gene[1] - 0.01*gene[0]**2)) + 0.01*abs(gene[0] + 10)

        return f
