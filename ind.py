import numpy as np

class ind:
    g = np.zeros([1,1])
    f = np.zeros(1)
    c = np.zeros(1)

    def ind_init(self, inum, gnum):
        self.g = np.zeros([inum,gnum],dtype = float)
        self.f = np.zeros([inum],dtype = float)
        self.c = np.zeros([inum],dtype = int)
