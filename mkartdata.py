#! /opt/local/bin/python2.7
# -"- coding: utf-8 -"-

import numpy as np
# import csv

# 線形分離可能な人工データを作成する

# データ点の個数
N = 1000

# ランダムな N*2 行列を生成 = 2次元空間上のランダムな点 N 個
# np.random.seed(0)
X = np.random.randn(N, 2)

def h(x, y):
    return 5 * x + 3 * y - 1 # TODO : 適当に決めた真の分離平面 5x + 3y


# TODO : Read correctly
T = np.array([[ 1 if h(x, y) > 0 else -1 for x, y in X ]])

XYT = np.concatenate((X, T.T), axis=1)

np.save('dataXYT.npy', XYT)


# iofile = open('artificialData.csv', mode='w')
# csv_writer = csv.writer( iofile )
# csv_writer.writerows( XYT )
# iofile.close()
