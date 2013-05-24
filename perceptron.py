#! /opt/local/bin/python2.7
# -"- encoding: utf-8 -"-

# package, module
import numpy as np
import random
import matplotlib.pyplot as plt
import time
# import csv

plt.close()


# function 関数
# イテレーション内でのプロット
def plotter(data, x, w):
    plt.clf()
    # axes
    xmax = np.max(data[:, 0])
    xmin = np.min(data[:, 0])
    x = np.arange(xmin, xmax)
    plt.xlim(xmin+1, xmax-1)
    plt.ylim(xmin+1, xmax-1)
    # plot
    plt.plot(data[data[:, 2] == 1, 0],
             data[data[:, 2] == 1, 1], 'o', color='red')
    plt.plot(data[data[:, 2] == -1, 0],
             data[data[:, 2] == -1, 1], 'o', color='blue')
    plt.plot(x, -(w[0]*x + w[2])/w[1])
    plt.draw()
    time.sleep(0.5)
    return 0


# 特徴ベクトルの生成
def phi(x):
    return np.concatenate((x, [1]))


# イテレーション一回分
def perIT(data, N, w):
    list = range(N)
    random.shuffle(list)
    misses = 0  # 予測を外した回数
    for n in list:
        X_n = data[n, 0:2]
        t_n = data[n, 2]
        # 予測
        predict = np.sign((w * phi(X_n)).sum())
        # 予測が不正解なら，パラメータを更新する
        if predict != t_n:
            w += t_n * phi(X_n)
            misses += 1
    return w, misses


# パーセプトロン実行関数
def pers(data):
    N = np.size(data, axis=0)  # プロットの個数
    # データのプロット準備
    plt.ion()
    # xの最大値，最小値
    xmax = np.max(data[:, 0])
    xmin = np.min(data[:, 0])
    x = np.arange(xmin, xmax)
    plt.xlim(xmin+1, xmax-1)
    plt.ylim(xmin+1, xmax-1)
    # 重みベクトルの初期化
    w = np.ones(3)  # 3次の0ベクトル
    plotter(data, x, w)
    # イテレーション回数
    count = 0
    while True:
        # イテレーション一回分
        w, misses = perIT(data, N, w)
        # 境界線プロット
        plotter(data, x, w)
        # 予測が外れる点がなくなるまで実行
        if misses == 0:
            plt.ioff()
            break
        count += 1
    return w, count


# メイン関数
if __name__ == "__main__":
    # データ読み込み
    data = np.load('dataXYT.npy')
    N = np.size(data, axis=0)
    # パーセプトロン実行
    w, count = pers(data)
    # 結果出力
    print("分離面係数(a,b,c):\t%s\n実行回数:\t%s"
          % (w/np.linalg.norm(w), count))
