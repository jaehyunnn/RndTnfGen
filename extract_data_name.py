import os
import glob
import pandas
import numpy as np

"""
def allfiles(path):
    res = []

    for root, dirs, files in os.walk(path):
        rootpath = os.path.join(os.path.abspath(path), root)

        for file in files:
            filepath = os.path.join(rootpath, file)
            res.append(filepath)

    return res

list = allfiles('/home/jhpark/Desktop/dataset/matching_datasets/GoogleEarth/current/TrainVal/')
print(len(list))
"""



current =  ['GoogleEarth/current/TrainVal/'+os.path.basename(x) for x in glob.glob('/home/add/Desktop/git/Aerial-Bi-A2Net/datasets/GoogleEarth/current/TrainVal/*.jpg')]
past = ['GoogleEarth/past/TrainVal/'+os.path.basename(x) for x in glob.glob('/home/add/Desktop/git/Aerial-Bi-A2Net/datasets/GoogleEarth/past/TrainVal/*.jpg')]

current = np.expand_dims(np.array(current),1)
past = np.expand_dims(np.array(past),1)

csv_file = np.concatenate([current,past],1)
try:
    np.savetxt(fname='outputs/train_pair.csv', X=csv_file, delimiter=',', header='Source(current), Target(past)',  fmt='%s')
    print(csv_file)
    print('\nDone!')
except:
    print('Occured error')