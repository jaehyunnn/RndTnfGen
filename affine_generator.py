import numpy as np
import pandas as pd

def affGen(num=9625,random_t=0.5, random_s=0.5, random_alpha=1/2):
    """
    :param num: # of theta
    :param random_t: maximum translation ratio
    :param random_s:  maximum scaling ratio
    :param random_alpha: maximum rotation radian; 1/2 = 180 degrees
    :return:
    """
    theta_list=list()
    for i in range(num):
        alpha = (np.random.rand(1) - 0.5) * 2 * np.pi * random_alpha
        theta = np.random.rand(6)
        theta[[2,5]] = (theta[[2,5]] - 0.5) * 2 * random_t
        theta[0] = (1 + (theta[0] - 0.5) * 2 * random_s) * np.cos(alpha)
        theta[1] = (1 + (theta[1] - 0.5) * 2 * random_s) * (-np.sin(alpha))
        theta[3] = (1 + (theta[3] - 0.5) * 2 * random_s) * np.sin(alpha)
        theta[4] = (1 + (theta[4] - 0.5) * 2 * random_s) * np.cos(alpha)
        theta_list.append(theta)
    theta_list = np.array(theta_list)
    print(theta_list)

    try:
        np.savetxt(fname='oututputs/aff_theta.csv', X=theta_list, delimiter=',', header='A11,A12,tx,A21,A22,ty', fmt='%s', comments='')
        print('\nDone!')
    except:
        print('Occurred error!')

def addRot(num=9625, random_alpha=1/2):
    csv = pd.read_csv('outputs/train.csv')
    theta = csv.iloc[:, 1:].values.astype('float')
    theta = theta[:,[3,2,5,1,0,4]]
    # Generate random rotation
    rot_theta = list()
    for i in range(num):
        alpha = (np.random.rand(1)) * 2 * np.pi * random_alpha
        rot = np.float32([[np.cos(alpha),(-np.sin(alpha)),0],[np.sin(alpha),np.cos(alpha),0],[0,0,1]])
        rot_theta.append(rot)
    rot_theta=np.array(rot_theta)

    # Multiply rotation
    theta = theta[:num]
    theta = np.concatenate((theta,np.tile(np.array([[0,0,1]]),[num,1])),1)
    theta = theta.reshape([-1,3,3])

    result = np.matmul(rot_theta,theta)
    result = result.reshape([num,-1])
    result = result[:,:6]
    print(result.shape)
    print(result)

    np.savetxt(fname='outputs/refined_aff.csv', X=result, delimiter=',', header='A11,A12,tx,A21,A22,ty', fmt='%f', comments='')

addRot()

