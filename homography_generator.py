import numpy as np
import pandas as pd

def homoGen(num=9625,random_t_homo=0.1):
    """
    :param num: # of theta
    :param random_t_homo: maximum translation ratio of 4points
    :return:
    """
    theta_list=list()
    for i in range(num):
        theta = np.array([0,0,1,1,0,1,0,1])
        theta = theta + (np.random.rand(8)-0.5)*2*random_t_homo
        theta = p2H(theta)
        theta_list.append(theta.squeeze())
    theta_list = np.float32(theta_list)
    print(theta_list)
    try:
        np.savetxt(fname='outputs/homo_theta.csv', X=theta_list, delimiter=',', header='H11,H12,tx,H21,H22,ty,H31,H32', fmt='%s', comments='')
        print('\nDone!')
    except:
        print('Occurred error!')

def p2H(p_B):
    p_A = [0,0,1,1,0,1,0,1]
    P = list()
    for i in range(4):
        x = p_A[i]
        y = p_A[i + 4]
        x_prime = p_B[i]
        y_prime = p_B[i + 4]

        P.append([x, y, 1, 0, 0, 0, -x * x_prime, -y * x_prime])
    for i in range(4):
        x = p_A[i]
        y = p_A[i + 4]
        x_prime = p_B[i]
        y_prime = p_B[i + 4]

        P.append([0, 0, 0, x, y, 1, -x * y_prime, -y * y_prime])
    M1 = np.float32(P)
    M2 = np.expand_dims(np.float32(p_B),1)

    H = np.matmul(np.linalg.inv(M1), M2)
    return H.squeeze()

def addRot(num=9625, random_alpha=1/2):
    csv = pd.read_csv('outputs/homo_theta.csv')
    theta = csv.values.astype('float')

    # Generate random rotation
    rot_theta = list()
    for i in range(num):
        alpha = (np.random.rand(1)) * 2 * np.pi * random_alpha
        rot = np.float32([[np.cos(alpha),(-np.sin(alpha)),0],[np.sin(alpha),np.cos(alpha),0],[0,0,1]])
        rot_theta.append(rot)
    rot_theta=np.array(rot_theta)

    # Multiply rotation
    theta = theta[:num]
    theta = np.concatenate((theta,np.tile(np.array([[1]]),[num,1])),1)
    theta = theta.reshape([-1,3,3])

    result = np.matmul(rot_theta,theta)
    result = result.reshape([num,-1])
    result = result[:,:8]
    print(result.shape)
    print(result)

    np.savetxt(fname='outputs/refined_homo.csv', X=result, delimiter=',', header='H11,H12,tx,H21,H22,ty,H31,H32', fmt='%f', comments='')

addRot()