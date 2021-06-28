import numpy as np
import math
import matplotlib.pyplot as plt
import pandas as pd
import itertools
from scipy.interpolate import Rbf
from scipy.spatial import KDTree
from scipy.spatial import distance
from scipy import stats


def df_from_gslib(file_name):
    '''
        
    :param file_name: name of the GSLib format file dataset to be converted.
              
    :return: Pandas DataFrame of the dataset. 
    
    '''
      
    list_of_names = []
    with open(file_name, 'r') as f:
        f.readline()  # skip first line
        n_var = int(f.readline().split()[0])

        for i in range(n_var):
            string_name = f.readline().split()[0]
            list_of_names.append(string_name)

    n_lines_header = n_var + 2

    df = pd.read_csv(filepath_or_buffer=file_name, header=None, names = list_of_names, skiprows=n_lines_header,  delim_whitespace=True)

    return df


def covariance(h,a):
     
    return np.exp(-(h/a)**2)


def cov_matrix(cov):
    '''
        
    :param cov: Covariance model chosen.
              
    :return C: Covariance Matrix. 
    
    '''
     
    cov_shape = cov.shape
    C_shape = (cov.shape[0]+1, cov.shape[1]+1)
    C = np.ones(C_shape)
    C[C_shape[0]-1, C_shape[1]-1] = 0
    C[:-1,:-1] = cov
    return C

def krig_val(grade, weight):
    '''
        
    :param grade: Array of grade values.

    :param weight: Array of kriging weight values.
    
    :return res_list : Array of values from the multiplication of grade and kriging weight. 
    
    '''    
    res_list = [grade[i] * weight[i] for i in range(len(grade))] 
    return sum(res_list)


def kriging(x_sample,y_sample,coord_node_grid, n_samples, a):
    '''
        
    :param x_sample: X value of a sample.

    :param y_sample: Y value of a sample.
    
    :param coord_node_grid: Array of coordinates from the built grid.

    :param n_samples: Number of samples to be searched.

    :param a: Contribuition value to be used.

    :return kriging_values : kriging value. 
    
    '''    
    samples_coords = np.vstack((x_sample,y_sample)).T
    samples_kdtree = KDTree(samples_coords)
    neighs = samples_kdtree.query(coord_node_grid,n_samples)
    samples2=samples_coords[neighs[1]]
    dist = distance.cdist(samples2,samples2)
    C = covariance(dist,a)
    D = covariance(neighs[0],a)
    C1 = cov_matrix(C)
    D1 = np.append(D, 1)
    C_inv = np.linalg.inv(C1)
    W = C_inv @ D1
    W2 =W[:-1]
    kriging_values = krig_val(neighs[1],W2)
        
    return kriging_values


def all_kriging(X, Y,coord_node_grid, n_samples, a):
    '''
        
    :param X: X value of a sample.

    :param Y: Y value of a sample.
    
    :param coord_node_grid: Array of coordinates from the built grid.

    :param n_samples: Number of samples to be searched.

    :param a: Contribuition value to be used.

    :return all_block : all kriged block values. 
    
    '''    
    
    
    all_krig = [kriging(X, Y, coord_node_grid[i], n_samples, a) for i in range(len(coord_node_grid))]
    
    all_block =  np.array(all_krig)
    
    return all_block