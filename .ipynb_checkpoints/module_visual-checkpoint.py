import numpy as np
import pandas as pd
import module_grid
import matplotlib.pyplot as plt

def pixelplot(grid,values):
    '''
    Return an estimate plot.
    
    :param grid: a dictionary which contains the grid used for kriging.
    
    :param values: an array of the kriging values.
    
    '''
    values = values.reshape(grid['ny'],grid['nx'],order='F')
    plt.figure(figsize=(8,8))
    plt.imshow(values, origin = 'lower')
    plt.colorbar()
    plt.show()
    

