import numpy as np
import itertools
import math



def add_coords(grid):
    '''
        
    :param grid: dictionary containing the origins of a grid, number of cells and size of cells. 
       
    :return: numpy ndarray dtype = int64. Each row of array represents a coordinate from the grid.
    
    '''
    xo = grid['xo']
    yo = grid['yo']
    nx = grid['nx']
    ny = grid['ny']
    sx = grid['sx']
    sy = grid['sy']
    
    x_coord = np.arange(xo, xo+(nx*sx), sx)
    y_coord = np.arange(yo, yo+(ny*sy), sy)

    coords_array = []
    
    
    for x, y in itertools.product(x_coord, y_coord):
        coords_array.append([x,y])

    return np.array(coords_array)


def auto_grid(x, y, sx, sy):
    '''
        
    :param x: X values from the dataset.
    
    :param y: Y values from the dataset. 
    
    :param sx: X size of the grid cell to be used. 
    
    :param sy: Y size of the grid cell to be used. 
       
    :return: grid dictionary containing the origins of a grid, number of cells and size of cells. 
    
    '''
    nx = int((x.max()-x.min())/sx)
    ny = int((y.max()-y.min())/sy)
    xo = x.min()
    yo = y.min()
    
    return {'xo':xo, 'yo':yo, 'nx':nx,'ny':ny, 'sx':sx, 'sy':sy}


