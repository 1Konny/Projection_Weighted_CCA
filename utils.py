import numpy as np


def standardize(array):
    """Standardize each row of array to zero mean and unit standard deviation.
    
    Args:
        array(numpy.ndarray): A numpy 2D array of shape (num_variables, num_datapoints)
        
    Returns:
        standardized_array(numpy.ndarray): variable-wise(row-wise) standardized array.
    """
    assert array.ndim == 2, 'array should be 2D of shape (num_variables, num_datapoints)'
    mean = array.mean(1, keepdims=True)
    std = array.std(1, keepdims=True)
    standardized_array = (array-mean)/(std)
    
    return standardized_array