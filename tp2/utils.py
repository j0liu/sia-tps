import numpy as np
import random

def normalize(array, module):
    new_array = array/sum(array) * module

    # Version for integer values
    # new_array = np.round(array/sum(array) * module).astype(int)
    # remainder = module - sum(new_array)

    # for _ in range(abs(remainder)):
    #     index = random.randint(0, len(new_array) - 1)
    #     new_array[index] += 1 if remainder > 0 else -1
        
    return new_array