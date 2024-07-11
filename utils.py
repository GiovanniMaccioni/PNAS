import itertools
import torch
import numpy as np
import random

import scipy as sp
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt

def set_reproducibility(seed):
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    #torch.use_deterministic_algorithms(True)
    #torch.backends.cudnn.deterministic = True
    return

def flatten(list_of_lists):
    """
    Utility function to make the list at each step a list of tuples. Each elemnt will be a cell, each tuple a block.
    The "list_of_lists" come in as:
        list_of_lists
            |
            |___cell____
                        |__elem__
                        |        |___block
                        |        |
                        |        |___block
                        |
                        |__block
    
    I want
        list_of_lists
            |
            |___cell____
                        |___block
                        |
                        |___block
                        |
                        |___block    
    """
    x = []
    for elem in list_of_lists:
        tup = elem[0]+ elem[1]
        x.append(tup)
    return x

def get_combinations(inputs_indices, operations_indices):
    """
    Get all the possible cells. By passing to the function a sliced inputs_indices, i
    can get all the possible cell structures!!!
    """
    #Obtain the combination with replacement of input indices
    #e.g. 'ABC' ----> 'AA', 'AB', 'AC', 'BB', 'BC', 'CC'
    x = list(itertools.combinations_with_replacement(inputs_indices, 2))

    #Obtain the pairs excluded in the previous list
    #e.g. 'ABC' ----> 'BA', 'CA', 'CB'
    xx = list(itertools.product(inputs_indices, repeat=2))
    xx = [elem for elem in xx if elem not in x]
    #xx = [elem for elem in xx if (torch.equal(xx[0][0], x[0][0]) and torch.equal(xx[0][1], x[0][1]))]

    #Obtain the combination with replacement of operations indices
    #e.g. '123' ----> '11', '12', '13', '22', '23', '33' 
    y = list(itertools.combinations_with_replacement(operations_indices, 2))

    #Obtain the combination of operations indices
    #e.g. '123' ----> '12', '13', '23'
    yy = list(itertools.combinations(operations_indices, 2))

    #Obtain the cartesian product
    final1 = list(itertools.product(x, y))
    final1 = [[elem] for elem in final1]
    final2 = list(itertools.product(xx, yy))
    final2 = [[elem] for elem in final2]

    
    return final1 + final2

def expand_cells(inputs_indices, operations_indices, num_blocks = 1, prev_combinations = None):
    combinations = get_combinations(inputs_indices[:2+num_blocks-1], operations_indices)
    
    #If num_blocks > 1, we must have the list of previous valid combinations and make the cross product to 
    #obtain the new cell structures
    if num_blocks > 1:
        combinations = list(itertools.product(prev_combinations, combinations))
        if num_blocks > 1:
            combinations = flatten(combinations)
    
    return combinations

def cells_to_tensor(cells):
    tensor_cells = []
    for cell in cells:
        blocks = []
        for block in cell:
            inputs = torch.tensor(block[0]).view(-1)
            operations = torch.tensor(block[1]).view(-1)
            in_op = torch.cat((inputs, operations), dim=0)
            blocks.append(in_op[None, :])
        
        blocks = torch.cat(blocks)
        tensor_cells.append(blocks[None,:])

    tensor_cells = torch.cat(tensor_cells)
    
    return tensor_cells

def order_cells_and_accuracies(cells, accuracies, beam_size):

    def criteria(e):
        return e[1]
    
    new_list = sorted(zip(cells, accuracies.squeeze().tolist()), key=criteria, reverse=True)
    cells, accuracies = zip(*new_list)
    
    top_k_cells = cells[:len(cells) if len(cells) <  beam_size else beam_size]
    top_k_accuracies = accuracies[:len(cells) if len(cells) <  beam_size else beam_size]

    print("FINISHED SORTING")
    return top_k_cells, top_k_accuracies

def plot_correlationplot(x, y, x_label, y_label, dest):
    matplotlib.use('Agg')
    sns.set_theme(style="ticks")

    sns.scatterplot(x=x, y=y)

    plt.xlim([0,1])
    plt.ylim([0,1])

    plt.xlabel(x_label)
    plt.ylabel(y_label)

    r = sp.stats.spearmanr(x, y)
    ax = plt.gca()
    plt.text(.05, .9, "r ={:.2f}".format(r.statistic), transform=ax.transAxes)
    #m, b = np.polyfit(x, y, 1)
    X_plot = np.linspace(ax.get_xlim()[0],ax.get_xlim()[1],100)
    plt.plot(X_plot, 1*X_plot + 0, '-', c="red")

    plt.savefig(dest, bbox_inches='tight', dpi=300)
    plt.close()
    return