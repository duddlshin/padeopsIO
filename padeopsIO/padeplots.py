"""
Basic plotting functionality. 

Incomplete/deprecated 2025. 

Kirby Heck
c. 2022
"""

import matplotlib.pyplot as plt
import numpy as np

from . import budgetkey

# ----------- additional helper functions ------------

def common_cbar(fig, image, ax=None, location='right', label=None, height=1., width=0.02): 
    """
    Helper function to add a colorbar
    
    colorbar: https://stackoverflow.com/questions/18195758/set-matplotlib-colorbar-size-to-match-graph 
 
    parameters
    ----------
    fig : figure object to add a colorbar to
    image : AxesImage object or similar that has a color scale
    ax : (optional) Axes object to reference for size of colorbar
    location : string, # TODO
    height : height of colorbar as a fraction of the given axis's height (or selected, if no axis given). 
        Default 1
    width : width of colorbar as a fraction of the give axis's height
        Default 0.02
    """
    if ax is None:  # need to pick an axis from the figure...
        # ax = fig.axes[0]  
        # TODO: make this better... maybe choose the largest one? Maybe make a new one each time? ugh
        ax = common_axis(fig)
    
    h = ax.get_position().height  # axis height
    cax = fig.add_axes([ax.get_position().x1+h*width, 
                        ax.get_position().y0+h/2*(1-height), 
                        h*width, 
                        height*h])  # [start x, start y, width, height]
    cbar = fig.colorbar(image, cax=cax)
    
    if label is not None: 
        cbar.set_label(label)
    
    return cbar  # this probably will never be used


def common_axis(fig, xlabel=None, ylabel=None, title=None): 
    """
    Helper function to format common axes

    format common axes: 
    https://stackoverflow.com/questions/16150819/common-xlabel-ylabel-for-matplotlib-subplots
    """
    # create a new axis spanning the whole figure
    ax_new = fig.add_subplot(111, frameon=False)
    plt.tick_params(labelcolor='none', which='both', top=False, bottom=False, left=False, right=False)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    
    return ax_new  # return the newly created axis

