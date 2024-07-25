"""
Math utility functions. 

Kirby Heck
2024 July 24
"""


import numpy as np


def assemble_tensor_1d(field_dict, keys, axis=-1): 
    """
    Stacks keys in a 1d tensor in the order they are given. 

    Parameters
    ----------
    field_dict : dict
        Dictionary of fields (e.g. {'ubar': <np.ndarray>, 'vbar': <np.ndarray>, 'wbar': <np.ndarray>})
    keys : list
        (Length-3) list of keys
    axis : int, optional
        Axis to stack. Default: -1
    
    Returns
    -------
    np.ndarray : (Nx, Ny, Nz, len(keys))
    """

    return np.stack([field_dict[key] for key in keys], axis=axis)


def assemble_tensor_nd(field_dict, keys): 
    """
    Assembles an n-dimensional tensor from a dictionary of fields. 

    Parameters
    ----------
    field_dict : dict
        Dictionary of fields 
        (e.g. {'ubar': <np.ndarray>, 'vbar': <np.ndarray>, 'wbar': <np.ndarray>})
    keys : list of lists
        (Length-3) list of lists (of lists of lists...) of keys
    basedim : int, optional
        Number of base dimensions, default 3 (x, y, z). 
    
    Returns
    -------
    np.ndarray
    """

    def _assemble_nd(keys): 
        """Recursive helper function"""
        if isinstance(keys[0], str):  # TODO: what if the key is not a string? 
            # base case, prepend stacks
            return assemble_tensor_1d(field_dict, keys, axis=0)  
        else: 
            # recursive call: return stack [of stacks], also prepended
            return np.stack([_assemble_nd(key) for key in keys], axis=0)  
    
    # use the recursive calls, which prepends each added index (e.g. [i, j, ..., x, y, z])
    tensors_rev = _assemble_nd(keys)

    # keys is a list or a list of nested lists
    key_ls = list(field_dict.keys())
    try: 
        ndim = field_dict[key_ls[0]].ndim
    except AttributeError: 
        return tensors_rev  # dictionary fields are not arrays
    
    return np.moveaxis(tensors_rev, range(-ndim, 0), range(ndim))


# ==================== index notation help ==========================


def e_ijk(i, j, k): 
    """
    Permutation operator, takes i, j, k in [1, 2, 3] 
    
    returns +1 if even permutation, -1 if odd permutation
    
    TODO: This is not elegant
    """
    
    if (i, j, k) in [(0, 1, 2), (1, 2, 0), (2, 0, 1)]: 
        return 1
    elif (i, j, k) in [(0, 2, 1), (1, 0, 2), (2, 1, 0)]: 
        return -1
    else: 
        return 0
    

# also E_ijk in tensor form: 
E_ijk = np.array([[[e_ijk(i, j, k)
                    for k in range(3)]
                   for j in range(3)]
                  for i in range(3)])


def d_ij(i, j): 
    """
    Kronecker delta
    """
    return int(i==j)


class DerOps(): 
    """
    Derivative operations

    # TODO - finish class and standardize 
    """

    def __init__(self, 
                 dx=(1, 1, 1), 
                 order=2, edge_order=2, 
                 periodic_x=False, periodic_y=False, periodic_z=False, 
                 pade_z=False): 
        
        self.order = order
        self.edge_order = edge_order
        self.periodic_x = periodic_x
        self.periodic_y = periodic_y
        self.periodic_z = periodic_z
        self.pade_z = pade_z
        self.dx = dx
        
    def gradient(self, f, axis=(0, 1, 2), stack=-1): 
        """
        Compute gradients of f
        """
        
        if hasattr(axis, '__iter__'): 
            args = [self.dx[k] for k in axis]
        else: 
            args = [self.dx[axis]]

        # TODO: fix gradient functions everywhere
        dfdxi = np.gradient(f, *args, axis=axis, edge_order=self.edge_order)

        if len(args) > 1: 
            return np.stack(dfdxi, axis=stack)
        return dfdxi


    def div(self, f, axis=-1, sum=False): 
        """
        Computes the 3D divergence of vector or tensor field f: dfi/dxi

        Parameters
        ----------
        f : (Nx, Ny, Nz, 3) or (Nx, Ny, Nz, ...) array
            Vector or tensor field f 
        axis : int, optional
            Axis to compute divergence, default -1 
            (Requires that f.shape[axis] = 3)
        sum : bool, optional
            if True, performs implicit summation over repeated indices. 
            Default False

        Returns
        -------
        dfi/dxi : f.shape array (if sum=True) OR drops the `axis` 
            axis if sum=False
        """

        res = np.zeros(f.shape)

        def get_slice(ndim, axis, index): 
            """
            Helper function to slice axis `axis` from ndarray 
            """
            s = [slice(None) for i in range(ndim)]
            s[axis] = slice(index, index+1)
            return tuple(s)

        # compute partial derivatives: 
        for i in range(3): 
            s = get_slice(f.ndim, axis, i)

            # TODO fix gradient functions everywhere
            res[s] = np.gradient(f[s], self.dx[i], axis=i, edge_order=self.edge_order)

        if sum: 
            return np.sum(res, axis=axis)
        else: 
            return res


if __name__=='__main__': 
    # run basic tests: 
    tmp = np.reshape(np.arange(24), (2,3,4))
    der = DerOps()

    # field = {'111': 111, '112': 112, '121': 121, '122': 122, '211': 211, '212': 212, '221': 221, '222': 222}
    # keys_test = [[['111', '112'], ['121', '122']], [['211', '212'], ['221', '222']]]
    field = {
        str(k): np.ones((3, 4)) * k for k in range(8)
    }
    keys_test = [[['0', '1',], ['2', '3']], [['4', '5'], ['6', '7']]]
    ret = assemble_tensor_nd(field, keys_test)

    print(ret[..., 0, 1, 1])
