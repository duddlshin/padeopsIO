"""
Includes the Slice object and slicing functions
for handling arrays. 

Kirby Heck
2024 July 24
"""

import numpy as np
import matplotlib.pyplot as plt
import warnings


class Grid3:
    """Grid of evenly spaced 3D data, supporting length-1 dimensions"""

    def __init__(self, x=None, y=None, z=None):

        self.x, self.nx = self._process_dimension(x, axis=0)
        self.y, self.ny = self._process_dimension(y, axis=1)
        self.z, self.nz = self._process_dimension(z, axis=2)

        self.dx = self._compute_spacing(self.x, self.nx)
        self.dy = self._compute_spacing(self.y, self.ny)
        self.dz = self._compute_spacing(self.z, self.nz)

        self.Lx = self._compute_length(self.x)
        self.Ly = self._compute_length(self.y)
        self.Lz = self._compute_length(self.z)

        self.hasx = self.nx > 1
        self.hasy = self.ny > 1
        self.hasz = self.nz > 1

        self.label_key = dict(x="$x/D$", y="$y/D$", z="$z/D$")

        self.index_axes()  # make the following properties index-able

    def _process_dimension(self, x, axis):
        """
        Helper function to process each dimension and handle edge cases.

        Returns
        -------
        arr, len
        """
        arr = np.asarray(x)  # cast to an array

        if arr.size == 0:
            raise ValueError(f"Axis {axis} cannot be an empty array")
        elif x is None:
            return arr, 0
        elif arr.ndim == 0:
            # single value
            return arr, 1
        elif arr.ndim == 1:
            # line of values
            return arr, len(arr)
        elif arr.ndim == 3:
            # gridded data
            arr_line = arr.take(0, axis=axis)
            if not np.allclose(arr_line, np.take(arr, slice(None), axis=axis)):
                raise ValueError(f"Axes must be equal along axis {axis}")
            return arr_line, len(arr_line)
        else:
            raise ValueError(f"Axis {axis} can only have 1 or 3 dimensions")

    def _compute_spacing(self, arr, n):
        """Compute the spacing, handling the special case of n=1."""
        if n > 1:
            return arr[1] - arr[0]
        else:
            return np.nan

    def _compute_length(self, arr):
        """Compute the length of the array."""
        if arr.ndim > 0:
            return arr[-1] - arr[0]
        else:
            return 0

    def index_axes(self):
        # these labels exist:
        self.labels = tuple(xi for xi in ["x", "y", "z"] if getattr(self, f"has{xi}"))

        # keep track of which axes this corresponds with (0, 1, 2) <=> (x, y, z)
        self.ids_exist = tuple(
            k for k, xi in enumerate(["x", "y", "z"]) if getattr(self, f"has{xi}")
        )

        # now index Lx, dx, x, and a matching key
        self.Lxi = tuple(getattr(self, f"L{xi}") for xi in self.labels)
        self.dxi = tuple(getattr(self, f"d{xi}") for xi in self.labels)
        self.xi = tuple(getattr(self, xi) for xi in self.labels)
        self.axes_index = {k: xi for k, xi in enumerate(self.labels)}

    def get_axes(self, named=True):
        """Returns axes with length > 1"""
        if named:  # returns a dictionary
            return {l: ax for l, ax in zip(self.labels, self.xi)}
        else:  # returns a list
            if self.ndim == 1:
                return self.xi[0]  # don't return as a length-1 array
            else:
                return self.xi

    def todict(self):
        return dict(x=self.x, y=self.y, z=self.z)

    def __repr__(self):
        return repr(self.todict())

    @property
    def shape(self):
        """Return the shape of the grid. Drops singleton dimensions."""
        return tuple(nxi for nxi in (self.nx, self.ny, self.nz) if nxi > 1)

    @property
    def ndim(self):
        """No. dimensions, drops singleton dimensions"""
        return sum(~np.isnan([self.dx, self.dy, self.dz]))

    @property
    def extent(self):
        """Build extents of non-singleton axes"""
        ret = []
        for xi in [self.x, self.y, self.z]:
            if xi.ndim > 0:
                ret += [xi.min(), xi.max()]
        return ret


class Slice:
    """
    Slice class for "wrapping" dictionaries of numpy arrays.

    Slices can be used for analysis and visualization, and contain
    SliceData objects, which all share the same grid and are
    organized by keys in the Slice object.


    """

    def __init__(
        self, *args, x=None, y=None, z=None, grid=None, strict_shape=True, **kwargs
    ):
        """
        Initialize with another slice object (args)
        """
        if len(args) > 0:
            if isinstance(args[0], Slice):
                self.data = args[0].data
                self.grid = args[0].grid
                return
            elif isinstance(args[0], dict):
                self.data = args[0]
            else:
                self.data = dict(field=args[0])
        else:
            self.data = kwargs  # assume all keyword arguments are fields

        # load a grid
        self.grid = grid or Grid3(x=x, y=y, z=z)

        self.cast_data(strict_shape)  # cast data to SliceData objects

    def cast_data(self, strict_shape=True):
        """Cast arrays to SliceData objects. Checks that grid sizes match"""
        ret = dict()
        for key, arr in self.data.items():
            ret[key] = self.cast_array(arr, key, strict_shape=strict_shape)
        self.data = ret

    def cast_array(self, arr, name, strict_shape=True):
        """Cast array `arr` to a SliceData object"""
        return SliceData(arr, self.grid, name=name, strict_shape=strict_shape)

    def slice(self, xlim=None, ylim=None, zlim=None, keys=None):
        """Create a new Slice object with data sliced along specified axes"""
        # somehow, get_xids works wtih the (None, ) axes
        xi_ids = get_xids(
            x=self.grid.x,
            y=self.grid.y,
            z=self.grid.z,
            xlim=xlim,
            ylim=ylim,
            zlim=zlim,
            return_slice=True,
            return_none=True,
        )

        # prepare slice args for the dimensions that exist
        slice_args = tuple(
            _ids for k, _ids in enumerate(xi_ids) if k in self.grid.ids_exist
        )

        # Filter keys if specified
        keys = keys or self.keys()

        # now build data in the new Slice()
        newdata = {key: self.data[key][slice_args] for key in keys}

        # Adjust the grid for the sliced dimensions
        newaxes = {
            label: getattr(self.grid, label)[slice_arg]  # finds self.x, y, or z
            for label, slice_arg in zip(self.grid.labels, slice_args)
        }

        return Slice(newdata, **newaxes)

    def keys(self):
        return self.data.keys()

    def extent(self):
        return self.grid.extent

    def ndim(self):
        return self.grid.ndim

    def __getitem__(self, key):
        if key == "extent":
            warnings.warn(
                "Deprecation warning: use Slice.extent instead of Slice['extent']"
            )
            return self.grid.extent
        elif key == "keys":
            warnings.warn(
                "Deprecation warning: use Slice.keys() instead of Slice['keys']"
            )
            return self.keys()
        elif key in ["x", "y", "z"]:
            return getattr(self.grid, key)
        return self.data[key]

    def __setitem__(self, key, arr):
        self.data[key] = self.cast_array(arr, key)  # may throw ValueError

    def __repr__(self):
        return repr(self.data)


class SliceData(np.ndarray):

    def __new__(cls, arr, grid, name=None, strict_shape=True):
        # initialize numpy array object
        obj = np.asarray(arr).view(cls)

        obj.name = name
        obj.grid = grid
        obj.strict_shape = strict_shape
        return obj

    def __array_finalize__(self, obj):
        if obj is None:
            return

        self.name = getattr(obj, "name", None)
        self.grid = getattr(obj, "grid", None)
        self.strict_shape = getattr(obj, "strict_shape", True)

    def __init__(self, *args, **kwargs):
        """Check array and grid shape match"""

        if self.shape != self.grid.shape:
            if self.shape[:3] == self.grid.shape and not self.strict_shape:
                # relaxed restrictions on grid shape to allow tensors
                pass
            else:
                raise ValueError(
                    f"Mismatch in shapes for '{self.name}': {str(self.shape)}, {str(self.grid.shape)}"
                )

    def __getitem__(self, key):
        result = super().__getitem__(key)
        # when slicing via indexing, need to also update grid
        if isinstance(result, SliceData):
            new_grid = self._adjust_grid(key)
            result.grid = new_grid
        return result

    def _adjust_grid(self, keys):
        """Slicing via indexing requires adjusting the grid"""
        if not isinstance(keys, tuple):
            keys = (keys,)

        # Ellipsis handling is the tough part
        if keys.count(Ellipsis) == 1:
            ellipsis_index = keys.index(Ellipsis)
            num_full_slices = self.grid.ndim - len(keys) + 1
            # pad keys with slice(None)
            keys = (
                keys[:ellipsis_index]
                + (slice(None),) * num_full_slices
                + keys[ellipsis_index + 1 :]
            )
        
        # create new axes
        new_axes = {
            label: getattr(self.grid, label)[slice_arg]
            for label, slice_arg in zip(self.grid.labels, keys)
        }

        return Grid3(**new_axes)

    def alias(self, name):
        self.name = name

    def slice(self, xlim=None, ylim=None, zlim=None):
        """Create a new SliceData object with data sliced along specified axes"""
        # somehow, get_xids works wtih the (None, ) axes
        xi_ids = get_xids(
            x=self.grid.x,
            y=self.grid.y,
            z=self.grid.z,
            xlim=xlim,
            ylim=ylim,
            zlim=zlim,
            return_slice=True,
            return_none=True,
        )

        # prepare slice args for the dimensions that exist
        slice_args = tuple(
            _ids for k, _ids in enumerate(xi_ids) if k in self.grid.ids_exist
        )

        # now build data in the new Slice()
        newdata = self[slice_args]

        # Adjust the grid for the sliced dimensions
        newaxes = {
            label: getattr(self.grid, label)[slice_arg]  # finds self.x, y, or z
            for label, slice_arg in zip(self.grid.labels, slice_args)
        }

        return SliceData(newdata, Grid3(**newaxes), name=self.name)

    def plot(self, ax=None, **kwargs):
        """Visualies the data"""
        if ax is None:
            fig, ax = plt.subplots()

        if self.grid.ndim < 1:
            raise ValueError("Cannot visualize 0D data")
        elif self.grid.ndim < 2:
            _ax = self.grid.get_axes(named=False)
            ax.set_xlabel(self.grid.label_key[self.grid.labels[0]])
            return ax.plot(_ax, self, **kwargs)

        elif self.grid.ndim < 3:
            ax.set_xlabel(self.grid.label_key[self.grid.labels[0]])
            ax.set_ylabel(self.grid.label_key[self.grid.labels[1]])
            return ax.imshow(self.T, origin="lower", extent=self.grid.extent, **kwargs)

        else:
            raise ValueError("Cannot visualize 3D data")


# ============== helper functions =================


def get_xids(
    x=None,
    y=None,
    z=None,
    xlim=None,
    ylim=None,
    zlim=None,
    return_none=False,
    return_slice=False,
):
    """
    Translates x, y, and z limits in the physical domain to indices
    based on axes x, y, z

    Parameters
    ---------
    x, y, z : 1d array
        Axes
    xlim, ylim, zlim : float or iterable (tuple, list, etc.)
        Physical locations to return the nearest index
    return_none : bool
        If True, populates output tuple with None if input is None. Default False.
    return_slice : bool
        If True, returns a tuple of slices instead a tuple of lists. Default False.

    Returns
    -------
    xid, yid, zid : list or tuple of lists
        Indices for the requested x, y, z, args in the order: x, y, z.
        If, for example, y and z are requested, then the returned tuple
        will have (yid, zid) lists. If only one value (float or int) is
        passed in for e.g. x, then an integer will be passed back in xid.
    """

    ret = ()

    # iterate through x, y, z, index matching for each term
    for s, s_ax in zip([xlim, ylim, zlim], [x, y, z]):
        if s is not None:
            if s_ax is None:
                raise AttributeError("Axis keyword not provided")

            if hasattr(s, "__iter__"):
                xids = [np.argmin(np.abs(s_ax - xval)) for xval in s]
            else:
                xids = np.argmin(np.abs(s_ax - s))

            xids = np.squeeze(np.unique(xids))

            if return_slice and xids.ndim > 0:  # append slices to the return tuple
                ret = ret + (slice(np.min(xids), np.max(xids) + 1),)

            else:  # append index list to the return tuple
                ret = ret + (xids,)

        elif return_none:  # fill with None or slice(None)
            if return_slice:
                ret = ret + (slice(None),)

            else:
                ret = ret + (None,)

    if len(ret) == 1:
        return ret[0]  # don't return a length one tuple
    else:
        return ret


if __name__ == "__main__":
    """Simple unit tests"""

    # ==== grid unit tests ====
    x = np.linspace(0, 1, 11)
    grid = Grid3(x, None, 1)  # None axes OK; empty axes not ok

    # a = SliceData(x, grid)
    # print(a.grid.shape)
    # print(a.name)
    # print(a + a)

    assert grid.ndim == 1
    assert np.allclose(grid.x, x)
    assert grid.shape == (11,)
    assert grid.extent == [0, 1]

    data = np.random.rand(*grid.shape)
    a = Slice(data, grid=grid)
    assert isinstance(a, Slice)
    a_slice = a.slice(xlim=[0.4, 1])
    assert isinstance(a_slice, Slice)

    print("all tests passed")
