FlatMap
=======

Description
-----------

The `flatmap` is a mapping from discrete voxels in a 3D space to discrete pixels in a 2D plane, or `canvas`.
Since the mapping is discrete, the values are `np.int`.
For a `LxMxN` voxelized space, the `flatmap` has shape (L, M, N, 2), where the last dimension is the x, y pixel on the `canvas`.
The 2D canvas is only in the positive quadrant.
Thus, negative values of either the x or y mean that the voxel does not have a valid value.
The file format used to store this data is `NRRD`_.
It is recommended that one uses `voxcell` for handling the loading, saving and accessing voxelized data.


Example
-------

Consider a hyper-rectangle of shape 7, 11, 13.
A `flatmap` for this shape could be the collapsed version along the `y-axis`.
To create the `NRRD` file for this:

.. code-block:: python

    >>> import numpy as np
    >>> import voxcell

    >>> one_layer_flatmap = np.mgrid[:13, :7].T[:, :, ::-1]
    >>> flatmap = voxcell.VoxelData(np.stack([one_layer_flatmap] * 11, axis=1),
                                    voxel_dimensions=(10., 10., 10.))

    # now, all voxels along tye 'y-axis' have the same mapping to the canvas
    >>> np.all(flatmap.raw[3, :, 10] == [3, 10])
    True

    # save it
    >>> flatmap.save_nrrd("flatmap_7x11x13.nrrd")

.. _NRRD: http://teem.sourceforge.net/nrrd/format.html
