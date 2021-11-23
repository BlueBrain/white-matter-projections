Concepts
========

Density Compensation
--------------------

As discussed in `BBPP82-62 <https://bbpteam.epfl.ch/project/issues/browse/BBPP82-62>`_

Suppose a projection has an average target density of ``x`` um^-3 synapses and the projection targets only 50% of the target region.
The resulting density will be ``x`` um^-3 in the half that is targeted and 0 in the half that is not targeted.
Thus, an average density of only x/2 um^-3 is attained in the region.
Hence, ``compensation`` is used to correct the target density.

The procedure works like:

1. Sample all voxels in the source region; convert their position to the 2D flatmap, map them to the target location based on the projection.
   This is the source point list, call it ``src``.
2. For a sample of neurons in the target region, convert their locations to the 2D flatmap.
   This is the target point list, call it ``tgt``.
3. The ratio of ``| tgt | / || src - tgt || < sigma`` is the ``strengthening`` or ``compensation`` multiplier needed to correct the density.
