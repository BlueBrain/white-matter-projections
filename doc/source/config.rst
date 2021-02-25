Config
======

Keys
----

projections_recipe
~~~~~~~~~~~~~~~~~~

The path to the recipe.
Non-absolute paths are relative to the location of the config.yaml file.

.. code-block:: yaml

    projections_recipe: rat_wm_recipe_tr_ll_ul_un_n_m_v2-flipped.yaml

circuit_config
~~~~~~~~~~~~~~

Location of the CircuitConfig to be used to determine the cells for which the whitematter is created.

.. code-block:: yaml

    circuit_config: /gpfs/bbp.cscs.ch/project/proj83/circuits/Bio_M/20200805/CircuitConfig

atlas_url
~~~~~~~~~

The atlas that is used for sampling and determining the region of different cells and synapse locations.

.. code-block:: yaml

    atlas_url: /gpfs/bbp.cscs.ch/project/proj83/data/atlas/S1/MEAN/P14-MEAN

indices
~~~~~~~

Location of the FlatIndex index files used to find the position of the segments.

.. code-block:: yaml

    indices: /gpfs/bbp.cscs.ch/project/proj83/circuits/Bio_M/20200805/


cache_dir
~~~~~~~~~

Location of cache directory, when not absolute, it is relative to where the command was executed.
This directory stores a faster to load recipe, and other information that is heavy to calculate, but used multiple times.

.. code-block:: yaml

    cache_dir: .cache-v2

delay_method
~~~~~~~~~~~~

Method for determining the distance along which the 'white-matter' portion of the connection is calculated.
There are three options: `streamlines`, `direct`, `dive`:

* direct: directly connect the source and target locations, as the crow flies
* dive: Drop to the lower-most layer on both source and target, and connect those points as the crow flies
* streamlines: use the `AIBS streamlines`_ to create a path

.. code-block:: yaml

    delay_method: dive

conduction_velocity
~~~~~~~~~~~~~~~~~~~

The values used for the calculation of delay in um/s.
`intra_region` is the non-myelinated portion
`inter_region` is the myelinated portion

.. code-block:: yaml

    conduction_velocity:
        intra_region: 300
        inter_region: 3500

module_grouping
~~~~~~~~~~~~~~~

A module is composed of different regions, this defines their grouping.
This is a nested list of [Module, [region1, region2, region3]].
Note: These are lists to keep the order

.. code-block:: yaml

    module_grouping:
        - [SSp-tr, [S1Tr]]
        - [SSp-ll, [S1HL]]
        - [SSp-ul, [S1FL, S1Sh]]
        - [SSp-un, [S1DZ]]
        - [SSp-n, [S1ULp]]
        - [SSp-m, [S1J, S1DZO]]

ignored_regions
~~~~~~~~~~~~~~~

List of regions that are ignored, and thus shouldn't throw errors if they exist in the recipe.

.. code-block:: yaml

    ignored_regions: [ACAd4, ACAv4, AId4, AIp4, AIv4, ECT4, FRP4, ILA4, MOp4, MOs4,
                      ORBl4, ORBm4, ORBvl4, PERI4, PL4, RSPagl4, RSPv4,
                      ]

region_subregion_translation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

To adapt/keep compatible recipes to various hierarchies; a series of translation and regex based extraction keys are available:

* region_subregion_format: Regex used to convert from recipe region and subregion names, to the ones contained in the Atlas hierarchy.
  Note the use of the regex matching in `Voxcell`, ex; the starting `@`
  A subregion is generally the name of the layers.
* region_subregion_extract_format: python style regex to extract the region and subregion from an acronym
* subregion_translation: Dictionary lookup to convert the subregion names to those used for the lookup within the atlas.

.. code-block:: yaml

    region_subregion_translation:
        region_subregion_format: '@{region}{subregion}'
        region_subregion_separation_format: '(?P<region>.*)(?P<subregion>\d+)'
        subregion_translation: {l1: '1', l2: '2', l3: '3', l4: '4', l5: '5', l6: '6', }

Or:

.. code-block:: yaml

    region_subregion_translation:
        region_subregion_format: '@{region};L{subregion}'
        region_subregion_separation_format: '(?P<region>.*);L(?P<region>\d+))'



populations_filters
~~~~~~~~~~~~~~~~~~~

The recipe has 'filters' in the population stanzas to narrow down the m-types to be used, this defines them.

.. code-block:: yaml

    populations_filters:
        EXC: ['L2_IPC', 'L6_IPC',
              'L2_TPC:A', 'L2_TPC:B', 'L3_TPC:A', 'L3_TPC:C',
              'L4_SSC',
              'L4_TPC', 'L4_UPC',
              'L5_TPC:A', 'L5_TPC:B', 'L5_TPC:C', 'L5_UPC',
              'L6_BPC', 'L6_HPC', 'L6_TPC:A', 'L6_TPC:C',
              'L6_UPC',
        ]
        intratelencephalic: ['L5_TPC:C', 'L5_UPC', ]
        'pyramidal tract': ['L5_TPC:A', 'L5_TPC:B', ]

flat_mapping
~~~~~~~~~~~~
Each of the `base_systems` named in the recipe must have an associated stanza.
Information related to the `flatmaps`:
* flat_map: location of the `NRRD` file with the mapping from 3D to 2D location
* brain_regions: location of `NNRD` with brain regions mapping, voxel size and count must be the same as in `flat_map`
* hierarchy: location of `hierarchy.json` related to `brain_regions`
The `center_line_2d` defines the demarcation between `left` and `right`, so that `ipsi`/`contra` can be determined.
The same is true for `center_line_3d`.

.. code-block:: yaml

    flat_mapping:
        base_system_name0:
            flat_map: /path/to/flatmap0.nrrd
            brain_regions: /path/to/brain_regions0.nrrd
            hierarchy: /path/to/hierarchy.json
            center_line_2d: 0
            center_line_3d: -6000
        base_system_name1:
            flat_map: /path/to/flatmap1.nrrd
            brain_regions: /path/to/brain_regions1.nrrd
            hierarchy: /path/to/hierarchy.json
            center_line_2d: 0
            center_line_3d: -6000

assignment
~~~~~~~~~~

Configuration related to assignment:
* closest_count: how many fibers to consider when calculating the Gaussian used to pick the source fiber

.. code-block:: yaml

    assignment:
        closest_count: 100

colors
~~~~~~

Mapping of colour names to their RGB values

.. code-block:: yaml

    colors:
        red: [0xff, 0, 0]
        yellow: [0xff, 0xff, 0x66]
        orange: [0xf9, 0x92, 0x2b]
        lightblue: [0x90, 0xbf, 0xf9]
        blue: [0x52, 0x52, 0xa9]
        purple: [0x7c, 0x42, 0x9b]

module_grouping_color
~~~~~~~~~~~~~~~~~~~~~

Mapping of modules to colour names, used for drawing flatmaps.

.. code-block:: yaml

    module_grouping_color:
        SSp-tr: red
        SSp-ll: yellow
        SSp-ul: orange
        SSp-un: lightblue
        SSp-n: blue
        SSp-m: purple

only_segments_from_region
~~~~~~~~~~~~~~~~~~~~~~~~~

Filter segments to only be ones from neurons whose soma are in the respective region.
Note: if the index is split, this has no effect.


compensation
~~~~~~~~~~~~

Use compensation, as defined in :ref:`Density Compensation`

.. code-block:: yaml

        compensation: true


.. _`AIBS streamlines`: http://api.brain-map.org/examples/lines/index.html
