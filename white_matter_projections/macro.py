'''Load and manipulate macro recipe

In general 'macro' connections define:
  * populations: these are groups of cells that can be referenced from
    the `projections` and `p-types` sections; they generally define a region
    in the brain, along with a set of layers.  They are either the source or
    the targets of the white-matter connectivity

  * projections: These define the connection between the source and destination
    populations for a particular set of projections.  Included in this
    definition is the density in the target region, along with how the two
    populations are geometrically mapped to each other

  * p-types: these describe how the previously defined projections are actually
    instantiated, based on the source regions: basically saying which
    projections are used, and in what fraction they are to be distributed

without access to an actual circuit.  In other words, they describe the
patterns of white-matter connectivity without saying which cells are connected
to which - this is the job the microconnectivity.
'''
from collections import defaultdict
import itertools as it
import hashlib
import json
import logging
import os
import functools
import operator
import yaml

import numpy as np
import pandas as pd
from pandas.api.types import CategoricalDtype
from scipy.spatial.distance import squareform
from white_matter_projections import utils

L = logging.getLogger(__name__)


class MacroConnections(object):
    '''Load and manipulate macro recipe'''
    SERIALIZATION_NAME = 'MacroConnections'
    SERIALIZATION_PATHS = {'populations': 'populations.feather',
                           'projections': 'projections.feather',
                           'ptypes': 'ptypes.feather',
                           'layer_profiles': 'layer_profiles.feather',
                           'projections_mapping': 'projections_mapping.json',
                           'ptypes_interaction_matrix': 'ptypes_interaction_matrix',
                           'synapse_types': 'synapse_types.json',
                           }

    def __init__(self, populations,
                 projections, projections_mapping,
                 ptypes, ptypes_interaction_matrix,
                 layer_profiles, synapse_types):
        self.populations = populations
        self.projections = projections

        # dictionary of population_name sources -> projection_name -> vertices
        self.projections_mapping = projections_mapping

        self.ptypes = ptypes
        self.ptypes_interaction_matrix = ptypes_interaction_matrix
        self.layer_profiles = layer_profiles
        self.synapse_types = synapse_types

    def check_consistency(self):
        '''check recipe has all the references needed

        ex: check that source and target populations exist for ptypes & projections
        '''
        # pylint: disable=too-many-locals
        needed_projections = set(self.ptypes.projection_name)
        projections = set(self.projections.projection_name)
        if not needed_projections.issubset(projections):
            L.warning('Missing projections %s', needed_projections - projections)

        unused_projections = projections - needed_projections
        if unused_projections:
            L.warning('Unused projections %s', projections - needed_projections)

        projection_populations = np.isin(self.projections.projection_name, list(needed_projections))
        projection_populations = self.projections[projection_populations]
        needed_populations = (set(self.ptypes.source_population) |
                              set(self.ptypes.merge(self.projections,
                                                    on='source_population')['target_population']) |
                              set(projection_populations.source_population) |
                              set(projection_populations.target_population))
        populations = set(self.populations.population)
        missing_populations = needed_populations - populations
        assert not missing_populations, 'Missing %s' % missing_populations

        # ...MAYBE removes populations if no p-ptypes reference them,
        # however, for the connectivity matrix, maybe we want them?
        unused_populations = set(populations - needed_populations)
        if unused_populations:
            L.warning('Unused populations: %s', sorted(unused_populations))
            self.populations = self.populations[
                np.invert(self.populations.population.isin(unused_populations))]

        ipsi_self_target = ((self.projections.hemisphere == 'ipsi') &
                            (np.equal(self.projections.target_population.values,
                                      self.projections.source_population.values))
                            )
        assert not np.any(ipsi_self_target), \
            'Matching source and target populations: %s' % self.projections[ipsi_self_target]

        # check if ptype interactions ever go above 1
        # Note: allowed to barely go above 1 because 'That is just a rounding error. '
        # https://bbpteam.epfl.ch/project/issues/browse/NCX-128?focusedCommentId=73759
        TOLERANCE = 1.001
        for source_population, df in self.ptypes_interaction_matrix.items():
            ptype = (self.ptypes
                     .query('source_population == @source_population')
                     .set_index('projection_name')
                     )
            for i, j in it.product(df.columns, df.columns):
                if i in ptype.index and j in ptype.index:
                    pi, pj = ptype.loc[i, 'fraction'], ptype.loc[j, 'fraction']
                    wij = df.loc[i, j]
                    msg = 'For %s, the projection: %s(p=%s) with weight: %s is %f'
                    if wij * pi > TOLERANCE:
                        L.warning(msg, source_population, i, pi, wij, pi * wij)
                    if wij * pj > TOLERANCE:
                        L.warning(msg, source_population, j, pj, wij, pj * wij)

        # TODO:  need to check that projections.target_layer_profile is in self.layers

    def _get_projections(self, hemisphere=None):
        '''Get the *used* projections; a projection is used if it is referenced by a ptype'''
        ret = (self.projections
               .query('projection_name in @self.ptypes.projection_name')
               .merge(self.populations, how='left',
                      left_on='target_population', right_on='population')
               .merge(self.populations, how='left',
                      left_on='source_population', right_on='population',
                      suffixes=('_tgt', '_src'))
               .drop(['population_tgt', 'population_src'], axis=1)
               )

        if hemisphere is not None:
            assert hemisphere in ('ipsi', 'contra')
            ret = ret.query('hemisphere == @hemisphere')

        return ret

    def _get_connection_map(self, aggregate_function, hemisphere):
        '''get dataframes with the desired synapse density from source -> target region
        '''
        projections = self._get_projections(hemisphere=hemisphere)

        group_by_src, group_by_tgt = 'region_src', 'region_tgt'
        sources = sorted(projections[group_by_src].unique())
        targets = sorted(projections[group_by_tgt].unique())

        ret = pd.DataFrame(index=sources, columns=targets)
        ret.index.name = 'Source Population'
        ret.columns.name = 'Target Population Density'

        for (src, tgt), df in projections.groupby([group_by_src, group_by_tgt]):
            ret.loc[src][tgt] = aggregate_function(src, tgt, df)

        assert not ret.index.hasnans, 'Problem with Source Population: missing values'
        ret.fillna(0., inplace=True)

        return ret

    def get_connection_density_map(self, hemisphere):
        '''

        Return:
            dataframe containing
        '''
        def agg_func(_, __, df):
            '''aggregate_function'''
            return df.target_density.sum()
        return self._get_connection_map(agg_func, hemisphere)

    def calculate_densities(self, norm_layer_profiles, hemisphere=None):
        '''Get overall density in each layer of the target region

        Point 2 of: https://bbpteam.epfl.ch/project/issues/browse/NCX-121?focusedCommentId=69966
        Method: let w be the vector of layer widths of the target region, p the layer profile of a
        projection. x  = sum(w) / sum(w * p). Absolute density of the projection is each layer is
        then x * p. In other words, the layer profile has to be scaled such that the weighted mean,
        with the weights being the layer widths, is 1.

        Args:
            norm_layer_profiles(df): corresponds to 'x' in the docstring
        '''
        hemisphere = hemisphere  # pylint workaround: variable used in query()
        redundant_columns = ['name', 'index', 'region', 'subregion']
        ret = (self._get_projections(hemisphere=hemisphere)
               .merge(self.layer_profiles, how='left',
                      left_on=['target_layer_profile_name', 'subregion_tgt'],
                      right_on=['name', 'subregion'])
               .merge(norm_layer_profiles.T.reset_index().melt('index'), how='left',
                      left_on=['region_tgt', 'target_layer_profile_name'],
                      right_on=['region', 'index'])
               .dropna()
               .drop(redundant_columns, axis=1)
               )

        ret['density'] = ret.value * ret.target_density * ret.relative_density

        return ret

    def get_target_region_density(self, norm_layer_profiles, hemisphere):
        '''return dataframe with all the densities for all the target regions'''
        projections = self.calculate_densities(norm_layer_profiles, hemisphere)

        regions = np.sort(projections.region_tgt.unique())
        subregions = np.sort(projections.subregion_tgt.unique())

        ret = pd.DataFrame(index=regions, columns=subregions)
        ret.index.name = 'Target Region'
        ret.columns.name = 'Target Layer'

        for (region, layer), df in projections.groupby(['region_tgt', 'subregion_tgt']):
            ret.loc[region][layer] = df.density.sum()

        return ret

    def get_target_region_density_modules(
            self, norm_layer_profiles, target_acronym, modules, hemisphere='ipsi'):
        '''Vertical profile of incoming projections, grouped by module'''
        target_acronym = target_acronym  # pylint workaround: variable used in query()
        projections = (self.calculate_densities(norm_layer_profiles, hemisphere)
                       .query('region_tgt == @target_acronym')
                       )

        ret = pd.DataFrame(index=sorted(projections['subregion_tgt'].unique()),
                           columns=sorted(m[0] for m in modules))
        ret.index.name, ret.columns.name = 'Target', 'Source'

        for module, regions in modules:
            regions = regions  # pylint workaround: variable used in query()
            for tgt, df in projections.query('region_src in @regions').groupby('subregion_tgt'):
                ret.loc[tgt][module] = df.density.sum()
        return ret

    def get_target_region_density_sources(
            self, norm_layer_profiles, target_acronym, hemisphere='ipsi'):
        '''Vertical profile of incoming projections

        For a target region a stacked bar plot of incoming projection densities for each
        region/subregion combination (i.e. L5it from VISp, etc).

        Point 4 of: https://bbpteam.epfl.ch/project/issues/browse/NCX-121?focusedCommentId=69966
        '''
        target_acronym = target_acronym  # pylint workaround: variable used in query()
        projections = (self.calculate_densities(norm_layer_profiles, hemisphere)
                       .query('region_tgt == @target_acronym')
                       )

        source = 'source_population'
        target = 'subregion_tgt'

        ret = pd.DataFrame(index=sorted(projections[target].unique()),
                           columns=sorted(projections[source].unique()))
        ret.index.name, ret.columns.name = 'Target', 'Source'

        for (tgt, src), df in projections.groupby([target, source]):
            ret.loc[tgt][src] = df.density.sum()
        return ret

    def _serialize(self, base_path):
        '''serialize recipe to `base_path`'''
        utils.ensure_path(base_path)

        for prop in ('populations', 'projections', 'ptypes', 'layer_profiles',):
            path = os.path.join(base_path, MacroConnections.SERIALIZATION_PATHS[prop])
            utils.write_frame(path, getattr(self, prop))

        ptypes_interaction_matrix = {}
        for k, v in self.ptypes_interaction_matrix.items():
            ptypes_interaction_matrix[k] = {'index': tuple(v.index),
                                            'values': tuple(v.values.ravel()),
                                            }

        def convert_vertices(d):
            '''convert vertices into dictionary'''
            ret = {}
            for k, v in d.items():
                if k == 'vertices':
                    ret[k] = tuple(v.ravel())
                elif isinstance(v, dict):
                    ret[k] = convert_vertices(v)
                else:
                    ret[k] = v
            return ret

        projections_mapping = convert_vertices(self.projections_mapping)

        def write_json(type_, obj):
            '''write json'''
            path = os.path.join(base_path, MacroConnections.SERIALIZATION_PATHS[type_])
            with open(path, 'w') as fd:
                json.dump(obj, fd)

        write_json('ptypes_interaction_matrix', ptypes_interaction_matrix)
        write_json('projections_mapping', projections_mapping)
        write_json('synapse_types', self.synapse_types)

    @classmethod
    def _deserialize(cls, base_path):
        '''deserialize recipe to `base_path`'''
        # pylint: disable=too-many-locals
        def load(name):
            '''load'''
            path = os.path.join(base_path, MacroConnections.SERIALIZATION_PATHS[name])
            return utils.read_frame(path)

        populations = load('populations')
        projections = load('projections')
        ptypes = load('ptypes')
        layer_profiles = load('layer_profiles')

        def load_json(type_):
            '''load json'''
            path = os.path.join(base_path, MacroConnections.SERIALIZATION_PATHS[type_])
            with open(path) as fd:
                return json.load(fd)

        synapse_types = load_json('synapse_types')

        ptypes_interaction_matrix = {}
        for k, v in load_json('ptypes_interaction_matrix').items():
            data = np.array(v['values']).reshape((len(v['index']), len(v['index'])))
            ptypes_interaction_matrix[k] = pd.DataFrame(data, index=v['index'], columns=v['index'])

        projections_mapping = load_json('projections_mapping')

        def convert_vertices(d):
            '''convert vertices into useable form'''
            ret = {}
            for k, v in d.items():
                if k == 'vertices':
                    ret[k] = np.array(v).reshape((3, 2))
                elif isinstance(v, dict):
                    ret[k] = convert_vertices(v)
                else:
                    ret[k] = v
            return ret
        projections_mapping = convert_vertices(projections_mapping)

        ret = cls(populations,
                  projections, projections_mapping,
                  ptypes, ptypes_interaction_matrix,
                  layer_profiles, synapse_types)

        return ret

    @classmethod
    def load_recipe(cls, recipe_yaml, region_map, cache_dir=None,
                    subregion_translation=None,
                    region_subregion_format='{region}_{subregion}'):
        '''load population/projection/p-type recipe

        Args:
            recipe_yaml(str): the recipe following format, in yaml
            region_map(voxcell.RegionMap): hierarchy to verify population acronyms against
            cache_dir(str): location to save cached data
            subregion_translation(dict): subregion alias -> subregion in atlas
            region_subregion_format(str): format string describing the region/subregion ->
            acronym conversion

        Returns:
            instance of MacroConnections
        '''
        # pylint: disable=too-many-locals
        m = hashlib.sha256()
        m.update(recipe_yaml)
        hexdigest = m.hexdigest()

        if cache_dir is not None:
            try:
                path = os.path.join(cache_dir, MacroConnections.SERIALIZATION_NAME, hexdigest)
                ret = cls._deserialize(path)
                L.debug('Using serialized recipe from %s', path)
                return ret
            except:  # noqa  # pylint: disable=bare-except
                pass

        recipe = yaml.load(recipe_yaml, Loader=yaml.FullLoader)

        if subregion_translation is None:
            subregion_translation = {}

        pop_cat, populations = _parse_populations(recipe['populations'],
                                                  region_map,
                                                  subregion_translation,
                                                  region_subregion_format)
        projections, projections_mapping = _parse_projections(recipe['projections'], pop_cat)
        ptypes, ptypes_interaction_matrix = _parse_ptypes(recipe['p-types'], pop_cat)
        layer_profiles = _parse_layer_profiles(recipe['layer_profiles'], subregion_translation)

        synapse_types = _parse_synapse_types(recipe['synapse_types'])

        ret = cls(populations,
                  projections, projections_mapping,
                  ptypes, ptypes_interaction_matrix,
                  layer_profiles, synapse_types)

        ret.check_consistency()

        if cache_dir is not None:
            path = os.path.join(cache_dir, MacroConnections.SERIALIZATION_NAME, hexdigest)
            ret._serialize(path)  # pylint: disable=protected-access
            with open(os.path.join(path, 'recipe.yaml'), 'w') as fd:
                fd.write(recipe_yaml)

        return ret

    def __repr__(self):
        return 'MacroConnections: [populations: {}, projections: {}, ptypes: {}]'.format(
            len(self.populations), len(self.projections), len(self.ptypes))

    __str__ = __repr__


def _parse_populations(populations, region_map, subregion_translation, region_subregion_format):
    '''parse_populations

    populations(dict): as loaded from yaml file
    region_map(voxcell.RegionMap): hierarchy to verify population acronyms against
    subregion_translation(dict): subregion alias -> subregion in atlas
    region_subregion_format(str): format string describing the region/subregion ->
    acronym conversion

    Returns:
        tuple of:
            CategoricalDtype - all the populations
            DataFrame: w/ columns:
                population: population name
                region: name of region
                layer: layer name
                acronym: name of region full name, including subregion
                id: id in hierarchy, -1 if the region/layer combo doesn't exist
                population_filter: Category of 'Empty'/EXC/intratelencephalic or
                'pyramidal tract'
    '''
    data, removed = [], []
    for pop in populations:
        pop_filter = 'Empty'
        if 'proj_type' in pop['filters']:
            pop_filter = pop['filters']['proj_type']
            assert pop_filter in ('intratelencephalic', 'pyramidal tract', ), \
                'only can consider "intratelencephalic", "pyramidal tract",  at the moment'
        elif 'synapse_type' in pop['filters']:
            pop_filter = pop['filters']['synapse_type']
            assert pop_filter == 'EXC', 'only can consider EXC at the moment'

        if isinstance(pop['atlas_region'], dict):
            atlas_regions = [pop['atlas_region'], ]
        else:
            atlas_regions = pop['atlas_region']

        for atlas_region in atlas_regions:
            region = atlas_region['name']
            for subregion in atlas_region['subregions']:
                subregion = subregion_translation.get(subregion, subregion)

                id_ = utils.region_subregion_to_id(
                    region_map, region, subregion, region_subregion_format)
                if id_ <= 0:
                    removed.append((region, subregion))
                    continue

                data.append((id_, pop['name'], region, subregion, pop_filter))

    columns = ['id', 'population', 'region', 'subregion', 'population_filter']
    populations = pd.DataFrame(data, columns=columns)

    pop_cat = CategoricalDtype(populations.population.unique())
    populations.population = populations.population.astype(pop_cat)

    populations.population_filter = populations.population_filter.astype('category')

    if removed:
        L.warning('%s are missing from atlas', sorted(removed))

    return pop_cat, populations


def _parse_projections(projections, pop_cat):
    '''parse_projections

    Returns: tuple of:
        DataFrame with columns:
            projection_name: name of the projection
            source_population: ref. to source population
            target_population: ref to target population
            hemisphere: 'ipsi' or 'contra'
            target_density(float): target synapse density
            target_layer_profile_name: ref. to layer_profiles stanza
            target_layer_profile_fraction: modifier to above profile
            connection_mapping: ref. connection_mapping stanza
            synapse_type_name: ref. to synapse_types stanza
            synapse_type_fraction: modifier to above type
        dict of projections_mapping keyed on: population_source -> projection_name ->
            {vertices -> np.array, target_population -> name}
    '''
    # pylint: disable=too-many-locals
    def get_vertices(node):
        ''' Note: the coordinate system uses x for columns, and y for rows, but within the code
        the normal convention is used: 0th axis is rows (ie: x), and 1th is columns
        '''
        vertices = np.array(zip(node['mapping_coordinate_system']['y'],
                                node['mapping_coordinate_system']['x'],))
        return vertices
    data = []
    projections_mapping = defaultdict(dict)
    missing_targets = []
    for proj in projections:
        if not proj['targets']:
            missing_targets.append(proj['source'])
            continue

        source = proj['source']

        for target in proj['targets']:
            if target['source_filters']:
                L.warning('Source filters not implemented')

            assert len(target['target_layer_profiles']) == 1, 'Too many layer targets!'
            projection_name = target['projection_name']
            target_layer_name = target['target_layer_profiles'][0]['name']
            target_layer_fraction = target['target_layer_profiles'][0]['fraction']

            mapping = target['presynaptic_mapping']
            assert mapping['mapping_coordinate_system']['base_system'] == 'Allen Dorsal Flatmap', \
                'Currently only handle Allen Dorsal Flatmap'
            assert projection_name not in projections_mapping[source], \
                'Duplicate projection target: %s -> %s' % (source, projection_name)
            projections_mapping[source][projection_name] = {
                'variance': mapping['mapping_variance'],
                'vertices': get_vertices(mapping),
                'target_population': target['population'],
            }

            assert len(target['synapse_types']) == 1, 'Too many synapses types!'
            synapse_type_name = target['synapse_types'][0]['name']
            synapse_type_fraction = target['synapse_types'][0]['fraction']

            data.append((
                projection_name,
                source,
                target['population'],
                target['hemisphere'],
                target['density'],
                target_layer_name,
                target_layer_fraction,
                target['connection_mapping']['type'],
                synapse_type_name,
                synapse_type_fraction,
            ))

        projections_mapping[source]['vertices'] = get_vertices(proj)

    columns = ['projection_name', 'source_population', 'target_population',
               'hemisphere', 'target_density', 'target_layer_profile_name',
               'target_layer_profile_fraction', 'connection_mapping',
               'synapse_type_name', 'synapse_type_fraction',
               ]

    projections = pd.DataFrame(data, columns=columns)
    projections.source_population = projections.source_population.astype(pop_cat)
    projections.target_population = projections.target_population.astype(pop_cat)
    projections.hemisphere = projections.hemisphere.astype(utils.HEMISPHERE)

    if missing_targets:
        L.warning('Sources %s do not have any associated targets', missing_targets)

    return projections, dict(projections_mapping)


def _parse_ptypes(ptypes, pop_cat):
    '''parse_ptypes

    Returns: tuple of
        DataFrame with columns:
            source_population: ref. to source population
            projection_name: ref. to projection
            fraction: fraction of the source that participates in the projection
        dict of interaction_matrices keyed on: source_population -> DataFrame;
        adjustments to the default interaction probability of the pairwise randomnly
        sampled GIDs of all above projection_names
    '''
    interaction_matrices = {}
    data = []
    for ptype in ptypes:
        source_population = ptype['population']
        if not ptype['fractions']:
            L.warning('Missing fractions for ptype population: %s', ptype['population'])
            continue
        for projection, fraction in ptype['fractions'].items():
            data.append((source_population, projection, fraction))

        if 'interaction_mat' in ptype:
            interaction_mat = squareform(ptype['interaction_mat']['strengths'])
            close_to_zero = np.isclose(np.zeros_like(interaction_mat), interaction_mat)
            interaction_mat[close_to_zero] = 1.
            assert source_population not in interaction_matrices, \
                'ptype for source_population(%s) already defined' % source_population
            interaction_matrices[source_population] = pd.DataFrame(
                interaction_mat,
                columns=ptype['interaction_mat']['projections'],
                index=ptype['interaction_mat']['projections'])

    ptypes = pd.DataFrame(data, columns=['source_population', 'projection_name', 'fraction'])
    ptypes.source_population = ptypes.source_population.astype(pop_cat)

    return ptypes, interaction_matrices


def _parse_layer_profiles(layer_profiles, subregion_translation):
    '''parse_layer_profiles

    Returns:
        DataFrame with columns: name, subregion, relative_density
    '''
    data = []
    for profile in layer_profiles:
        for densities in profile['relative_densities']:
            for subregion in densities['layers']:
                subregion = subregion_translation.get(subregion, subregion)
                data.append((profile['name'], subregion, densities['value']))

    layer_profiles = pd.DataFrame(data, columns=['name', 'subregion', 'relative_density'])
    return layer_profiles


def _parse_synapse_types(synapse_types):
    '''
    - name: type_1
      physiology:
        - phys_parameter: U
          distribution:
            - type: truncated_gaussian
              dist_parameters:
                - mean: 0.46
                - std: 0.26
        - ...
    '''
    def get_phys_parameters(synapse_type):
        '''return phys_parameters for a synapse type'''
        return {physiology['phys_parameter'] for physiology in synapse_type['physiology']}

    if len(synapse_types) > 1:
        phys_parameters = [get_phys_parameters(t) for t in synapse_types]
        all_parameters = functools.reduce(operator.or_, phys_parameters)
        common_parameters = functools.reduce(operator.and_, phys_parameters)
        difference = all_parameters - common_parameters

        assert not difference, \
            'phys_parameter(s) not defined in all synapse types: %s' % ', '.join(difference)

    return synapse_types
