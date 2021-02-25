'''When sampling, need to know the density for regions, this handles that complexity'''
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from white_matter_projections import utils


L = logging.getLogger(__name__)

REGION_DENSITIES_PATH = Path('REGION_DENSITIES')


class SamplingRegionDensities(object):
    '''When sampling, need to know the density for regions, this tames that complexity'''
    def __init__(self, recipe, cache_dir, use_volume=True):
        '''

        Args:
            recipe(macro.MacroConnections): recipe
            cache_dir(str): location to store calculated values, None for no cache
            use_volume(False): whether to use subregion height or subregion volume
            Height was used when describing in NCX-121, but in
            bolanos-wmproj_v2/validation.py volume is used

        '''
        self.recipe = recipe
        self.cache_dir = Path(cache_dir) if cache_dir is not None else None
        self.use_volume = use_volume

    def get_region_layer_weights(self, atlas, target_population):
        '''return DataFrame with ['region', 'subregion', 'weight'] columns for `target_population`

        Args:
            atlas(voxcell.nexus.voxelbrain.Atlas): atlas to work with
            target_population(str): target for which densities are required

        Note: when a region doesn't materialize, returns empty DataFrame
        '''

        def _get_path():
            base_path = self.cache_dir / REGION_DENSITIES_PATH
            return base_path / f'{target_population}_volume_{self.use_volume}.csv'

        if self.cache_dir:
            path = _get_path()
            if path.exists():
                weights = pd.read_csv(path)
                weights['subregion'] = weights.subregion.apply(str)
                return weights

        needed_subregions = self._get_target_needed_subregions(target_population)

        if self.use_volume:
            weights = _calculate_region_layer_volume(atlas, needed_subregions)
        else:
            weights = _calculate_region_layer_heights(atlas, needed_subregions)

        if self.cache_dir:
            base_path = self.cache_dir / REGION_DENSITIES_PATH
            utils.ensure_path(base_path)
            weights.to_csv(_get_path(), index=False)

        return weights

    def get_sample_densities_by_target_population(self, atlas, target_population):
        '''for `target_population`, return densities per subregion/atlas id required for sampling

        Args:
            atlas(voxcell.nexus.voxelbrain.Atlas): atlas to work with
            target_population(str): target for which densities are required
        '''
        region_layer_weights = self.get_region_layer_weights(atlas, target_population)
        relative_densities = self._get_target_needed_subregions(target_population)

        return _get_sample_densities_by_target_population(region_layer_weights, relative_densities)

    def _get_target_needed_subregions(self, target_population):
        return _get_target_needed_subregions(self.recipe.populations,
                                             self.recipe.projections,
                                             self.recipe.layer_profiles,
                                             target_population)


def _get_sample_densities_by_target_population(region_layer_weights, relative_densities):
    '''calculate the densities per subregion/atlas_id needed for sampling in a target population

    Args:
        region_layer_weights(DataFrame): has columns ['region', 'subregion', 'weight'];
        where `weight` is either the 'layer width' or 'layer volume' for each subregion
        relative_densities(DataFrame): for each projection_name's regions and subregions,
        the relative and target density values; this comes from the projection's
        'target_layer_profiles' and 'density'

    Initial explanation (referred to as '1st'):

    Point 2 of: https://bbpteam.epfl.ch/project/issues/browse/NCX-121?focusedCommentId=69966
    ```
        Method: let
            * w be the vector of layer widths of the target region
            * p the layer profile of a projection

            let x = sum(w) / sum(w * p)

        Absolute density of the projection is each layer is then x * p.

        In other words, the layer profile has to be scaled such that the weighted mean,
        with the weights being the layer widths, is 1.
    ```

    Note: x is the inverse weighted mean of the profile, with the weights being the 'layer widths'

    ********************************************************************************

    Another explanation (referred to as '2nd'):

    from bolanos-wmproj_v2/validation.py:
    ```
        Densities explained by Michael

        The 'relative_densities' vector is scaled by a constant factor x
        such that the resulting average density from L6_bot to L1_top
        is the value specified as 'density'.

        Let:
        D be the vector of 'relative_densities' [FROM LAYER PROFILE]
        R the vector of relative volume fractions of layers (per region) [FROM ATLAS]
        d the specified total 'density' [FROM PROJECTION]

        sum_layers(x * D * R) = d
        <=> x = d / sum_layers(D * R)
         => x * D = d * D / sum_layers(D * R)

        and the scaled D, i.e. x * D is the actually specified vector of densities in layers.
    ```

    ********************************************************************************
    Reconciling the two explanations:
        p == D
        R == w / sum(w)               [w is fm 1st, R is fm 2nd]

            x = d / sum_layers(D * R) [fm 2nd]
            where `sum_layers(D * R)`
                dot(D, R) <=> dot(D, w) / sum(w)  [`w` fm 1st]
            so:
            x = d * sum(w) / dot(D, w) [w fm 1st and x, d, D fm 2nd]

            so x_1st = d * x_2nd
    '''

    weights = region_layer_weights.copy()  # `w` from 1st
    weights['normalized_weight'] = weights.weight / weights.weight.sum()  # 'R' from 2nd

    ret = weights.merge(relative_densities,
                        left_on=['region', 'subregion'],
                        right_on=['region', 'subregion'])

    # fm 2: sum_layers(D * R)
    ret['denom'] = ret.normalized_weight * ret.relative_density
    ret['denom'] = ret.groupby('projection_name')[['denom']].transform('sum')

    # x * D = d * D / sum_layers(D * R)
    ret['density'] = ret.target_density * ret.relative_density / ret['denom']

    del ret['denom']

    return ret


def _get_target_needed_subregions(populations,
                                  projections,
                                  layer_profiles,
                                  target_population):
    '''find the all the regions needed for `target_population`'''
    target_population = target_population  # trick pylint, since used in pandas query()
    target_population_profiles = (
        projections
        .query('target_population == @target_population')
        [['projection_name',
          'source_population',
          'target_population',
          'target_layer_profile_name',
          'target_density',
          'hemisphere',
          ]]
    )
    assert target_population_profiles.duplicated().sum() == 0, \
        'Have duplicated target population profiles'

    regions = (
        target_population_profiles
        .merge(populations, how='left',
               left_on='target_population',
               right_on='population')
        [['source_population',
          'target_population',
          'projection_name',
          'region',
          'subregion',
          'id',
          'hemisphere',
          'target_layer_profile_name',
          'target_density',
          ]]
        .drop_duplicates()
        .set_index(['target_layer_profile_name', 'subregion'])
        .merge(layer_profiles,
               left_index=True,
               right_on=['name', 'subregion'])
    ).reset_index(drop=True)

    return regions


def _calculate_region_layer_heights(atlas, needed_subregions):
    '''find region layer heights for subregions

    Args:
        atlas(voxcell.nexus.voxelbrain): atlas to be used for region lookup
        needed_subregions(df with [region, subregion, id]): subregions making
        up the region of interest
    '''

    brain_regions = atlas.load_data('brain_regions')

    ret = []
    for row in needed_subregions.drop_duplicates(subset=['region', 'subregion', 'id']).itertuples():
        mask = brain_regions.raw == row.id
        if not np.any(mask):
            L.warning('%s does not have any voxels', row)
            ret.append((row.region, row.subregion, 0.))
            continue

        ph = atlas.load_data('[PH]%s' % row.subregion).raw[mask]

        ret.append((row.region, row.subregion, np.mean(ph[:, 1] - ph[:, 0])))

    ret = pd.DataFrame(ret, columns=['region', 'subregion', 'weight'])
    return ret


def _calculate_region_layer_volume(atlas, needed_subregions):
    '''find region layer weights by volume

    Args:
        atlas(voxcell.nexus.voxelbrain): atlas to be used for region lookup
        needed_subregions(df with [region, subregion, id]): subregions making
        up the region of interest
    '''
    brain_regions = atlas.load_data('brain_regions')

    ret = []
    for row in needed_subregions.drop_duplicates(subset=['region', 'subregion', 'id']).itertuples():
        mask = brain_regions.raw == row.id

        if not np.any(mask):
            L.warning('%s does not have any voxels', row)
            ret.append((row.region, row.subregion, 0.))
            continue

        ret.append(
            (row.region, row.subregion, np.count_nonzero(mask) * brain_regions.voxel_volume))

    ret = pd.DataFrame(ret, columns=['region', 'subregion', 'weight'])
    return ret
