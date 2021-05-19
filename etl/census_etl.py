import numpy as np
import pandas as pd
import censusdata
import geopandas as gp
from geopandas.tools import overlay
import os

CENSUS_URL = 'https://data.cityofchicago.org/resource/bt9m-d2mf.geojson?$limit=9999999'
BEATS_GEOJSON = '../data/geographies/Boundaries - Police Beats (current).geojson'

def load_data():
    '''
    loads relevant datasets
    '''
    # Download Census block boundaries for Chicago
    census_gdf = gp.read_file(CENSUS_URL)

    # Pull ACS data
    census_tables = {
        'GEO_ID': 'GEO_ID',
        'B02001_001E': 'Total Pop',
        'B02001_002E': 'White',
        'B02001_003E': 'Black',
        'B19013_001E': 'Median Income',
        'B03003_003E': 'Hispanic'}

    acs_df = censusdata.download("acs5", 
                                 2018,
                                 censusdata.censusgeo([("state", "17"),
                                                        ("county", "031"),
                                                        ("tract", "*"),
                                                        ("block group", "*")]),
                                 list(census_tables.keys()))

    # Rename columns
    acs_df.rename(columns=census_tables, inplace=True)

    beats_gdf = gp.read_file(BEATS_GEOJSON)

    return census_gdf, acs_df, beats_gdf

def clean_data(census_gdf, acs_df, beats_gdf):
    '''
    Spatially merges and cleans data
    '''
    # convert to geo12
    acs_df['geo_12'] = acs_df["GEO_ID"].map(lambda x: str(x)[-12:])
    census_gdf['geo_12'] = census_gdf["geoid10"].map(lambda x: str(x)[:12])

    # Dissolve census data
    census_bg_gdf = census_gdf[['geometry', 'geo_12']].dissolve(by='geo_12')

    # Merge ACS data with Census block group boundaries
    merged_acs_gdf = gp.GeoDataFrame(census_bg_gdf.merge(acs_df, on="geo_12",
                                     how="inner"), crs=census_bg_gdf.crs)

    # Replace -6666 with na
    merged_acs_gdf = merged_acs_gdf.drop(['GEO_ID'], axis=1)
    merged_acs_gdf['Median Income'] = np.where(merged_acs_gdf['Median Income'] < 0,
                                        np.nan, merged_acs_gdf['Median Income'])

    # Calculate BG area
    merged_acs_gdf['bg_area_km2'] = merged_acs_gdf.to_crs('EPSG:3857').area / 10**6

    # Calculate Beat area
    beats_gdf['beat_area_km2'] = beats_gdf.to_crs('EPSG:3857').area / 10**6

    # Ensure same projection
    beats_gdf.crs = merged_acs_gdf.crs

    # Merge data
    beats_acs_gdf = overlay(beats_gdf, merged_acs_gdf, how='union')

    # Drop if beat is NA
    beats_acs_gdf = beats_acs_gdf.dropna(subset=['beat'])

    return beats_acs_gdf

def create_features(beats_acs_gdf):
    '''
    creates final featureset
    '''
    # 1. Calculate proportion of each block group in beat
    beats_acs_gdf['beat_bg_area_km2'] = beats_acs_gdf.to_crs('EPSG:3857').area / 10**6
    beats_acs_gdf['beat_bg_area_prop'] = beats_acs_gdf['beat_bg_area_km2'] / (
                                              beats_acs_gdf['bg_area_km2'])

    #2. Recalculate populations
    for col in ['Total Pop', 'White', 'Black', 'Hispanic']:
        beats_acs_gdf[col] = beats_acs_gdf[col] * beats_acs_gdf['beat_bg_area_prop']
    #3. Sum population by beat
    beats_demo_gdf = beats_acs_gdf.groupby(
        ['beat', 'beat_num', 'district', 'sector']).agg(
        {'Total Pop': ['sum'],
        'White': ['sum'],
        'Black': ['sum'],
        'Hispanic': ['sum'],
        'Median Income': ['mean']})
    #4. Calc percentages 
    for col in ['White', 'Black', 'Hispanic']:
        beats_demo_gdf[col] = beats_demo_gdf[col] / beats_demo_gdf['Total Pop']

    # rename columns
    beats_demo_gdf = beats_demo_gdf.reset_index()
    beats_demo_gdf.columns = [col[0] for col in beats_demo_gdf.columns]

    # Assign median to NAs
    beats_demo_gdf['Total Pop'] = np.where(beats_demo_gdf['Total Pop']==0, 
                                           np.nan, beats_demo_gdf['Total Pop'])
    beats_demo_gdf = beats_demo_gdf.fillna(beats_demo_gdf.median())
    return beats_demo_gdf

def go():
    '''
    Main function that runs all steps.
    '''
    print('Loading data...')
    census_gdf, acs_df, beats_gdf = load_data()

    print('Cleaning data...')
    beats_acs_gdf = clean_data(census_gdf, acs_df, beats_gdf)

    print('Creating features...')
    beats_demo_gdf = create_features(beats_acs_gdf)

    # make features directory if not present
    if not os.path.exists('../data/features'):
        os.mkdir('../data/features')

    print('Saving features...')
    beats_demo_gdf.to_csv('../data/features/census_demographics.csv', index=False)
    print('Generated features for Census data')

if __name__ == "__main__":
    go()
