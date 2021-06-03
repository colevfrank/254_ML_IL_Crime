import pandas as pd
import numpy as np
from util_clustering import get_clusters

def create_training_data_race(data, keep_cols, demog):
    '''
    Subset training data to include 9 features: ISR, UOF, Complaints for each of 3 races
    '''
    # Create training data
    training_data = data.copy()

    # keep demographic columns
    demo_cols = [c for c in training_data.columns if demog in c]
    keep_cols_copy = keep_cols + demo_cols + ['BEAT']
    
    # filter columns
    remove_cols = [c for c in training_data.columns if c not in keep_cols_copy]
    training_data.drop(remove_cols, axis=1, inplace=True)
    
    # Sum by year
    training_data = training_data.groupby(by='BEAT').agg(np.sum).reset_index()
    beats = training_data['BEAT']
    training_data.drop('BEAT', axis=1, inplace=True)
    return training_data, beats

def create_training_data_year(data, remove_cols, year):
    '''
    Filter training data by removing columns and subsettting to a certain year.
    '''
    # Create training data
    training_data = data.copy()
    
    # filter columns
    for dem in remove_cols:
        cols = [c for c in training_data.columns if dem in c]
        training_data.drop(cols, axis=1, inplace=True)
        
    # filter by year
    training_data = training_data[training_data['YEAR']==year]
    training_data.drop('YEAR',axis=1)

    # drop beat feature
    beats = training_data['BEAT']
    training_data.drop('BEAT', axis=1, inplace=True)
    return training_data, beats
    
def get_match_mat(labels):
    '''
    create adjacency matrix, where entries are adjacent if they're in same cluster
    '''
    mat = pd.merge(labels, labels, how='cross') \
        .assign(ADJACENT=lambda x: x['Cluster_x'] == x['Cluster_y'])
    return mat['ADJACENT']
    
def get_pct_match(mat1, mat2, n):
    '''
    Get similarity between two nxn matrices.
    '''
    total_matches = np.sum(mat1 & mat2)
    # remove diagonal matches, and divide by two to only count a match once (matrix is symmetric)
    actual_matches = (total_matches - n) / 2
    mat1_size = (np.sum(mat1) - n )/2
    mat2_size = (np.sum(mat2) - n )/2
    max_matches = max(mat1_size, mat2_size)
    percent_match = actual_matches / max_matches
    return percent_match

def get_cosine_match(mat1, mat2, n):
    '''
    Get similarity between two nxn matrices.
    '''
    total_matches = np.sum(mat1 & mat2)
    # remove diagonal matches, and divide by two to only count a match once (matrix is symmetric)
    actual_matches = (total_matches - n) / 2
    mat1_size = np.sqrt((np.sum(mat1) - n )/2)
    mat2_size = np.sqrt((np.sum(mat2) - n )/2)
    cos_match = actual_matches / (mat1_size * mat2_size)
    return cos_match

def compute_cluster_sim_score(labels1, labels2):
    '''
    Takes two vectors of labels (that correspond to the same beats),
    returns the similarity of those two cluster label sets
    '''
    n = len(labels1)
    mat1 = get_match_mat(labels1)
    mat2 = get_match_mat(labels2)
    return get_cosine_match(mat1, mat2, n)

def compute_similarity_matrix(data, comp_values, filter_cols, training_func, model):
    """
        Runs clustering over subsets of data and measures similarity.
    Params:
        data - data frame of features
        comp_values - distinct values for subsetting
        filter_cols - columns to filter on (remove for yearly, keep for race)
        training_func - function to create training data
        model - clustering model
    Returns:
        nxn matrix similarity scores as data frame
    """
    long_df = pd.DataFrame()
    for value_1 in comp_values:
        training_data_1, beats_1 = training_func(data, filter_cols, value_1)
        _, clustered_data_1 = get_clusters(training_data_1, beats_1, model) # We are subsetting the labels later
        
        for value_2 in comp_values:
            # Get clusters
            training_data_2, beats_2 = training_func(data, filter_cols, value_2)
            _, clustered_data_2 = get_clusters(training_data_2, beats_2, model)

            # Exclude beats that Merge on beat and extract new cluster labels only for those that match
            common_beats = clustered_data_1.index.intersection(clustered_data_2.index)
            cluster_labels_1 = clustered_data_1.loc[common_beats,'Cluster']
            cluster_labels_2 = clustered_data_2.loc[common_beats,'Cluster']

            # Compare
            sim_score = compute_cluster_sim_score(cluster_labels_1, cluster_labels_2)
            
            long_df = long_df.append({'row':value_1, 'col':value_2, 'sim':sim_score}, ignore_index=True)
    wide_df = long_df.pivot(index='row',columns='col',values='sim')
    return wide_df.sort_index(axis=0).sort_index(axis=1)