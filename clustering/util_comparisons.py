import pandas as pd
import numpy as np

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
    training_data.drop('BEAT', axis=1)
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

def get_sim_clusters(labels1, labels2):
    '''
    Takes two vectors of labels (that correspond to the same beats),
    returns the similarity of those two cluster label sets
    '''
    n = len(labels1)
    mat1 = get_match_mat(labels1)
    mat2 = get_match_mat(labels2)
    return get_cosine_match(mat1, mat2, n)

