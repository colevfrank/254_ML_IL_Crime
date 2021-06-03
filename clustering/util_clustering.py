import datetime
import pandas as pd
import numpy as np
from sklearn.metrics import calinski_harabasz_score, davies_bouldin_score, silhouette_score
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

def pre_process(training_data):
    '''
    Prepare raw training data for clustering; do PCA.
    '''
    # Normalize columns
    scaler = StandardScaler()
    scaled_training_data = pd.DataFrame(scaler.fit_transform(training_data),\
        columns=training_data.columns)
    
    # Do PCA
    N_TOP_PCA_COMPONENTS = 6 # See Model Selection.ipynb for why we chose this number.
    pca_training_data, pca = generate_pca_data(scaled_training_data, N_TOP_PCA_COMPONENTS)
    return pca_training_data

def setup_and_run_model(cluster_model, pca_training_data, beats):    
    '''
    Using model passed in, fit to training data and return labels.
    '''
    # run model
    cluster_labels = cluster_model.fit_predict(pca_training_data)
    clustered_data = pd.concat([pca_training_data, pd.Series(cluster_labels, name='Cluster'),beats], axis=1)
    return cluster_labels, clustered_data

def get_clusters(training_data, beats, cluster_model):
    '''
    Pre-process and run model based on training data
    '''
    pca_training_data = pre_process(training_data)
    cluster_labels, clustered_data = setup_and_run_model(cluster_model, pca_training_data, beats)
    return cluster_labels, clustered_data    
    
def generate_pca_data(dataset, n_components=None):
    """
        Fit and transform dataset into principal component space.
    Params:
        dataset - data frame of observations
        n_components - number of top primary components to keep. if None, keep all.
    Returns:
        tuple of (transformed data frame, pca instance)
    """
    if n_components is None:
        n_components = dataset.shape[1]
    elif n_components > dataset.shape[1]:
        n_components = dataset.shape[1]
    pca = PCA(n_components=n_components)
    pc_dataset = pd.DataFrame(pca.fit_transform(dataset))
    pc_dataset.columns = ['PC'+str(i+1) for i in range(n_components)]
    return pc_dataset, pca

def grid_search_clustering(model, param_grid, data, metric=None):
    """ 
    Implements grid search for hyperparameter tuning without cross validation. 
    Returns a data frame with model params, timing, and sklearn's non-supervised
    performance metrics.
    Params:
        model - Instance of clustering estimator to train
        param_grid - Parameters created by sklearn's ParameterGrid
        metric - Estimator-specific metric name to report, e.g. 'inertia_'
        data - pandas data frame or ndarray to cluster
    """
    results = pd.DataFrame(columns=["Model","Params","Score","Calinski-Harabasz","Davies-Bouldin","Silhouette","Time","NLabels","Labels"])
    for params in param_grid: 
        # Train the clustering model
        model_name = type(model).__name__
        print(f"Training {model_name} with: {params}")
        start = datetime.datetime.now()
        model.set_params(**params)
        labels = model.fit_predict(data)
        stop = datetime.datetime.now()
        print("Training Time Elapsed:", stop - start)
        # Compute user-specified score of the clustering quality
        if type(metric) == str:
            if metric == 'bic':
                # Hack for GMM
                score = model.bic(data)
            else:
                score = getattr(model, metric)
        else:
            score = np.nan
        # Compute common cluster quality scores
        try:
            vr_score = calinski_harabasz_score(data, labels)
        except:
            # returns one cluster
            vr_score = np.nan
        try:
            sil_score = silhouette_score(data, labels)
        except:
            # returns one cluster
            sil_score = np.nan
        try:
            db_score = davies_bouldin_score(data, labels)
        except:
            # returns one cluster
            db_score = np.nan
        # Save results
        results = results.append({ \
                        "Model": model_name, \
                        "Params":params, \
                        "Score": score, \
                        "Calinski-Harabasz": vr_score, \
                        "Davies-Bouldin": db_score, \
                        "Silhouette": sil_score, \
                        "Time": stop-start, \
                        "NLabels": np.unique(labels).shape[0], \
                        "Labels": labels}, \
                        ignore_index=True)

    print("Grid search completed.")
    return results