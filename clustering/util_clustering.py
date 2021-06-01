import datetime
import pandas as pd
import numpy as np
from sklearn.metrics import calinski_harabasz_score, davies_bouldin_score, silhouette_score

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
    results = pd.DataFrame(columns=["Model","Params","Score","Halinski-Harabasz","Davies-Bouldin","Silhouette","Time","NLabels","Labels"])
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
                        "Halinski-Harabasz": vr_score, \
                        "Davies-Bouldin": db_score, \
                        "Silhouette": sil_score, \
                        "Time": stop-start, \
                        "NLabels": np.unique(labels).shape[0], \
                        "Labels": labels}, \
                        ignore_index=True)

    print("Grid search completed.")
    return results