import pandas as pd
import numpy as np

def listcolumn_pivot_wider(data, column_name, prefix):
    '''
    Split a pipe-separated multi-valued string column into dummy variables. 
    Append dummies to tbl and remove original column.
    '''
    cleaned = data[column_name].str.replace(" ", "", regex=False)
    dummies = cleaned.str.get_dummies()
    dummies.rename(columns=lambda c: prefix + c, inplace=True)
    data.drop(labels=column_name, axis='columns', inplace=True)
    return pd.concat([data, dummies], axis=1)

def listcolumn_pivot_longer(data, column_name, prefix, sep=None):
    '''
        Pivots dummy columns into a single multi-valued categorical column.
        Inverse of listcolumn_pivot_wider ... (defined in ETL/Complaints ETL)
    Inputs:
        data - pandas data frame
        column_name - output column name for categorical variable
        prefix - common prefix to identify dummy columns
        sep - output column as 'sep'-concatenated string. Default: output column as list
    '''
    # First find all categorical values -- in order!
    dummyvals = []
    for column in data.columns:
        if column.startswith(prefix):
            dummyval = column[len(prefix):]
            dummyvals.append(dummyval)
    # Next get the multi-dummy values per row:
        # The indexing is pretty ugly... 
        # the idea is to slice a row of dummies, ie [0,1,1,0]
        # against the dummy vals ['foo','bar','baz','buz']
        # to get the dummies present in that row ['bar','baz']
        # repeating the process for every row in the data frame
    catcolumns = data.columns.str.startswith(prefix)
    dummyvals = np.array(dummyvals)
    multiseries = data.loc[:, catcolumns].T.apply(lambda row: dummyvals[row.to_numpy(bool)])
    if sep:
        multiseries = multiseries.apply(lambda x: sep.join(x))
    # Drop the old columns and add the new
    return pd.concat([data.loc[:, ~catcolumns], pd.Series(multiseries, name=column_name)], axis=1)