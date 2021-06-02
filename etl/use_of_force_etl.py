import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

categ_cols = ['SUBJECT_RACE', 'SUBJECT_GENDER', 'SUBJECT_ARMED_DESC']

binary_cols = ['TOTAL_COUNT', 
               'MEMRESP_RWOW',
               'MEMRESP_RWW']


def load_data():
    
    uof_csv = '../data/Use_of_force_CPD.csv'
    df = pd.read_csv(uof_csv)
    #print("DF IS A", type(df))
    return df


def clean_data(df):

    # conver to datetime
    df['DATETIME'] = pd.to_datetime(df['DATE'])
    df['YEAR'] = df['DATETIME'].dt.year
    df['DATE'] = df['DATETIME'].dt.date
    df['MONTHYEAR'] = df['DATETIME'].dt.to_period('M')

    # limit to 2016-2019
    df = df[df['YEAR'].between(2016, 2019)]

    # Keep only most recent record per card number
    df = df.drop_duplicates(subset = 'TRR_REPORT_ID')

    return df


def add_mem_resp_indicators(df):
    '''
    '''
    feat_lst = [k for k in df.columns if k[:2] == "ME" or k[:4] == "SUBA"]

    FM_lst = [k for k in df.columns if k[:10] == "MEMRESP_FM"]
    CT_lst = [k for k in df.columns if k[:10] == "MEMRESP_CT"]
    RWOW_lst = [k for k in df.columns if k[:12] == "MEMRESP_RWOW"]
    RWW_lst = [k for k in df.columns if k[:11] == "MEMRESP_RWW"]

    df['MEMRESP_FM'] = df['MEMRESP_FM_MEM_PRESENCE_I']
    for col in FM_lst:
        df['MEMRESP_FM'] = np.where(df[col] == 'Y', 'Y', df['MEMRESP_FM'])
        
    df['MEMRESP_CT'] = df['MEMRESP_CT_ARMBAR_I']
    for col in CT_lst:
        df['MEMRESP_CT'] = np.where(df[col] == 'Y', 'Y', df['MEMRESP_CT'])

    df['MEMRESP_RWOW'] = df['MEMRESP_RWOW_KICKS_I']
    for col in RWOW_lst:
        df['MEMRESP_RWOW'] = np.where(df[col] == 'Y', 'Y', df['MEMRESP_RWOW'])
        
    df['MEMRESP_RWW'] = df['MEMRESP_RWW_CANINE_I']
    for col in RWW_lst:
        df['MEMRESP_RWW'] = np.where(df[col] == 'Y', 'Y', df['MEMRESP_RWW'])
    
    return df

    
def add_features_at_beatyear(df):
    '''
    Aggregate features at the beat/year level
    '''
    #add dummies
    dummy_cols = pd.get_dummies(df[categ_cols], columns = categ_cols, prefix = categ_cols).columns
    df = pd.get_dummies(df, columns = categ_cols, prefix = categ_cols)

    # Convert to string to then be converted below
    df['TOTAL_COUNT'] = 'Y'

    for col in binary_cols:
        df[col] = np.where(df[col]=='Y',1,0)

    # add list of features to include
    binary_cols.extend(dummy_cols)
    binary_cols.extend(['BEAT', 'YEAR'])

    uof_beat_yr = df[binary_cols].groupby(['BEAT', 'YEAR']).sum().reset_index()

    # Finalize limited set of features and rename
    uof_beat_yr['HISPANIC'] = uof_beat_yr['SUBJECT_RACE_BLACK HISPANIC'] + (
                              uof_beat_yr['SUBJECT_RACE_WHITE HISPANIC'])
    uof_beat_yr['BLACK'] = uof_beat_yr['SUBJECT_RACE_AFRICAN-AMERICAN']
    uof_beat_yr['WHITE'] = uof_beat_yr['SUBJECT_RACE_WHITE']
    uof_beat_yr['POLICE_W_WEAPON'] = uof_beat_yr['MEMRESP_RWW']
    uof_beat_yr['POLICE_WO_WEAPON'] = uof_beat_yr['MEMRESP_RWOW']

    final_cols = ['BEAT', 'YEAR', 'TOTAL_COUNT', 'POLICE_W_WEAPON', 
                  'POLICE_WO_WEAPON','HISPANIC', 'BLACK', 'WHITE']
    uof_beat_yr = uof_beat_yr[final_cols]

    return uof_beat_yr


def go():

    print('Loading data...')
    uof_df_raw = load_data()

    print('Cleaning data...')
    uof_df_clean = clean_data(uof_df_raw)
    uof_df = add_mem_resp_indicators(uof_df_clean)

    print('Creating features...')
    uof_beatyear = add_features_at_beatyear(uof_df)

    uof_beatyear.to_csv('../data/features/use_of_force.csv', index=False)
    print('Generated features for use of force data')


if __name__ == "__main__":
    go()
