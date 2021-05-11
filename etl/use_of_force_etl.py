import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

categ_cols = ['SUBJECT_RACE', 'SUBJECT_GENDER', 'SUBJECT_ARMED_DESC']

binary_cols = ['TOTAL_COUNT', 
               'MEMRESP_FIREARM_I',
               'MEMRESP_FM',
               'MEMRESP_CT',
               'MEMRESP_RWOW',
               'MEMRESP_RWW',
               'MEMRESP_IMPACT_WEAP_I',
               'SUBAC_DEADLY_FORCE_I', 
               'SUBJECT_ARMED_I',
               'SUBAC_ATTACK_NW_I',
               'SUBAC_ATTACK_WW_I']


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
    '''
    #add dummies
    dummy_cols = pd.get_dummies(df[categ_cols], columns = categ_cols, prefix = categ_cols).columns
    df = pd.get_dummies(df, columns = categ_cols, prefix = categ_cols)

    df['TOTAL_COUNT'] = 1

    for col in binary_cols:
        df[col] = np.where(df[col]=='Y',1,df[col])
        df[col] = np.where(df[col]=='N',0,df[col])

    binary_cols.extend(dummy_cols)
    uof_beat_yr = df.groupby(['DISTRICT', 'BEAT', 'YEAR'])[binary_cols].sum().reset_index()

    return uof_beat_yr


def go():

    uof_df_raw = load_data()

    uof_df_clean = clean_data(uof_df_raw)

    uof_df = add_mem_resp_indicators(uof_df_clean)

    uof_beatyear = add_features_at_beatyear(uof_df)
    
    uof_beatyear.to_csv('../data/features/use_of_force.csv')
    print('Generated features for use of force data')


if __name__ == "__main__":
    go()
