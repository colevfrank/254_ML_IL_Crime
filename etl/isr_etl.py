# import packages
import pandas as pd
import numpy as np
import os

ISR_CSV1 = '../data/ISR/01-JAN-2016 to 28-FEB-2017 - ISR - JUV Redacted.csv'
ISR_CSV2 = '../data/ISR/29-FEB-2016 thru 16-JAN-2018 - ISR - JUV Redacted.csv'
ISR_CSV3 = '../data/ISR/ISR-1-Jan-2018--31-Dec-2019-Juv-Redacted.csv'

CAT_COLS = ['RACE_CODE_CD', 
            'ENFORCEMENT_TYPE_CD', 
            'CONTACT_TYPE_CD']
# enter binary cols to use - any others?
BINARY_COLS = ['TOTAL_COUNT', 
            'VEHICLE_INVOLVED_I', 
            'BODY_CAMERA_I', 
            'SEARCH_I', 
            'SEARCH_CONTRABAND_FOUND_I', 
            'PAT_DOWN_I', 
            'ENFORCEMENT_ACTION_TAKEN_I']

def load_data():
    '''
    load and concatenate data from csvs to single df
    '''
# set filepaths (3 files)
    csv_list = [ISR_CSV1, ISR_CSV2, ISR_CSV3]
    # Load all files and concatenate into one
    isr_dfs = []
    for csv in csv_list:
        new_df = pd.read_csv(csv, low_memory=False)
        isr_dfs.append(new_df)
    isr_raw_df = pd.concat(isr_dfs)
    return isr_raw_df

def clean_data(isr_raw_df):
    '''
    cleans df to remove duplicate records.
    '''
    # Remove redacted juvenile records
    isr_df = isr_raw_df[isr_raw_df['CONTACT_DATE']!='REDACTED']
    # Drop duplicates (Appear to be overlapping dates)
    isr_df = isr_df.drop_duplicates()

    isr_df['DATETIME'] = pd.to_datetime(isr_df['CONTACT_DATE'])
    #isr_df['DATE'] = isr_df['DATETIME'].dt.date
    #isr_df['MONTHYEAR'] = isr_df['DATETIME'].dt.to_period('M')
    isr_df['MODIFIED_DATETIME'] = pd.to_datetime(isr_df['MODIFIED_DATE'])

    # Sort by date of record modification (descending)
    isr_df = isr_df.sort_values(by='MODIFIED_DATETIME', ascending=False)

    # Keep only most recent record per card number
    isr_df = isr_df.drop_duplicates(subset = 'CARD_NO')

    # Add year as variable
    isr_df['YEAR'] = isr_df['DATETIME'].dt.year

    return isr_df


def create_features(isr_df, categ_cols, binary_cols):
    '''
    create final feature set, dropping nonessential variables and aggregating
    at police beat/year level.
    '''

    dummy_cols = pd.get_dummies(isr_df[categ_cols], 
                                columns = categ_cols, 
                                prefix = categ_cols).columns
    isr_df = pd.get_dummies(isr_df, 
                            columns = categ_cols, 
                            prefix = categ_cols)

    isr_df['TOTAL_COUNT'] = 1

    # convert from y/n to 1/0
    for col in binary_cols:
        isr_df[col] = np.where(isr_df[col]=='Y',1,isr_df[col])
        isr_df[col] = np.where(isr_df[col]=='N',0,isr_df[col]) 

    # combine into one list
    binary_cols.extend(dummy_cols)

    isr_beat_yr = isr_df.groupby(['DISTRICT', 'SECTOR', 'BEAT', 'YEAR']
                                )[binary_cols].sum().reset_index()

    # finalize dataset for export

    # Add hispanic indicator
    isr_beat_yr['CNT_HISPANIC'] = isr_beat_yr['RACE_CODE_CD_WBH'] + isr_beat_yr['RACE_CODE_CD_WWH']

    isr_beat_yr = isr_beat_yr[['DISTRICT', 
                               'SECTOR',
                               'BEAT',
                               'YEAR',
                               'TOTAL_COUNT',
                               'RACE_CODE_CD_BLK',
                               'RACE_CODE_CD_WHI',
                               'CNT_HISPANIC']]

    isr_beat_yr.columns = [['DISTRICT', 
                            'SECTOR',
                            'BEAT',
                            'YEAR',
                            'CNT_ISR_TOTAL',
                            'CNT_ISR_BLACK',
                            'CNT_ISR_WHITE',
                            'CNT_ISR_HISPANIC']]
    return isr_beat_yr


def go():
    '''
    Main function that runs all steps.
    '''
    print('Loading data...')
    isr_raw_df = load_data()
    
    print('Cleaning data...')
    isr_df = clean_data(isr_raw_df)

    print('Creating features...')
    isr_beat_yr = create_features(isr_df, CAT_COLS, BINARY_COLS)
    
    # make features directory if not present
    if not os.path.exists('../data/features'):
        os.mkdir('../data/features')

    print('Saving features...')
    isr_beat_yr.to_csv('../data/features/isr.csv', index=False)
    print('Generated features for ISR data')

if __name__ == "__main__":
    go()
