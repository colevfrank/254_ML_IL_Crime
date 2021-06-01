# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %% [markdown]
# # Merges the four features csvs into one file

# %%
import pandas as pd
import numpy as np


# %%
complaints = pd.read_csv('../data/features/complaints.csv')
crime = pd.read_csv('../data/features/crime.csv')
isr = pd.read_csv('../data/features/isr.csv')
uof = pd.read_csv('../data/features/use_of_force.csv')
census = pd.read_csv('../data/features/census_demographics.csv')


# %%
#standardize col names
crime.rename(columns={'Beat':'BEAT', 'Year':'YEAR'}, inplace=True)
complaints.rename(columns={'COMPLAINT_YEAR':'YEAR'}, inplace=True)
census.rename(columns={'beat_num':'BEAT'}, inplace=True)


# %%
# Remove missing beats from complaints data and convert to numeric
complaints = complaints[complaints['BEAT']!='Unknown']
complaints['BEAT']= pd.to_numeric(complaints['BEAT'])


# %%
# prefix columns with dataset name to make merging tidier
complaints.rename(columns=lambda c: c if c in ['BEAT','YEAR'] else "COMPLAINTS_"+c, inplace=True)
crime.rename(columns=lambda c: c if c in ['BEAT','YEAR'] else "CRIME_"+c, inplace=True)
isr.rename(columns=lambda c: c if c in ['BEAT','YEAR'] else "ISR_"+c, inplace=True)
uof.rename(columns=lambda c: c if c in ['BEAT','YEAR'] else "UOF_"+c, inplace=True)
census.rename(columns=lambda c: c if c in ['BEAT','YEAR'] else "CENSUS_"+c, inplace=True)


# %%
# Convert beat to int for all data
for df in [complaints, crime, isr, uof, census]:
    df['BEAT'] = df['BEAT'].astype(int)


# %%
merged_df = pd.merge(complaints, crime, how='inner', on=['BEAT','YEAR'])
merged_df = pd.merge(merged_df, isr, how='inner', on=['BEAT', 'YEAR'])
merged_df = pd.merge(merged_df, uof, how='inner', on=['BEAT', 'YEAR'])
merged_df = pd.merge(merged_df, census, how='inner', on=['BEAT'])


# %%
# Write to disk
import os
if not os.path.exists("../data/features"):
    os.mkdir("../data/features")
merged_df.to_csv("../data/features/merged.csv", index=False)


# %%



