#!/usr/bin/env python
# coding: utf-8

# In[10]:


import pandas as pd
import seaborn as sns
import numpy as np


# # Complainants / Subjects

# In[244]:


# Load "by complaintant" dataset
comdata = pd.read_csv("../data/raw/Complaints/COPA_Cases_-_By_Complainant_or_Subject.csv",                    dtype={"LOG_NO":str,"CASE_TYPE":str})         .assign(DATETIME = lambda x: pd.to_datetime(x.COMPLAINT_DATE, format = "%m/%d/%Y %H:%M:%S %p"))         .assign(COMPLAINT_YEAR = lambda x: x.DATETIME.dt.year)
comdata['LOG_NO'] = comdata['LOG_NO'].astype(str)


# In[245]:


# XXX:  # Commented out because these may be multi-party complaints
# Discard literal duplicates
# comdata.drop_duplicates(inplace = True)


# In[246]:


# Drop all time columns except year, which is needed for aggregation
comdata.drop(labels=['COMPLAINT_DATE', 'COMPLAINT_HOUR',                      'COMPLAINT_DAY', 'COMPLAINT_MONTH', 'DATETIME'],              axis='columns',             inplace = True)


# In[247]:


# The question of which cases move through the system and how quickly is interesting, 
# but since we got rid of the "time" axis, its easier to assume all cases are "Closed" for now
comdata.drop(labels='CURRENT_STATUS', axis='columns', inplace=True)


# In[248]:


# For similar reasons, let's drop assignment until we read up on who sees which kind of cases
comdata.drop(labels='ASSIGNMENT', axis='columns', inplace=True)


# In[249]:


# For similar reasons, let's drop case_type until we decide we want to include these procedural details
comdata.drop(labels='CASE_TYPE', axis='columns', inplace=True)


# In[250]:


# XXX: Commented out because I think the 'duplicated' records actually refer to a party of complaintants

# Clean up duplicates
# logcounts = comdata['LOG_NO'].value_counts()
# duplogs = logcounts[logcounts > 1].index

# def clean_updated_unknowns(data, column_name, dups):
#     is_unknown = data[column_name] == 'Unknown'
#     is_dup = data['LOG_NO'].isin(dups)
#     return data[~(is_unknown & is_dup)]

# for col in comdata.columns:
#     comdata = clean_updated_unknowns(comdata, col, duplogs)
    
# Clean up duplicates
# logcounts = comdata['LOG_NO'].value_counts()
# duplogs = logcounts[logcounts > 1].index
# comdata[comdata['LOG_NO'] == duplogs[45]]


# In[251]:


# Create new indicator columns for the multi-categorical columns

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

comdata = listcolumn_pivot_wider(comdata, 'CURRENT_CATEGORY', 'COMPLAINT_CAT_')
comdata = listcolumn_pivot_wider(comdata, 'FINDING_CODE', 'COMPLAINT_FINDING_')
comdata = listcolumn_pivot_wider(comdata, 'RACE_OF_COMPLAINANT', 'COMPLAINANT_RACE_')
comdata = listcolumn_pivot_wider(comdata, 'SEX_OF_COMPLAINANT', 'COMPLAINANT_SEX_')
comdata = listcolumn_pivot_wider(comdata, 'AGE_OF_COMPLAINANT', 'COMPLAINANT_AGE_')


# In[252]:


# Pivot "beats" column longer: create more rows when complaint spans beats
comdata = comdata.assign(BEAT = comdata['BEAT'].str.replace("\s*", "", regex=True))                  .assign(BEAT = comdata['BEAT'].str.split("|"))                  .explode('BEAT')


# In[253]:


# Transform Police Shooting to numeric
comdata = comdata.assign(POLICE_SHOOTING = lambda x: x.POLICE_SHOOTING.map({'No':0.0, 'Yes':1.0}))


# In[254]:


# Aggregate
comdata_agg = comdata.drop(labels='LOG_NO', axis='columns').groupby(by=['BEAT','COMPLAINT_YEAR']).sum()


# In[255]:


# Write to disk
get_ipython().system('mkdir -p ../data/processed')
comdata_agg.to_csv("../data/processed/complaints.csv")


# # Officers

# In[199]:


# Load "by officer" dataset
offdata = pd.read_csv("../data/raw/Complaints/COPA_Cases_-_By_Involved_Officer.csv",                       dtype={"LOG_NO":str, "CASE_TYPE":str})             .assign(DATETIME = lambda x: pd.to_datetime(x.COMPLAINT_DATE, format = "%m/%d/%Y %H:%M:%S %p"))             .assign(COMPLAINT_YEAR = lambda x: x.DATETIME.dt.year)
offdata['LOG_NO'] = offdata['LOG_NO'].astype(str)


# In[200]:


# XXX: Commented out because these may be multi-party complaints
# Drop literal duplicates
# offdata.drop_duplicates(inplace=True)


# In[202]:


# Drop all time columns except year, which is needed for aggregation
offdata.drop(labels=['COMPLAINT_DATE', 'COMPLAINT_HOUR',                      'COMPLAINT_DAY', 'COMPLAINT_MONTH', 'DATETIME'],              axis='columns',             inplace = True)

# The question of which cases move through the system and how quickly is interesting, 
# but since we got rid of the "time" axis, its easier to assume all cases are "Closed" for now
offdata.drop(labels='CURRENT_STATUS', axis='columns', inplace=True)

# For similar reasons, let's drop assignment until we read up on who sees which kind of cases
offdata.drop(labels='ASSIGNMENT', axis='columns', inplace=True)

# For similar reasons, let's drop case_type until we decide we want to include these procedural details
offdata.drop(labels='CASE_TYPE', axis='columns', inplace=True)


# In[ ]:


# TODO: merge into by-complainant

