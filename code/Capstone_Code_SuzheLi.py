#%%

import numpy as np
import pandas as pd 
import os

# %%
# Shelter datasets from Austin Animal Center

aac_intakes = pd.read_csv('AAC_Intakes.csv')
aac_intakes.info()
aac_intakes.head()


#%%
aac_outcomes = pd.read_csv('AAC_Outcomes.csv')
aac_outcomes.info()
aac_outcomes.head()

#%%
# Shelter Datasets from Long Beach (For Los Angeles Area, South California)
in_and_out2 = pd.read_csv('LongBeach_Intakes_Outcomes.csv')
in_and_out2.info()
in_and_out2.head()

#%%
# Shelter Datasets from Sonoma County (For Bay Area, North California)
in_and_out3 = pd.read_csv('Soco_Intakes_Outcomes.csv')
in_and_out3.info()
in_and_out3.head()

# %%

#%%
in_and_out2.columns

#%%
in_and_out3.columns


#%%

############################### Austin Animal Center Outcomes ######################################


# Some EDA Example: https://www.kaggle.com/datasets/aaronschlegel/austin-animal-center-shelter-outcomes-and/code 

## Features needed to be removed: Animal ID, Name, DateBirth, Outcome Subtype
## Check unique values' count for these features: Outcome Type, Animal Type, Sex upon Outcome, Breed, Color
## Features needed manipulations: Outcome Type, Sex Upon Outcome, Age upon Outcome

#%%
# Basic check
aac_outcomes.info()
aac_outcomes.head()

# %%
# Removing useless features
remove = ['Animal ID', 'Name', 'Date of Birth', 'Outcome Subtype']
outcomes1 = aac_outcomes.drop(remove, axis=1)
outcomes1.info()

# %%
# Renaming features for better understanding
new_names = {"DateTime":"Outcome DateTime", "MonthYear":"Outcome MonthYear"}
outcomes1 = outcomes1.rename(columns=new_names)
outcomes1.info()

# %%
# Checking unique values for some features and decide the manipulation methods

unique_check1 = ['Outcome Type', 'Animal Type', 'Sex upon Outcome', 'Breed', 'Color']

for feature in unique_check1:
    print(f" '{feature}' feature has {outcomes1[feature].nunique()} unique values.")


#%%

## Animal Type manipulation

outcomes1['Animal Type'].value_counts()

# Because most of the Animals are just Dog and Cat,
# we want to categorize 'Bird' and 'Livestock' into 'Other' category too
outcomes1['Animal Type'] = outcomes1['Animal Type'].replace({'Bird': 'Other', 
                                                             'Livestock': 'Other'})
outcomes1['Animal Type'].value_counts()


#%%

## Sex upon Outcome manipulation

outcomes1['Sex upon Outcome'].value_counts()

# First we want remove the 'Unknown', 
# only 8% of total datasets, which is meanlingless to our analysis
##(outcomes1['Sex upon Outcome'] == 'Unknown').sum()
outcomes1 = outcomes1[outcomes1['Sex upon Outcome'] != 'Unknown']
outcomes1['Sex upon Outcome'].value_counts()

#%%
# Besides, we want to seperated the Neutered/Spayed information out as an another unique new feature column
outcomes1[['Neutered/Spayed Status', 'Sex']] = outcomes1['Sex upon Outcome'].str.split(' ', 1, expand=True)
outcomes1 = outcomes1.drop(['Sex upon Outcome'], axis=1)
outcomes1.info()

# Categorize 'Neutered' and 'Spayed' these 2 values both into one value 'Neutered/Spayed' ? 
# then the whole feature just has 2 values: Neutered/Spayed, Intact ?

# %%

## Outcome Type manipulation (Roadblock): 

# Outcome Type feature has 11 unique values, 
# which is more than our first though
# So we try to check and see how to manipulate it into only 2 options

outcomes1['Outcome Type'].value_counts()

## Based on the check, we can of course first erased some meaningless values:

# %%
