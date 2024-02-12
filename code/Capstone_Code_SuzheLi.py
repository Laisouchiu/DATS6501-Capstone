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

# Besides, we want to seperated the Neutered/Spayed information out as an another unique new feature column
# and also drop the original "Sex upon Outcome" column
outcomes1[['Neutered/Spayed Status', 'Sex']] = outcomes1['Sex upon Outcome'].str.split(' ', 1, expand=True)
outcomes1 = outcomes1.drop(['Sex upon Outcome'], axis=1)
outcomes1.info()

# Categorize 'Neutered' and 'Spayed' these 2 values both into one value 'Neutered/Spayed' ? 
# then the whole feature just has 2 values: Neutered/Spayed, Intact ?

## outcomes1['Neuter Status'] = outcomes1['Neuter Status'].str.contains('Neutered|Spayed').replace({True: 'Neutered/Spayed', False: 'Intact'})

#%%
## Transform the Datetime feature

# Drop the 'Outcome DateTime' column (Duplicate info with 'Outcome MonthYear')
outcomes1 = outcomes1.drop(['Outcome DateTime'], axis=1)

# Then we want to separate the month and year information
outcomes1[['Outcome Month', 'Outcome Year']] = outcomes1['Outcome MonthYear'].str.split(' ', 1, expand=True)
outcomes1 = outcomes1.drop(['Outcome MonthYear'], axis=1)

outcomes1.info()


#%%

## Transform the 'Age' numerical feature:
# Because the majority animals here are Dogs/Cats. 
# For larger mammals like dogs or cats, years are typically used, as their lifespans are longer.

# Check how many types of time units for the Age feature
outcomes1[['Age Number', 'Age Unit']] = outcomes1['Age upon Outcome'].str.split(' ', 1, expand=True)
# outcomes1.info()

print(f"Animal age feature has {outcomes1['Age Unit'].nunique()} unique values, which are:")
outcomes1['Age Unit'].value_counts()

# We need to confirm that those single time units like 'year' and 'week' are surely equal to 1. 
single_units = ['year', 'month', 'week', 'day']
for unit in single_units:
    test_df = outcomes1[outcomes1['Age Unit'] == unit] 
    number = test_df['Age Number'].unique()
    print(f"Animal Age feature with '{unit}' unit has {test_df['Age Number'].nunique()} unique values, which is equal to {number[0]}")
    # test_df.head()


#%%
# Create a function to convert Age units all into Year(s) and apply it.
def age_in_years(age):
    if 'year' in age or 'years' in age:
        return float(age.split()[0])
    elif 'month' in age or 'months' in age:
        return float(age.split()[0]) / 12
    elif 'week' in age or 'weeks' in age:
        return float(age.split()[0]) / 52 
    elif 'day' in age or 'days' in age:
        return float(age.split()[0]) / 365 
    else:
        return None

outcomes1['Age upon Outcome (Year)'] = outcomes1['Age upon Outcome'].apply(age_in_years)





# %%

# Roadblock1: How to manipulate so many categories for 'Breed' feature  
# Roadblock2: How to manipulate so many categoires for 'Color' feature

#%%
## Outcome Type manipulation (Roadblock3): 

# Outcome Type feature has 11 unique values, 
# which is more than our first though
# So we try to check and see how to manipulate it into only 2 options

outcomes1['Outcome Type'].value_counts()

## Based on the check, we can of course first erased some meaningless values:

# %%
