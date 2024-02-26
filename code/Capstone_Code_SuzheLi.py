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
aac_in_out = pd.merge(aac_intakes, aac_outcomes, on='Animal ID', how='inner')
aac_in_out.info()
#%%

#%%
columns_dropped = ['Animal ID', 'Outcome Subtype', # Useless features for our analysis 
                   'Name_y', 'Animal Type_y', 'Breed_y', 'Color_y', # Duplicated features after merged joining
                   'MonthYear_x', 'MonthYear_y'] # Duplicate information with Datetime 
aac_in_out = aac_in_out.drop(columns_dropped, axis=1)
aac_in_out.info()

#%%
aac_in_out.rename(columns={'Name_x' : 'Name', 'Animal Type_x': 'Type', 
                           'Breed_x':'Breed', 'Color_x':'Color', 
                           'DateTime_x':'Income_Datetime', 'DateTime_y':'Outcome_Datetime',  
                           'MonthYear_x':'Income_MonthYear', 'MonthYear_y':'Outcome_MonthYear'}, 
                  inplace=True)
aac_in_out.info()

#%%
new_column_order = [
    'Name', 'Type', 'Breed', 'Color', 'Date of Birth', 'Age upon Intake',
    'Income_Datetime', 'Found Location', 'Intake Condition', 'Intake Type', 'Sex upon Intake',
    'Outcome_Datetime', 'Sex upon Outcome', 'Age upon Outcome', 'Outcome Type'
]

# Reorder the DataFrame columns
df = aac_in_out[new_column_order]
df.info()
df.head(10)

#%%
datetime_columns = ['Income_Datetime', 'Outcome_Datetime']
for column in datetime_columns:
    df[column] = pd.to_datetime(df[column], 
                                format='%m/%d/%Y %I:%M:%S %p'
                                )
df.info()
df.head(10)

#%%
wrong = df[df['Outcome_Datetime'] <= df['Income_Datetime']]
wrong.info()

#%%
# We got some portion of wrong datetime where the income datetime is later than the outcome datetime, 
# which is not reasonable, so we need to subset them out
df1 = df[df['Outcome_Datetime'] > df['Income_Datetime']]
df1.info()

#%%
df1.isnull().sum()

#%%
sex_nulls = df1[df1['Sex upon Outcome'].isnull()]
sex_nulls
# We can see the null values from sex upon intakes also null in sex upon outcome
# so we can just subset out these null values

#%% 
df_clean = df1[df1['Sex upon Outcome'].notnull()]
df_clean.isnull().sum()
#df_clean.info()

#%%
df_clean['Duration in Shelter'] = df_clean['Outcome_Datetime'] - df_clean['Income_Datetime']
df_clean.info()


#%%
# We subset out the other types of animal but focus on the majority type of shelter animal, dog and cat
df_clean['Type'].value_counts()
df_clean = df_clean[(df_clean['Type'] == 'Dog') | (df_clean['Type'] == 'Cat')]
df_clean.info()


#%%
df_clean.isnull().sum()

#%%
# After cleaning and subsets, now we only have Name and Outcome Type column have null values.
# For Name, it's just for better identification to know which specific animal needs more care, which won't affect our analysis, so we just leave these nulls
# For Outcome Type, because we only have 36 null among 171335 observations, and it doesn't help our analysis at all, so we subset them out too.
df_clean = df_clean[df_clean['Outcome Type'].notnull()]
df_clean.isnull().sum()
# Now we can see the dataset is clean

#%%
# Remove duplicate rows
df_unique = df_clean.drop_duplicates(keep=False)
df_unique.info()



#%%
df_clean['Outcome Type'].value_counts()


#%%
## Transform the 'Age' numerical feature:
# Because the majority animals here are Dogs/Cats. 
# For larger mammals like dogs or cats, years are typically used, as their lifespans are longer.

# Check how many types of time units for the Age feature
df_clean[['Age Number', 'Age Unit']] = df_clean['Age upon Outcome'].str.split(' ', 1, expand=True)
# outcomes1.info()

print(f"Animal age feature has {df_clean['Age Unit'].nunique()} unique values, which are:")
df_clean['Age Unit'].value_counts()

#%%
# We need to confirm that those single time units like 'year' and 'week' are surely equal to 1. 
single_units = ['year', 'month', 'week', 'day']
for unit in single_units:
    test_df = df_clean[df_clean['Age Unit'] == unit] 
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

df_clean['Age upon Outcome (Year)'] = df_clean['Age upon Outcome'].apply(age_in_years)
df_clean['Age upon Intake (Year)'] = df_clean['Age upon Intake'].apply(age_in_years)

df_clean = df_clean.drop(columns=['Age upon Intake', 'Age upon Outcome', 'Age Unit', 'Age Number'], axis=1)
df_clean.info()



#%%

## Sex upon Outcome manipulation

df_clean['Sex upon Outcome'].value_counts()

# First we want remove the 'Unknown', 
# only 8% of total datasets, which is meanlingless to our analysis
##(outcomes1['Sex upon Outcome'] == 'Unknown').sum()
df_clean = outcomes1[outcomes1['Sex upon Outcome'] != 'Unknown']
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
aac_outcomes.info()
# Date of birth as income? 


#%%
aac_intakes.info()

#%%

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
## Transform the Datetime feature

# Drop the 'Outcome DateTime' column (Duplicate info with 'Outcome MonthYear')
outcomes1 = outcomes1.drop(['Outcome DateTime'], axis=1)

# Then we want to separate the month and year information
outcomes1[['Outcome Month', 'Outcome Year']] = outcomes1['Outcome MonthYear'].str.split(' ', 1, expand=True)
outcomes1 = outcomes1.drop(['Outcome MonthYear'], axis=1)

outcomes1.info()





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
in_and_out3.info()
# %%


#%%

## Example modeling code
# animal type, sex, and neutered/spayed features can be encoded as 1 or 0 directly
# for breed and color features, like the conversion project from机构, 对它进行nlp的预处理 by unigarm and bigram, 每种不同颜色是一个vector (1 or 0), 然后再encoded?


