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
age_means = df_clean['Age upon Outcome (Year)'].mean()


#%%
euthanized = df_clean[df_clean['Outcome Type'] == 'Euthanasia']

#%%
euthanized_mean = euthanized['Age upon Outcome (Year)'].mean()

#%%
duration_mean = euthanized['Duration in Shelter'].mean()


#%%
# Visualization: 
import seaborn as sns
import matplotlib.pyplot as plt

sns.set_theme(style='darkgrid')

# Visualizations:
# Part 1. 
# --- --- Objective: From animal's perspective, help to 'See which animals needs more help and Reduce animals stayed in the shelters' 
# --- --- Sub-category them, then print out thir adopted time's distribution
# --- --- In order to see which category needs more help 


# Part.2. 
# --- --- Objective: From adopter's perspectives, help to 'Increasing adopted persons'
# --- --- Visualize by location,
# --- --- Think about the argument about which if the location will affect the adoption rate or other things (e.g. rural to city)


#%%
# Convert the datetime into total hours 
df_clean['Duration in Hours'] = df_clean['Duration in Shelter'].dt.total_seconds() / 3600
df_clean['Duration in Shelter'].head(5)

#%%
df_clean['Type'].value_counts()

#%%
df_dogs = df_clean[df_clean['Type'] == 'Dog']
# df_dogs['Type'].value_counts()

df_cats = df_clean[df_clean['Type'] == 'Cat']
# df_cats['Type'].value_counts()

#%%
sns.histplot(df_dogs['Duration in Hours'], bins=24, kde=True)
plt.title('Distribution of Dogs Adopted Time')
plt.xlabel('Hour of the Day')
plt.ylabel('Frequency')
plt.grid()
plt.show()

#%%
sns.histplot(df_cats['Duration in Hours'], bins=24, kde=True)
plt.title('Distribution of Adopted Time (Cats)')
plt.xlabel('Hour of the Day')
plt.ylabel('Frequency')
plt.grid()
plt.show()


#%%
# 根据上面的print出来来看，因为如果全部print出来，很多outliers，确实有相当一部分动物
# 所以根据这点，第一可以先defined一个stay in shelter long的threshold然后筛选出threshold以内的这部分animal进行distribution的查看；
# 最重要的是找到每种types的animals的most frequency的那部分adopted times
# 其次那些stay in the shelter long的threshold以外的animals，对比看看那种动物stay in the shelter long的占比更多以及distribution如何？

#%%
df_dogs1 = df_dogs[df_dogs['Duration in Hours']<1000]
df_dogs1.info()

#%%
df_cats1 = df_cats[df_cats['Duration in Hours']<1000]
df_cats1.info()

#%%
# Distribution of Dogs within 'Stay Long' threshold
plt.figure(figsize=(11,8))
sns.histplot(df_dogs1['Duration in Hours'], bins=24, kde=True)
plt.title('Distribution of Adopted Time (Dogs)')
plt.xlabel('Hour of the Day')
plt.ylabel('Frequency')
plt.show()

#%%
# Distribution of Cats within 'Stay Long' threshold
plt.figure(figsize=(11,8))
sns.histplot(x='Duration in Hours', data=df_cats1, bins=24, kde=True)
plt.title('Distribution of Adopted Time (Cats)')
plt.xlabel('Hour of the Day')
plt.ylabel('Frequency')
plt.show()

#%%


#%%
# Modeling? \
# adopted, find out the duration distribution 
# try to predict the duration (regression model)

#%%



#%%
# Threshold? : 
# 1). Generally, an animal is considered to have a long shelter stay 
#     if it remains unadopted past the average time animals are usually adopted within that facility or community; 
# 2). Some animals may consider a stay of 2-3 months as long because of their income situation or that shelter is a Short-term shelters, 
#     and the goal is to find homes for animals as quickly as possible, 
#     but some Long-term shelters may have a different perspective, 
#     where animals can stay for years if necessary, making a "long" stay a much more extended period.


# Ask: Longest period animal can stay in the shelter

# For now, because we are using the dataset from Austin Animal Center, 
# which is a big organization, and also this project is for a comprehensive goal for every animal shelters, 
# so maybe using the Average adopted time as the threshold in my data pipeline would be a nice choice? 




