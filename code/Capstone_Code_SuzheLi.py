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
# Distribution of overall Dogs within 'Stay Long' threshold
plt.figure(figsize=(11,8))
sns.histplot(df_dogs1['Duration in Hours'], bins=24, kde=True)
plt.title('Distribution of Adopted Time (Dogs)')
plt.xlabel('Hour of the Day')
plt.ylabel('Frequency')
plt.show()

#%%
# Distribution of overall Cats within 'Stay Long' threshold
plt.figure(figsize=(11,8))
sns.histplot(x='Duration in Hours', data=df_cats1, bins=24, kde=True)
plt.title('Distribution of Adopted Time (Cats)')
plt.xlabel('Hour of the Day')
plt.ylabel('Frequency')
plt.show()

#%%

# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- #


#%%
# Neutered conditions

df_clean['Sex upon Outcome'].unique()

# " Neutered" and "Spayed" both refer to animals that have been surgically sterilized to prevent reproduction, but they apply to different genders:
# - Neutered typically refers to male animals that have had their testicles removed.
# - Spayed refers to female animals that have had their ovaries and usually their uterus removed.

# Intact Male: A male animal that has not been neutered; it still has its testicles.
# Intact Female: A female animal that has not been spayed; it still has its ovaries and usually the uterus.

#%%
# Create a new column for neutered status
df_clean['Neutered Status (Outcome)'] = df_clean['Sex upon Outcome'].apply(lambda x: 'Neutered' if 'Neutered' in x or 'Spayed' in x else 'Intact' if 'Intact' in x else 'Unknown')

# Create a new column for sex
df_clean['Sex (Outcome)'] = df_clean['Sex upon Outcome'].apply(lambda x: 'Male' if 'Male' in x else 'Female' if 'Female' in x else 'Unknown')

# Drop the original column
columns_to_drop = ['Sex upon Intake', 'Sex upon Outcome', 'Duration in Shelter']
df_clean = df_clean.drop(columns_to_drop, axis=1)

# Display the updated DataFrame
df_clean.info()

#%%
neutered_dogs = df_clean[(df_clean['Type'] == 'Dog') & (df_clean['Neutered Status (Outcome)'] == 'Neutered')]
intact_dogs = df_clean[(df_clean['Type'] == 'Dog') & (df_clean['Neutered Status (Outcome)'] == 'Intact')]
neutered_dogs.info()

#%%
neutered_cats = df_clean[(df_clean['Type'] == 'Cat') & (df_clean['Neutered Status (Outcome)'] == 'Neutered')]
intact_cats = df_clean[(df_clean['Type'] == 'Cat') & (df_clean['Neutered Status (Outcome)'] == 'Intact')]
neutered_cats.info()

#%%
neutered_dogs1 = neutered_dogs[neutered_dogs['Duration in Hours']<1000]
intact_dogs1 = intact_dogs[intact_dogs['Duration in Hours']<1000]

plt.figure(figsize=(25,15))

plt.subplot(1, 2, 1)
sns.histplot(neutered_dogs1['Duration in Hours'], bins=24, kde=True)
plt.title('Distribution of Adopted Time (Neutered Dogs)', fontsize=15)
plt.xlabel('Hour of the Day', fontsize=12)
plt.ylabel('Frequency', fontsize=12)

plt.subplot(1, 2, 2)
sns.histplot(intact_dogs1['Duration in Hours'], bins=24, kde=True)
plt.title('Distribution of Adopted Time (Intact Dogs)', fontsize=15)
plt.xlabel('Hour of the Day', fontsize=12)
plt.ylabel('Frequency', fontsize=12)

plt.tight_layout()
plt.show()

#%%
neutered_cats1 = neutered_cats[neutered_cats['Duration in Hours']<1000]
intact_cats1 = intact_cats[intact_cats['Duration in Hours']<1000]

plt.figure(figsize=(12,9))

plt.subplot(1, 2, 1)
sns.histplot(neutered_cats1['Duration in Hours'], bins=24, kde=True)
plt.title('Distribution of Adopted Time (Neutered Cats)', fontsize=15)
plt.xlabel('Hour of the Day', fontsize=12)
plt.ylabel('Frequency', fontsize=12)

plt.subplot(1, 2, 2)
sns.histplot(intact_cats1['Duration in Hours'], bins=24, kde=True)
plt.title('Distribution of Adopted Time (Intact Dogs)', fontsize=15)
plt.xlabel('Hour of the Day', fontsize=12)
plt.ylabel('Frequency', fontsize=12)

plt.tight_layout()
plt.show()


#%%

### Plus with Sex to see the resutls (Dogs): 

neutered_dogs_male = neutered_dogs1[neutered_dogs1['Sex (Outcome)'] == 'Male']
neutered_dogs_female = neutered_dogs1[neutered_dogs1['Sex (Outcome)'] == 'Female']

plt.figure(figsize=(20, 12)) 

# Male dogs subplot
plt.subplot(1, 2, 1)
sns.histplot(neutered_dogs_male['Duration in Hours'], bins=24, kde=True)
plt.title('Distribution of Duration (Neutered Male Dogs)', fontsize=15)
plt.xlabel('Hour of the Day', fontsize=12)
plt.ylabel('Frequency', fontsize=12)

# Female dogs subplot
plt.subplot(1, 2, 2) 
sns.histplot(neutered_dogs_female['Duration in Hours'], bins=24, kde=True)
plt.title('Distribution of Duration (Neutered Female Dogs)', fontsize=15)
plt.xlabel('Hour of the Day', fontsize=12)
plt.ylabel('Frequency', fontsize=12)

plt.tight_layout() 
plt.show()


#%%
### Plus with Sex to see the resutls (Cats): 

neutered_cats_male = neutered_cats1[neutered_cats1['Sex (Outcome)'] == 'Male']
neutered_cats_female = neutered_cats1[neutered_cats1['Sex (Outcome)'] == 'Female']

plt.figure(figsize=(20, 12)) 

# Male cats subplot
plt.subplot(1, 2, 1) 
sns.histplot(neutered_cats_male['Duration in Hours'], bins=24, kde=True)
plt.title('Distribution of Duration (Neutered Male Dogs)', fontsize=15)
plt.xlabel('Hour of the Day', fontsize=12)
plt.ylabel('Frequency', fontsize=12)

# Female cats subplot
plt.subplot(1, 2, 2) 
sns.histplot(neutered_cats_female['Duration in Hours'], bins=24, kde=True)
plt.title('Distribution of Duration (Neutered Female Dogs)', fontsize=15)
plt.xlabel('Hour of the Day', fontsize=12)
plt.ylabel('Frequency', fontsize=12)

plt.tight_layout() 
plt.show()

#%%

# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- #

#%%
# Adoption duration by different age of animal 

# <1 year age is considered as a Puppy
puppy_dogs = df_dogs[df_dogs['Age upon Outcome (Year)'] <= 1]
puppy_dogs1 = puppy_dogs[puppy_dogs['Duration in Hours'] < 1000]

# 1-7 year(s) age is considered as an Adult: 
adult_dogs = df_dogs[ (df_dogs['Age upon Outcome (Year)'] > 1) & (df_dogs['Age upon Outcome (Year)'] <= 7)]
adult_dogs1 = adult_dogs[adult_dogs['Duration in Hours'] < 1000]

# >7 years age is considered as a Senior
senior_dogs = df_dogs[df_dogs['Age upon Outcome (Year)'] > 7]
senior_dogs1 = senior_dogs[senior_dogs['Duration in Hours'] < 1000]

# Create visualizations (subplots) to compare the adopted time distribution of dogs with different aging
plt.figure(figsize=(20, 12)) 

# Puppy dogs subplot subplot
plt.subplot(1, 3, 1) 
sns.histplot(puppy_dogs1['Duration in Hours'], bins=24, kde=True)
plt.title('Distribution of Duration (Puppy Dogs)', fontsize=15)
plt.xlabel('Hour of the Day', fontsize=12)
plt.ylabel('Frequency', fontsize=12)

# Adult dogs suplot
plt.subplot(1, 3, 2) 
sns.histplot(adult_dogs1['Duration in Hours'], bins=24, kde=True)
plt.title('Distribution of Duration (Adult Dogs)', fontsize=15)
plt.xlabel('Hour of the Day', fontsize=12)
plt.ylabel('Frequency', fontsize=12)

# Senior dogs suplot
plt.subplot(1, 3, 3) 
sns.histplot(senior_dogs1['Duration in Hours'], bins=24, kde=True)
plt.title('Distribution of Duration (Senior Dogs)', fontsize=15)
plt.xlabel('Hour of the Day', fontsize=12)
plt.ylabel('Frequency', fontsize=12)


plt.tight_layout() 
plt.show()

#%%

# <1 year age is considered as a Puppy
kitten_cats = df_cats[df_cats['Age upon Outcome (Year)'] <= 1]
kitten_cats1 = kitten_cats[kitten_cats['Duration in Hours'] < 1000]

# 1-7 year(s) age is considered as an Adult: 
adult_cats = df_cats[ (df_cats['Age upon Outcome (Year)'] > 1) & (df_cats['Age upon Outcome (Year)'] <= 7)]
adult_cats1 = adult_cats[adult_cats['Duration in Hours'] < 1000]

# >7 years age is considered as a Senior
senior_cats = df_cats[df_cats['Age upon Outcome (Year)'] > 7]
senior_cats1 = senior_cats[senior_cats['Duration in Hours'] < 1000]

# Create visualizations (subplots) to compare the adopted time distribution of cats with different aging
plt.figure(figsize=(20, 12)) 

# Puppy dogs subplot subplot
plt.subplot(1, 3, 1) 
sns.histplot(kitten_cats1['Duration in Hours'], bins=24, kde=True)
plt.title('Distribution of Duration (Kitten Cats)', fontsize=15)
plt.xlabel('Hour of the Day', fontsize=12)
plt.ylabel('Frequency', fontsize=12)

# Adult dogs suplot
plt.subplot(1, 3, 2) 
sns.histplot(adult_cats1['Duration in Hours'], bins=24, kde=True)
plt.title('Distribution of Duration (Adult Cats)', fontsize=15)
plt.xlabel('Hour of the Day', fontsize=12)
plt.ylabel('Frequency', fontsize=12)

# Senior dogs suplot
plt.subplot(1, 3, 3) 
sns.histplot(senior_cats1['Duration in Hours'], bins=24, kde=True)
plt.title('Distribution of Duration (Senior Cats)', fontsize=15)
plt.xlabel('Hour of the Day', fontsize=12)
plt.ylabel('Frequency', fontsize=12)


plt.tight_layout() 
plt.show()

#%%
# Scatterplot to understand relationship between 2 numerical variables: Outcome Age & Shelter Duration
plt.figure(figsize=(10, 6))
sns.regplot(data=df_clean, 
            x='Age upon Outcome (Year)', 
            y='Duration in Hours', 
            scatter_kws={'alpha': 0.6})

plt.title('Scatter Plot of Outcome Age vs. Duration in Hours with Trend Line', fontsize=14)
plt.xlabel('Age upon Outcome (Years)', fontsize=12)
plt.ylabel('Duration in Shelter (Hours)', fontsize=12)

plt.show()

#%%

# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- #


#%%

df_clean['Intake Condition'].value_counts()

# Strategy for Re-categorization the 'Income Condition' column & Group by General Health Status:
# - Healthy: Include 'Normal'.
# - Medical Attention Needed: Include 'Injured', 'Sick', 'Medical', 'Med Attn', 'Med Urgent', 'Neurologic', 'Agonal', 'Congenital', 'Panleuk'.
# - Special Care Required: Include 'Nursing', 'Neonatal', 'Pregnant'.
# - Behavioral or Other Issues: Include 'Aged', 'Feral', 'Behavior', 'Other', 'Space'.

# Implement Re-categorization staregy by code:
condition_mapping = {
    'Normal': 'Healthy', 
    'Injured': 'Medical Attention Needed', 'Sick': 'Medical Attention Needed', 'Medical': 'Medical Attention Needed', 'Med Attn': 'Medical Attention Needed', 'Med Urgent': 'Medical Attention Needed', 'Neurologic': 'Medical Attention Needed', 'Agonal': 'Medical Attention Needed', 'Congenital': 'Medical Attention Needed', 'Panleuk': 'Medical Attention Needed',
    'Nursing': 'Special Care Required', 'Neonatal': 'Special Care Required', 'Pregnant': 'Special Care Required', 
    'Aged': 'Behavioral or Other Issues', 'Feral': 'Behavioral or Other Issues', 'Behavior': 'Behavioral or Other Issues', 'Other': 'Behavioral or Other Issues', 'Space': 'Behavioral or Other Issues'
}

# Apply the mapping to the 'Intake Condition' column
df_clean['Intake Condition (Health Status)'] = df_clean['Intake Condition'].map(condition_mapping).fillna('Unknown')

# Check the new value counts
df_clean['Intake Condition (Health Status)'].value_counts()

#%%
normal_health = df_clean[df_clean['Intake Condition (Health Status)'] == 'Healthy']
normal_health1 = normal_health[normal_health['Duration in Hours'] < 1000]

attention_needed = df_clean[df_clean['Intake Condition (Health Status)'] == 'Medical Attention Needed']
attention_needed1 = attention_needed[attention_needed['Duration in Hours'] < 1000]

special_care = df_clean[df_clean['Intake Condition (Health Status)'] == 'Special Care Required']
special_care1 = special_care[special_care['Duration in Hours'] < 1000]

behavioral_others = df_clean[df_clean['Intake Condition (Health Status)'] == 'Behavioral or Other Issues']
behavioral_others1 = behavioral_others[behavioral_others['Duration in Hours'] < 1000]


# Create visualizations (subplots) to compare the adopted time distribution of animals with different intake conditions
plt.figure(figsize=(20, 12)) 

plt.subplot(2, 2, 1) 
sns.histplot(normal_health1['Duration in Hours'], bins=24, kde=True)
plt.title('Distribution of Duration (Healthy Animals)', fontsize=15)
plt.xlabel('Hour of the Day', fontsize=12)
plt.ylabel('Frequency', fontsize=12)

plt.subplot(2, 2, 2) 
sns.histplot(attention_needed1['Duration in Hours'], bins=24, kde=True)
plt.title('Distribution of Duration (Animals need attention)', fontsize=15)
plt.xlabel('Hour of the Day', fontsize=12)
plt.ylabel('Frequency', fontsize=12)

plt.subplot(2, 2, 3) 
sns.histplot(special_care1['Duration in Hours'], bins=24, kde=True)
plt.title('Distribution of Duration (Animals need special care)', fontsize=15)
plt.xlabel('Hour of the Day', fontsize=12)
plt.ylabel('Frequency', fontsize=12)

plt.subplot(2, 2, 4) 
sns.histplot(behavioral_others1['Duration in Hours'], bins=24, kde=True)
plt.title('Distribution of Duration (Others)', fontsize=15)
plt.xlabel('Hour of the Day', fontsize=12)
plt.ylabel('Frequency', fontsize=12)

plt.tight_layout() 
plt.show()


#%%
# Part.2. 
# --- --- Objective: From adopter's perspectives, help to 'Increasing adopted persons'
# --- --- Visualize by location,
# --- --- Think about the argument about which if the location will affect the adoption rate or other things (e.g. rural to city)


#%%
# Split the 'Found Location' column on ' in ', which includes the space before and after 'in' to ensure clean splits.
df_clean[['Specific Location', 'City & State']] = df_clean['Found Location'].str.split(' in ', expand=True, n=1)

df_clean.info()

#%%
# find reasons behind why they stayed for too long, it's okay without hurting the resources on other animals
# just euthanized: not hurting others, saving resources; 
# Idenitfying these stay 'too long' animals, find reasons behind it and see if some animal can be euthanzied, 
# or if some institution (alternative agencies, like 'animal paradies' place, like non-profit orgnaization for animals) and sending them to their


#%%
# don't put in some simple charat, not some simple chart, 
# like a bar chart with only 2 bars, one bar 50% another 50%, that's so easy to interpret. 

# not some visualizations to help to just show what the data look like,
# but the visualizations that can help for the business. 


#%%


# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- #


#%%

# df_clean['Intake Type'].value_counts(): 
# Stray                 121600
# Owner Surrender        36335
# Public Assist          11663
# Abandoned               1449
# Euthanasia Request       251
# Wildlife                   1
# Name: Intake Type, dtype: int64

#%%

# Modeling? \
# adopted, find out the duration distribution 
# try to predict the duration (regression model)


