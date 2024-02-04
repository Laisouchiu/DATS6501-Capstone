#%%

import numpy as np
import pandas as pd 
import os

# %%

intakes = pd.read_csv('AAC_Intakes.csv')
outcomes = pd.read_csv('AAC_Outcomes.csv')

# All datasets are from 2023 to 02/04/2024


#%%

intakes.info()

#%%
intakes.head()

#%%
outcomes.info()


# %%
outcomes.head()