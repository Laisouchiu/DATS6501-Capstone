#%%

import numpy as np
import pandas as pd 
import os

# %%
# Shelter datasets from Austin Animal Center

intakes1 = pd.read_csv('AAC_Intakes.csv')
intakes1.info()
intakes1.head()


#%%
outcomes1 = pd.read_csv('AAC_Outcomes.csv')
outcomes1.info()
outcomes1.head()

#%%
# Shelter Datasets from Long Beach (For Los Angeles Area)

in_and_out2 = pd.read_csv('LongBeach_Intakes_Outcomes.csv')
in_and_out2.info()
in_and_out2.head()

# %%
in_and_out3 = pd.read_csv('Soco_Intakes_Outcomes.csv')
in_and_out3.info()
in_and_out3.head()

# %%
