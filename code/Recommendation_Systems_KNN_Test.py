# %%

import pandas as pd
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

# Create a simple example dataset with numerical 'Age upon Outcome'
data = {
    'Animal Name': ['Buddy', 'Kitty', 'Max', 'Luna', 'Rocky'],  # Animal names
    'Animal Type': ['Dog', 'Cat', 'Dog', 'Cat', 'Dog'],
    'Age upon Outcome': [3, 2, 1, 4, 2],  # Age in years
    'Breed': ['Labrador Retriever', 'Siamese', 'Labrador Retriever', 'Persian', 'Golden Retriever'],
    'Color': ['Black', 'White', 'Golden', 'Gray', 'Brown'],
    'Neutered/Spayed Status': ['Neutered', 'Spayed', 'Not neutered', 'Spayed', 'Neutered'],
    'Sex': ['Male', 'Female', 'Male', 'Female', 'Male']
}

# Assume the datasets have been cleaned

df = pd.DataFrame(data)
df1 = df.copy() # copy the dataset 

#%%
# Drop the 'Animal Name' column temporarily before encoding
animal_names = df.pop('Animal Name')

# Encode categorical variables
df_encoded = pd.get_dummies(df)

# Standardize numerical features
scaler = StandardScaler()
df_encoded[['Age upon Outcome']] = scaler.fit_transform(df_encoded[['Age upon Outcome']])

# Instantiate and fit k-NN model
k = 3  # Number of neighbors to consider
knn_model = NearestNeighbors(n_neighbors=k)
knn_model.fit(df_encoded)

#%%
# Function to recommend animals based on user preferences
def recommend_animals(user_preferences, knn_model, animal_data):
    # Create a DataFrame with all possible categories for each categorical feature
    all_categories = {}
    for feature in animal_data.columns:
        if feature != 'Age upon Outcome':
            all_categories[feature] = list(animal_data[feature].unique())
    all_categories_df = pd.DataFrame(all_categories)
    
    # Encode user preferences using one-hot encoding
    user_preferences_encoded = pd.get_dummies(pd.DataFrame(user_preferences))
    
    # Ensure all categories are present in user preferences encoded
    for col in all_categories_df.columns:
        if col not in user_preferences_encoded.columns:
            user_preferences_encoded[col] = 0
    
    # Standardize numerical features
    user_preferences_encoded[['Age upon Outcome']] = scaler.transform(user_preferences_encoded[['Age upon Outcome']])
    
    # Find k nearest neighbors
    distances, indices = knn_model.kneighbors(user_preferences_encoded)
    
    # Store recommended animals in a DataFrame
    recommended_animals = animal_data.iloc[indices[0]]
    
    return recommended_animals

#%%
# Example user preferences
user_preferences = {'Animal Type': ['Dog'], 'Age upon Outcome': [4], 'Breed': ['Labrador Retriever'], 'Color': ['Black'], 'Neutered/Spayed Status': ['Neutered'], 'Sex': ['Male']}

# Get recommended animals based on user preferences
recommended_animals_df = recommend_animals(user_preferences, knn_model, df_encoded)

# Add animal names back to the recommendations DataFrame
recommended_animals_df['Animal Name'] = animal_names[recommended_animals_df.index]

print("Recommended animals:")
for name in recommended_animals_df['Animal Name']:
    print(df1[df1['Animal Name']== name])
    
    
    
# %%
