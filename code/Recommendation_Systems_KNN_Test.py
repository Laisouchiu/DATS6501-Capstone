# %%

import pandas as pd
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler



#%%
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

# For a content-based recommendation system, especially when dealing with animal adoption where the goal is to match animals with potential adopters based on the characteristics of the animals, several models stand out for their effectiveness, flexibility, and ease of interpretation. Here are some recommendations:

# 1. K-Nearest Neighbors (K-NN)
# Why: K-NN is intuitive and easy to implement for content-based filtering. It works by finding the most similar items to a given user's preferences.
# Best For: Situations where the dataset is not extremely large, and the dimensional space of features (characteristics) is manageable.
# 2. Decision Trees and Random Forests
# Why: These models can handle both numerical and categorical data well. They are useful for understanding which features (e.g., age, breed, color) are most important in determining a good match.
# Best For: Gaining insights into the factors influencing adoption preferences and providing interpretable recommendations.
# 3. Support Vector Machines (SVM)
# Why: SVM can be used for classification or regression tasks in high-dimensional spaces. With the kernel trick, SVM is capable of capturing complex relationships between features.
# Best For: Cases where the boundary between different classes of recommendations (e.g., highly recommended vs. not recommended) is not linear.
# 4. Neural Networks
# Why: Deep learning models, particularly those with embedding layers, can learn to represent categorical data in a dense, continuous space, capturing nuanced relationships between features.
# Best For: Large datasets with complex, non-linear relationships between features. Autoencoders, for example, can be particularly effective in learning compressed representations of items, useful for recommendation.
# 5. Matrix Factorization Techniques
# Why: Although more commonly associated with collaborative filtering, techniques like Singular Value Decomposition (SVD) can be applied to content-based systems by treating the problem as one of latent factor discovery based on item characteristics.
# Best For: Situations where you're interested in uncovering latent factors that explain adoption preferences.
# 6. Gradient Boosting Machines (GBMs)
# Why: Models like XGBoost, LightGBM, and CatBoost are powerful for handling tabular data, automatically handling feature interactions and non-linearities.
# Best For: Scenarios where model performance is paramount and the dataset has complex relationships that simpler models can't capture.
# Selection Considerations:
# Dataset Size and Complexity: Neural networks require large datasets to perform well without overfitting, whereas models like K-NN and decision trees can work well with smaller datasets.
# Feature Types: If your dataset includes a mix of numerical and categorical data, consider models that can natively handle this diversity, like decision trees or neural networks with embedding layers.
# Interpretability: If being able to explain why a particular recommendation was made is important, simpler models like decision trees might be preferred over more complex models like neural networks.
# Update Frequency: If the data or preferences change frequently, models that can be easily updated or retrained quickly, like K-NN or decision trees, might be advantageous.
# Final Recommendation:
# Start with simpler models like K-NN or decision trees to establish a baseline. These models are not only easier to interpret but also quicker to implement and iterate upon. As you understand your dataset and requirements better, experiment with more complex models like neural networks for potential improvements in recommendation quality. The choice of model should ultimately be guided by a combination of performance on your specific dataset, operational considerations, and the importance of interpretability in your application.




#%%
## Revised Code (Considering the unseen features situation) 

def recommend_animals(user_preferences, knn_model, animal_data, scaler):
    # Convert user preferences to DataFrame
    user_prefs_df = pd.DataFrame([user_preferences])
    
    # Encode user preferences using one-hot encoding
    # Ensure it matches the training data's structure
    user_prefs_encoded = pd.get_dummies(user_prefs_df)
    missing_cols = set(animal_data.columns) - set(user_prefs_encoded.columns)
    for c in missing_cols:
        user_prefs_encoded[c] = 0
    user_prefs_encoded = user_prefs_encoded[animal_data.columns]
    
    # Standardize numerical features for user preferences
    if 'Age upon Outcome' in user_preferences:
        user_prefs_encoded[['Age upon Outcome']] = scaler.transform(user_prefs_encoded[['Age upon Outcome']])
    
    # Find k nearest neighbors
    distances, indices = knn_model.kneighbors(user_prefs_encoded)
    
    # Retrieve recommended animals
    recommended_indices = indices[0]
    recommended_animals = df1.iloc[recommended_indices]
    
    return recommended_animals

# Usage:
user_preferences = {
    'Animal Type': 'Dog',
    'Age upon Outcome': 4,
    'Breed': 'Labrador Retriever',
    'Color': 'Pink',  # Assuming 'Blue' is an unseen category, where didn't appear in original datasets
    'Neutered/Spayed Status': 'Neutered',
    'Sex': 'Male'
}

# Get recommended animals based on user preferences
recommended_animals_df = recommend_animals(user_preferences, knn_model, df_encoded, scaler)

# Display the recommendations
print("Recommended animals:")
print(recommended_animals_df)
# %%
