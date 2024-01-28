#%%
import json
import os
import shutil

#%%

def save_to_json(data, output_file_path):
    with open(output_file_path, 'w') as output_file:
        json.dump(data, output_file, indent=2)

semester2code = { "sp":"01", "spr":"01", "spring":"01", "su":"02", "sum":"02", "summer":"02", "fa":"03", "fall":"03"}
thisfilename = os.path.basename(__file__) # should match _ver for version, ideally 3-digit string starting as "000", up to "999"

data_to_save = \
    {
        # -----------------------------------------------------------------------------------------------------------------------
        "Version":
            """030""",
        # -----------------------------------------------------------------------------------------------------------------------
        "Year":
            """2024""",
        # -----------------------------------------------------------------------------------------------------------------------
        "Semester":
            """Spring""",
        # -----------------------------------------------------------------------------------------------------------------------
        "project_name":
            """U.S Sheltered Companion Animals Data Analysis""",
        # -----------------------------------------------------------------------------------------------------------------------
        "Objective":
            """ 
            Based on the record of National Council on Pet Population Study and Policy (NCPPSP) and the American Society for the Prevention of Cruelty to Animals (ASPCA), 
            each year U.S. animal shelters receive around 6.5 million companion animals and approximately 1.5 million of them will face euthanasia. 
            The primary goal of this project is to dig into animal adoption data, boost public awareness about animal care, and aim to bring down the high euthanasia rates in U.S. animal shelters. 
            This will make citizens learn more info about sheltered animals, and help citizens make more informed decisions when considering pet adoption. 
            """,
        # -----------------------------------------------------------------------------------------------------------------------
        "Dataset":
            """
            We want to use the open source datasets from Austin Animal Center
            https://data.austintexas.gov/browse?limitTo=datasets&q=pet 
            , which included animal intakes and outcomes record data, stray map, and affordable housing directory. 
            """,
        # -----------------------------------------------------------------------------------------------------------------------
        "Rationale":
            """
            This project is going to mitigate high euthanasia rates, and help to enhance the well-being of both animals and communities.
            """,
        # -----------------------------------------------------------------------------------------------------------------------
        "Approach":
            """
            I plan on approaching this capstone through several steps:  

            1. Extract sheltered animals data (2013 ~ 2023) from Austin Animal Center and integrate the related data and clean the data. 
            2. Conduct Exploratory Data Analysis (EDA) and especially Data Visualizations to learn and  understand the Animal Adoption pattern and Public Awareness.  
            3. Develop Predictive Model(s) to predict if an animal will be adopted or euthanized. 
            4. Develop a Recommendation System to citizens who are interested in adopting pets if possible. 
            """,
        # -----------------------------------------------------------------------------------------------------------------------
        "Timeline":
            """
            This a rough time line for this project:  
        
            - (1 weeks) Exploratory Data Analysis (EDA) and initial insights
            - (2 weeks) Data Visualization (Dashboard maybe)
            - (2 weeks) Feature Importance and Selection 
            - (3 weeks) Model Development and Training 
            - (2 weeks) Evaluation and Fine-tuning of Models
            - (1 week) Compiling Results
            - (1 week) Final Video Presentation
            - (1 week) Writing up Paper and Final Submission
            """,
        # -----------------------------------------------------------------------------------------------------------------------
        "Expected Number Students":
            """
            For this project, we plan to have 3 students maximum to work on it.  
            """,
        # -----------------------------------------------------------------------------------------------------------------------
        "Possible Issues":
            """
            We may face 2 challenging situations during this project:
            1. First challenge is the data preprocessing section, which are integrating or merging multiple relational data, 
               and then cleaning the integrated data; 
            2. Second challenge part maybe developing the recommendation system, because may not have enough relevant data.
            """,
        # -----------------------------------------------------------------------------------------------------------------------
        "Proposed by": "Suzhe Li, Guoshan Yu",
        "Proposed by email": "suzhel@gwmail.gwu.edu, guoshanyu@gwmail.gwu.edu ",
        "instructor": "Professor Edwin Lo",
        "instructor_email": "edwinlo@gwu.edu",
        "github_repo": "https://github.com/Laisouchiu/DATS6501-Capstone",
        # -----------------------------------------------------------------------------------------------------------------------
    }
os.makedirs(
    os.getcwd() + f'{os.sep}Proposals{os.sep}{data_to_save["Year"]}{semester2code[data_to_save["Semester"].lower()]}{os.sep}{data_to_save["Version"]}',
    exist_ok=True)
output_file_path = os.getcwd() + f'{os.sep}Proposals{os.sep}{data_to_save["Year"]}{semester2code[data_to_save["Semester"].lower()]}{os.sep}{data_to_save["Version"]}{os.sep}'
save_to_json(data_to_save, output_file_path + "input.json")
shutil.copy(thisfilename, output_file_path)
print(f"Data saved to {output_file_path}")
