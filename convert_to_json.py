import pandas as pd
import json


df = pd.read_csv("archive2/movies_metadata.csv")
df = df[["id", "title"]]
#take only half of the dataset
df = df[:10000]

# Convert the DataFrame to a list of dictionaries (JSON format)
movies_data = df.to_dict(orient="records")

# Write the JSON data to a file
with open("movies.json", "w") as json_file:
    json.dump(movies_data, json_file, indent=4)

print("CSV data has been successfully converted to JSON.")