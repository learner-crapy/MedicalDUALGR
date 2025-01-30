import pandas as pd

# Read the CSV file
df = pd.read_csv('../data/export.csv')

# Create entity dataframe (first 60580 rows)
df_entity = df.iloc[0:41397]
columns_to_extract = ['_id', '_labels', 'content', 'desc', 'name', 'semanticType']
df_filtered_entity = df_entity[columns_to_extract]

# Create relationship dataframe (from row 60581 onwards)
df_relationship = df.iloc[41397:]
relationship_columns = ['_start', '_end', '_type']
df_filtered_relationship = df_relationship[relationship_columns]

# Write to two separate CSV files with different columns
df_filtered_relationship.to_csv('../data/exports_re.csv', index=False)
df_filtered_entity.to_csv('../data/exports_en.csv', index=False)

print("Files have been created successfully!")
print(f"Number of entity rows: {len(df_filtered_entity)}")
print(f"Number of relationship rows: {len(df_filtered_relationship)}")
