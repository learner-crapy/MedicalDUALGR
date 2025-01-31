import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import torch
from sklearn.preprocessing import normalize
from pymilvus import model

# Initialize the sentence transformer
sentence_transformer_ef = model.dense.SentenceTransformerEmbeddingFunction(
    model_name="all-distilroberta-v1",  # Specify the model name
    device='cuda:0',  # Specify the device to use
    normalize_embeddings=True  # This will help with consistency
)

file_path = "./cluster_analysis_sentence-transformer_all-distilroberta-v1.csv"

def encode_text(text, embedding_dim=512):
    vectors = sentence_transformer_ef.encode_documents([text])
    vectors = pad_or_truncate_vector(vectors, embedding_dim)
    return vectors

def pad_or_truncate_vector(vector, target_size=512):
    vector = np.array(vector)
    if vector.ndim == 1:
        vector = vector.reshape(1, -1)
    
    if vector.shape[1] < target_size:
        padding = np.zeros((vector.shape[0], target_size - vector.shape[1]))
        vector = np.concatenate([vector, padding], axis=1)
    elif vector.shape[1] > target_size:
        vector = vector[:, :target_size]
    
    # Normalize the vector
    vector = normalize(vector)
    return vector.squeeze()

# Read the CSV file
df = pd.read_csv(file_path)

def find_center(row_text):
    # Split the text into words if there are multiple
    words = str(row_text).split()
    
    if len(words) <= 1:
        return row_text.split('(')[0].strip()  # Return cleaned single word
    
    # Convert words to embeddings
    embeddings = np.array([encode_text(word) for word in words])
    
    # Calculate pairwise cosine similarities
    similarities = cosine_similarity(embeddings)
    
    # Find the word with highest average similarity to all other words
    avg_similarities = np.mean(similarities, axis=0)
    center_index = np.argmax(avg_similarities)
    
    # Return cleaned center word (removing parentheses)
    return words[center_index].split('(')[0].strip()

# Apply the function to each row in the 3rd column (index 2) and create 'center' column
df['center'] = df.iloc[:, 2].apply(find_center)

# Print results
print("Center words for each cluster:")
for i, row in df.iterrows():
    print(f"Cluster {i}: Original words: {row.iloc[2]}, Center: {row['center']}")

# Save the updated DataFrame back to the CSV file
df.to_csv(file_path, index=False)