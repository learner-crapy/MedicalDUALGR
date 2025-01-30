import scipy.io
import numpy as np
import matplotlib.pyplot as plt

# Load the MATLAB file
mat_data = scipy.io.loadmat('../data/ACM3025.mat')

# Print the keys to see what variables are stored in the file
print("Variables in the .mat file:")
print(mat_data.keys())

# Print basic information about each variable
for key in mat_data.keys():
    if not key.startswith('__'):  # Skip metadata keys
        print(f"\n{key}:")
        print(f"Type: {type(mat_data[key])}")
        print(f"Shape: {mat_data[key].shape}")
        print(f"Sample data:\n{mat_data[key][:5]}")  # Show first 5 elements/rows

# If there's numerical data, create a simple visualization
# This is just an example - adjust based on actual data structure
for key in mat_data.keys():
    if not key.startswith('__'):
        data = mat_data[key]
        if isinstance(data, np.ndarray) and data.size > 0:
            plt.figure(figsize=(10, 6))
            plt.title(f'First 100 elements of {key}')
            plt.plot(data.flatten()[:100])
            plt.show()
