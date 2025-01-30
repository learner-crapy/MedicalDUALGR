file_list = [f"./result/thread_{i}/silhouette_scores/thread_{i}_sentence-transformer_all-distilroberta-v1.csv" for i in range(0, 9)]

# Initialize empty lists for x and y coordinates
x_coords = []
y_coords = []

# Read each CSV file and extract the data
for file_path in file_list:
    with open(file_path, 'r') as file:
        # Skip header if it exists
        next(file, None)
        for line in file:
            # Split the line by comma and extract 2nd and 3rd columns
            values = line.strip().split(',')
            x_coords.append(float(values[1]))  # 2nd column
            y_coords.append(float(values[2]))  # 3rd column
            print(x_coords, y_coords)

# Import matplotlib
import matplotlib.pyplot as plt

# Create the line plot
plt.figure(figsize=(10, 6))
plt.plot(x_coords, y_coords, '-', linewidth=1)  # '-' specifies a solid line

# Add labels and title
plt.xlabel('X Values (2nd Column)')
plt.ylabel('Y Values (3rd Column)')
plt.title('Line Plot of CSV Data')

# Add grid
plt.grid(True, linestyle='--', alpha=0.7)

# Show the plot
plt.show()
