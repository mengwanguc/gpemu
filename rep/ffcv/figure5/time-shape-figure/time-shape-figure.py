import numpy as np
import matplotlib.pyplot as plt

# data1 = np.random.randn(1000)  # Sample data for line 1
# data2 = np.random.randn(1000) + 1  # Sample data for line 2 (offset for illustration)

# Initialize an empty list to store the numbers
read_list = []

# Open the file in read mode
with open('read-time.txt', 'r') as file:
    # Read and convert each line to a floating-point number
    for line in file:
        line = line.strip()  # Remove leading/trailing whitespace
        if line:  # Check if the line is not empty
            try:
                number = float(line)  # Convert the line to a float
                read_list.append(number)  # Append the number to the list
            except ValueError:
                print(f"Warning: Skipped line '{line}' as it is not a valid float.")

# Initialize an empty list to store the numbers
crop_list = []

# Open the file in read mode
with open('crop-time.txt', 'r') as file:
    # Read and convert each line to a floating-point number
    for line in file:
        line = line.strip()  # Remove leading/trailing whitespace
        if line:  # Check if the line is not empty
            try:
                number = float(line)  # Convert the line to a float
                crop_list.append(number)  # Append the number to the list
            except ValueError:
                print(f"Warning: Skipped line '{line}' as it is not a valid float.")

resize_list = []

# Open the file in read mode
with open('resize-time.txt', 'r') as file:
    # Read and convert each line to a floating-point number
    for line in file:
        line = line.strip()  # Remove leading/trailing whitespace
        if line:  # Check if the line is not empty
            try:
                number = float(line)  # Convert the line to a float
                resize_list.append(number)  # Append the number to the list
            except ValueError:
                print(f"Warning: Skipped line '{line}' as it is not a valid float.")

normalize_list = []

# Open the file in read mode
with open('normalize-time.txt', 'r') as file:
    # Read and convert each line to a floating-point number
    for line in file:
        line = line.strip()  # Remove leading/trailing whitespace
        if line:  # Check if the line is not empty
            try:
                number = float(line)  # Convert the line to a float
                print(number)
                normalize_list.append(number)  # Append the number to the list
            except ValueError:
                print(f"Warning: Skipped line '{line}' as it is not a valid float.")

shape_reader_list = []
with open('shape-reader-crop.txt', 'r') as file:
    # Read and convert each line to a floating-point number
    for line in file:
        line = line.strip()  # Remove leading/trailing whitespace
        if line:  # Check if the line is not empty
            try:
                number = float(line)  # Convert the line to a float
                print(number)
                shape_reader_list.append(number)  # Append the number to the list
            except ValueError:
                print(f"Warning: Skipped line '{line}' as it is not a valid float.")



import matplotlib.pyplot as plt

# Sample time and shape data
# time_list = [0, 1, 2, 3, 4, 5]
# shape_list = [0, 1, 4, 9, 16, 25]

# Create a scatter plot with data points
plt.scatter(shape_reader_list, resize_list, marker='o')

# Add labels and a title
plt.xlabel('Shape')
plt.ylabel('Time')
plt.title('Time vs. Shape Scatter Plot (fn.resize)')

# Show the plot
plt.grid()  # Optionally, add a grid


plt.savefig("time-shape-fig-resize.jpg", format="jpg")

