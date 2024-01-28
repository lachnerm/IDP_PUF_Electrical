import json
import matplotlib.pyplot as plt
import numpy as np

# Read data from JSON file
with open('results_miPUF.json', 'r') as file:
    data = json.load(file)

# Set the desired architectures
architectures = ["Aseeri", "Wisiol", "Custom5", "Custom11", "Custom13"]
architectures = ["Wisiol",  "Custom5"]

# Initialize dictionaries to store the average values
averages = {}

# Process the data and calculate the averages
for bits, bits_data in data.items():
    if bits not in ["64", "32"]:
        continue
    for m, m_data in bits_data.items():
        for start, start_data in m_data.items():
            for bl, bl_data in start_data.items():
                for k, k_data in bl_data.items():
                    for N, N_data in k_data.items():
                        for arch, arch_data in N_data.items():
                            if arch in architectures:
                                if bits not in averages:
                                    averages[bits] = {}
                                if k not in averages[bits]:
                                    averages[bits][k] = {}
                                if m not in averages[bits][k]:
                                    averages[bits][k][m] = {}
                                if arch not in averages[bits][k][m]:
                                    averages[bits][k][m][arch] = []
                                average_value = sum(arch_data.values()) / len(arch_data)
                                averages[bits][k][m][arch].append((int(N), average_value))

# Create a grid of subplots
num_bits = len(averages)
num_architectures = len(architectures)
fig, axs = plt.subplots(num_bits, num_architectures, figsize=(15, 10))

# Generate a colormap with 20 distinct colors
cmap = plt.cm.get_cmap('tab20')

for i, (bits, k_data) in enumerate(averages.items()):
    row_max_x = 0  # Maximum x value in the current row
    row_axes = []  # Axes within the current row

    for j, arch in enumerate(architectures):
        ax = axs[i, j]
        row_axes.append(ax)
        labels = []

        color_index = 0  # Index for color assignment within each subplot
        for k, m_data in k_data.items():
            for m, arch_data in m_data.items():
                if arch in arch_data.keys():
                    values = arch_data[arch]
                    # Sort the values based on N
                    values.sort(key=lambda x: x[0])
                    N_values = [value[0] for value in values]
                    average_values = [value[1] for value in values]

                    # Add the label to the list
                    label = f"k={k}, m={m}"
                    labels.append((int(k), int(m), label))

                    # Get the color for the specific (k, m) combination
                    combination_color = cmap(color_index)
                    color_index = (color_index + 1) % 20

                    # Plot the line with the assigned color
                    ax.plot(N_values, average_values, color=combination_color)

                    # Update the maximum x value within the current row
                    row_max_x = max(row_max_x, max(N_values))

        # Sort the labels by k and m values
        labels.sort()

        # Create a new sorted legend labels list
        legend_labels = [label for _, _, label in labels]

        # Set the plot title and labels for each subplot
        ax.set_title(f"Average Values for Architecture: {arch}, bits={bits}")
        ax.set_xlabel("N")
        ax.set_ylabel("Average Value")

        # Add sorted legend and grid
        ax.legend(legend_labels)
        ax.grid()

    # Set the x-axis limits for the current row
    for ax in row_axes:
        ax.set_xlim(0, row_max_x)

    # Set the y-axis limits for all subplots
    for ax in axs[i]:
        ax.set_ylim(0.4, 1)

# Adjust the spacing between subplots
plt.tight_layout()

# Show the plots
plt.show()
