import matplotlib.pyplot as plt
import numpy as np

categories = ["48:24\ngpu", "48:24\ncpu+emu", "1:14\ngpu", "1:14\ncpu+emu"]
times1 = [35, 32, 159, 150]
times2 = [34, 32, 34, 34]
times3 = [37, 32, 39, 35]
# times4 = [66, 188, 62, 30]
times5 = [29, 31, 38, 35]

# Set the width of the bars and their positions
width = 0.15  # Adjust the width to fit five bars side by side
x = np.arange(len(categories))

# Create the figure and axes
fig, ax = plt.subplots()

# Create bars for "categories 1," "categories 2," "categories 3," "categories 4," and "categories 5" side by side
ax.bar(x - width, times1, width, label='tf', hatch='//')
ax.bar(x, times2, width, label='tf-dsr-all', hatch='xx')
ax.bar(x + width, times3, width, label='tf-dslr-all', hatch='\\\\')
# ax.bar(x + width, times4, width, label='dali', hatch='--')
ax.bar(x + 2*width, times5, width, label='ff', hatch='oo')

# Set labels, title, and legend
ax.set_xlabel('vCPUs local GPU: vCPUs CPU remote')
ax.set_ylabel('Time per rpoch (seconds)')
ax.set_title('Performance in various workloads and resource environments ')
ax.set_xticks(x)
ax.set_xticklabels(categories)
ax.legend()

# Show the plot
plt.xticks(rotation=45)
plt.tight_layout()
# plt.show()




plt.savefig("figure6-fastflow-gpu_cpu_emulator.jpg", format="jpg")