import matplotlib.pyplot as plt
import numpy as np

categories = ["48:24\ntf", "48:24\ntf-dsr", "48:24\ntf-dslr", "48:24\nff", "1:14(*)\ntf"]
times1 = [30, 29, 29, 30, 204]
times2 = [37, 37, 37, 36, 36]
# times3 = [30, 54, 36, 29]
# times4 = [66, 188, 62, 30]
# times5 = [36, 29, 39, 40]

# Set the width of the bars and their positions
width = 0.15  # Adjust the width to fit five bars side by side
x = np.arange(len(categories))

# Create the figure and axes
fig, ax = plt.subplots()

# Create bars for "categories 1," "categories 2," "categories 3," "categories 4," and "categories 5" side by side
# ax.bar(x - width*2, times1, width, label='tf', hatch='//')
# ax.bar(x - width, times2, width, label='tf-dsr-all', hatch='xx')
# ax.bar(x, times3, width, label='tf-dslr-all', hatch='\\\\')
# ax.bar(x + width, times4, width, label='dali', hatch='--')
# ax.bar(x + width*2, times5, width, label='ff', hatch='oo')

ax.bar(x - width/2, times1, width, label='GPU', hatch='//')
ax.bar(x + width/2, times2, width, label='CPU+Emulator', hatch='xx')

# Set labels, title, and legend
ax.set_xlabel('vCPUs local: vCPUs remote (offloading types)')
ax.set_ylabel('Time per rpoch (seconds)')
ax.set_title('Performance on GPU vs CPU+Emulator')
ax.set_xticks(x)
ax.set_xticklabels(categories)
ax.legend()

# Show the plot
plt.xticks(rotation=45)
plt.tight_layout()
# plt.show()




plt.savefig("figure6-fastflow-test.jpg", format="jpg")