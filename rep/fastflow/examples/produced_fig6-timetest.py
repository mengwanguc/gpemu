
import matplotlib.pyplot as plt
import numpy as np

categories = ["48:24", "1:14(*)"]
# times1 = [37, 38]
# times2 = [204, 36]

times1 = [30, 204]
times2 = [37, 39]
times3 = [37, 37]
times4 = [37, 37]
times5 = [36, 37]

# Set the width of the bars and their positions
width = 0.15
x = np.arange(len(categories))

# Create the figure and axes
fig, ax = plt.subplots()

# Create bars for "categories 1" and "categories 2" side by side with different hatch patterns
# ax.bar(x - width/2, times1, width, label='tf(GPU)', hatch='//')
# ax.bar(x + width/2, times2, width, label='tf(CPU+Emulator)', hatch='xx')

ax.bar(x - width*2, times1, width, label='tf(GPU)', hatch='//')
ax.bar(x - width, times2, width, label='tf(CPU+Emulator)', hatch='xx')
ax.bar(x, times3, width, label='tf-dsr-all(CPU+Emulator)', hatch='\\\\')
ax.bar(x + width, times4, width, label='tf-dslr-all(CPU+Emulator)', hatch='--')
ax.bar(x + width*2, times5, width, label='ff(CPU+Emulator)', hatch='oo')

# Set labels, title, and legend
ax.set_xlabel('vCPUs local GPU: vCPUs CPU remote')
ax.set_ylabel('Time per epoch (seconds)')
ax.set_title('Performance when using GPU vs emulator+CPU')
ax.set_xticks(x)
ax.set_xticklabels(categories)
ax.legend()

# Show the plot
plt.xticks(rotation=45)
plt.tight_layout()
# plt.show()


plt.savefig("figure6-fastflow-test.jpg", format="jpg")