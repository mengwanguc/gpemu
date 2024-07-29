import matplotlib.pyplot as plt
import pandas as pd
fig = plt.gcf()
columns = ['alexnet','renet18','mobilenet_v2','resnet50']
df = pd.read_csv('time5.dat', sep='\t', header=None, names=columns)

alexnet = df['alexnet'].tolist()
renet18 = df['renet18'].tolist()
mobilenet_v2 = df['mobilenet_v2'].tolist()
resnet50 = df['resnet50'].tolist()
# Sample data for three lines
x = [1, 2, 3, 6, 12, 24]
y = [20,40,60,80,100]
y1 = alexnet
y2 = renet18
y3 = mobilenet_v2
y4 = resnet50

# Plotting the lines
plt.plot(x, y1, marker='o', color = "orange", label='alexnet')
plt.plot(x, y2, marker='s', color = "red", label='renet18')
plt.plot(x, y3, marker='^', color = "blue", label='mobilenet_v2')
plt.plot(x, y4, marker='*', color = "green", label='resnet50')

# Adding labels and title
plt.xlabel('Numer of CPU cores per GPU', fontsize=17)
plt.ylabel('Number of images per second', fontsize=17)
plt.title('Impact of CPU cores on training (Emulator)\nOn HDD + resnet18 + 1GPU (emulated p100)', fontsize=17)
plt.xticks(x, fontsize=15)
plt.yticks(y, fontsize=15)

# Adding a legend
plt.legend(fontsize=13)
fig.set_size_inches(8, 8)
fig.set_dpi(100)
plt.savefig('Figure_5.png',  bbox_inches='tight')
# Displaying the chart
plt.show()
