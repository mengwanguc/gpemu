import matplotlib.pyplot as plt
import pandas as pd

textfontsize=40

plt.rcParams["font.family"] = "Helvetica"
plt.rcParams["font.size"] = textfontsize

fig = plt.gcf()
columns = ['alexnet',"renet18",'mobilenet_v2','resnet50']
df = pd.read_csv('ds-cpu-emu.csv', sep='\t', header=None, names=columns)

alexnet = df['alexnet'].tolist()
renet18 = df['renet18'].tolist()
mobilenet_v2 = df['mobilenet_v2'].tolist()
resnet50 = df['resnet50'].tolist()
# Sample data for three lines
x = [1, 2, 3, 6, 12, 24]
y = [1000,2000,3000,4000]
y1 = alexnet
y2 = renet18
y3 = mobilenet_v2
y4 = resnet50

# Plotting the lines
plt.plot(x, y1, marker='o', color = "blue", label='AlexNet')
plt.plot(x, y2, marker='s', color = "green", label='ResNet18')
plt.plot(x, y3, marker='^', color = "orange", label='MobileNet')
plt.plot(x, y4, marker='*', color = "red", label='ResNet50')

# Adding labels and title
plt.xlabel('Numer of CPU cores per GPU', fontsize=textfontsize)
plt.ylabel('# of images trained per sec', fontsize=textfontsize)
plt.title('(c) Impact of CPU cores on training', fontsize=textfontsize, pad=20)
plt.xticks(x, fontsize=textfontsize)
plt.yticks(y, fontsize=textfontsize)

# Adding a legend
plt.legend(fontsize=textfontsize, loc='best', markerfirst=False, ncol=2, handletextpad=0.3, columnspacing=0.5)
fig.set_size_inches(11.5, 8)
fig.set_dpi(100)
plt.savefig('ds-cpu.eps',  bbox_inches='tight')
# Displaying the chart
plt.show()
