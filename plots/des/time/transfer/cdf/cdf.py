import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import pandas as pd

title_font_size = 52
tick_font_size = 48
label_size = 40

matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
plt.rcParams["font.family"] = "Times New Roman"
annotation_font = {'fontname':'Times New Roman', 'fontsize':title_font_size}

linewidth=5


legends = ['AlexNet-V100-B64', 'ResNet18-P100-B64']
# filenames = ['alexnet-batch64-v100.csv', 'resnet18-batch64-p100.csv']
filenames = ['alexnet-batch64-p100.csv']
colors = ['blue', 'darkgreen']
all_data = []
for filename in filenames:
	df = pd.read_csv(filename, delimiter='\t')
	dl = df['cpu2gpu_time'].to_list()
	# drop the first 5 values
	dl = dl[5:-1]
	dl.insert(0,0)
	all_data.append(dl)
# for d in all_data:
# 	print(d)
# exit(0)


figure, axes = plt.subplots()
x_range = [0,0.02]
y_range = [0,1]
axes.set_xlim(x_range)
axes.set_ylim(y_range)
plt.xticks([0,0.01,0.02], ['0',0.01,0.02],fontsize=tick_font_size)
plt.yticks([0,0.2,0.4,0.6,0.8,1],[0,0.2,0.4,0.6,0.8,1], fontsize=tick_font_size)

for i in range(len(all_data)):
	data = all_data[i]
	data_sorted = np.sort(data)
	cdf = np.arange(0, len(data_sorted)) / float(len(data_sorted)-1)

	plt.plot(data_sorted, cdf, linewidth=linewidth, label=legends[i], color=colors[i])


# axes.set_aspect( 1.1 )

plt.xlabel('Per-batch Host-to-GPU Time', fontsize=title_font_size)
plt.ylabel('CDF', fontsize=title_font_size)
plt.title('CDF of Per-batch Host-to-GPU Transfer Time', fontsize=title_font_size, y=1.05)

# plt.legend(fontsize=label_size, markerfirst=False, borderpad=0.2, bbox_to_anchor=(0.55, 0.7))
figure.set_size_inches(12, 8)
figure.set_dpi(100)
plt.savefig('cdf.eps', bbox_inches='tight')