import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

fontsize=25

fig = plt.gcf()

columns = ['loader', 'emulator', 'gpu', 'paper']

# ssd, 24 workers, 8 v100
# df = pd.read_csv('ffcv.csv.dali', sep='\t', names=columns)
df = pd.read_csv('ffcv.csv', sep='\t', names=columns)

def norm(values):
	base = float(values[-1])
	values = [float(x)/base for x in values]
	return values

emulated_values = df['emulator'].tolist()
gpu_values = df['gpu'].tolist()
paper_values = df['paper'].tolist()
categories = df['loader'].tolist()

emulated_values = norm(emulated_values)
gpu_values = norm(gpu_values)
paper_values = norm(paper_values)

num_subcategories = 3
bar_width = 0.2
index = np.arange(len(categories))

# Plotting the bar chart
print(emulated_values)
plt.bar(index, emulated_values, bar_width, label='w/ GPEmu', color='green')
plt.bar(index + 1* bar_width, gpu_values, bar_width, label='w/ Real GPU', color='red')
plt.bar(index + 2 * bar_width, paper_values, bar_width, label='From FFCV Paper', color='blue')


plt.ylim(0,1.0)
# Adding labels and title
plt.xticks(index + (bar_width * num_subcategories) / 2, categories, fontsize=fontsize)
plt.yticks(fontsize=fontsize)
plt.xlabel('Data Loader', fontsize=fontsize)
plt.ylabel('Training Time (normalized)', fontsize=fontsize)
plt.title('Training Time per Epoch Using Different Data Loaders',fontsize=fontsize, pad=20)

fig.set_size_inches(10, 6)
fig.set_dpi(100)
plt.legend(fontsize=fontsize)
plt.savefig('ffcv.eps',  bbox_inches='tight')
# Displaying the chart
plt.show()