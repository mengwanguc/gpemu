import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

fontsize=35
plt.rcParams["font.family"] = "Helvetica"
plt.rcParams["font.size"] = fontsize

fig = plt.gcf()

columns = ['ratio', 'TF-NO', 'TF-DS', 'FastFlow']

# ssd, 24 workers, 8 v100
df = pd.read_csv('ff-transformer.dat', sep='\t', names=columns)

tf_values = df['TF-NO'].tolist()
ds_values = df['TF-DS'].tolist()
ff_values = df['FastFlow'].tolist()
ratios = df['ratio'].tolist()

num_subcategories = 3
bar_width = 0.2
index = np.arange(len(ratios))

# Plotting the bar chart
plt.bar(index, tf_values, bar_width, label='TF-NO', color='blue')
plt.bar(index + 1* bar_width, ds_values, bar_width, label='TF-DS', color='green')
plt.bar(index + 2 * bar_width, ff_values, bar_width, color='orange')


plt.ylim(0,80)
# Adding labels and title
plt.xticks(index + (bar_width * num_subcategories) / 2, ratios, fontsize=fontsize)
plt.yticks(range(0,100,20), fontsize=fontsize)
plt.xlabel('Local CPU # : Remote CPU #', fontsize=fontsize)
plt.ylabel('Epoch Duration (s)', fontsize=fontsize)
plt.title('(a) Transformer ASR',fontsize=fontsize, pad=20)

fig.set_size_inches(8, 6)
fig.set_dpi(100)
plt.legend(fontsize=fontsize, ncol=3, columnspacing=0.6, markerfirst=False, frameon=False, handletextpad=0.5)
plt.savefig('ff-transformer.eps',  bbox_inches='tight')
# Displaying the chart
plt.show()