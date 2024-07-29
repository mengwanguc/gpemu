import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

fontsize=39
plt.rcParams["font.family"] = "Arial"
plt.rcParams["font.size"] = fontsize

fig = plt.gcf()
fig.set_dpi(100)


fig.set_size_inches(15, 6)

plt.subplot(1,2,1)

columns = ['nodes', 'baseline', 'gpemu', 'paper']

# ssd, 24 workers, 8 v100
df = pd.read_csv('65cache/data.csv', sep='\t', header=None, names=columns)

num_subcategories = 2
baseline_values = df['baseline'].tolist()
gpemu_values = df['gpemu'].tolist()
paper_values = df['paper'].tolist()

nodes = df['nodes'].tolist()
categories = nodes

bar_width = 0.2
index = np.arange(len(categories))

# Plotting the bar chart
plt.bar(index, baseline_values, bar_width, label='Baseline', color='red')
plt.bar(index + 1 * bar_width, gpemu_values, bar_width, label='CoorDL-GPEmu', color='green')
plt.bar(index + 2 * bar_width, paper_values, bar_width, label='CoorDL-Paper', color='blue')

plt.legend(loc="upper center", bbox_to_anchor=(1.15, 1.5), ncol=3, markerfirst=False, columnspacing=1)

plt.ylim(0,12)
# Adding labels and title
plt.xticks(index + (bar_width * num_subcategories) / 2, categories, fontsize=fontsize)
plt.yticks(range(0,12+1, 4), range(0,12+1, 4), fontsize=fontsize)
plt.xlabel('# of nodes', fontsize=fontsize)
plt.ylabel('Speedup', fontsize=fontsize, labelpad=20)
plt.title('(a) With 65% Local cache',fontsize=fontsize, pad=25)






plt.subplot(1,2,2)

columns = ['nodes', 'baseline', 'gpemu', 'paper']

# ssd, 24 workers, 8 v100
df = pd.read_csv('40cache/data.csv', sep='\t', header=None, names=columns)

num_subcategories = 3
baseline_values = df['baseline'].tolist()
gpemu_values = df['gpemu'].tolist()
paper_values = df['paper'].tolist()

nodes = df['nodes'].tolist()
categories = nodes

bar_width = 0.2
index = np.arange(len(categories))

# Plotting the bar chart
plt.bar(index, baseline_values, bar_width, label='Baseline', color='red')
plt.bar(index + 1 * bar_width, gpemu_values, bar_width, label='CoorDL-GPEmu', color='green')
plt.bar(index + 2 * bar_width, paper_values, bar_width, label='CoorDL-Paper', color='blue')

# plt.legend(loc="upper center", bbox_to_anchor=(1.15, 1.5), ncol=3, markerfirst=False, columnspacing=1)

plt.ylim(0,12)
# Adding labels and title
plt.xticks(index + (bar_width * num_subcategories) / 2, categories, fontsize=fontsize)
plt.yticks(range(0,12+1, 4), range(0,12+1, 4), fontsize=fontsize)
plt.xlabel('# of nodes', fontsize=fontsize)
# plt.ylabel('Speedup', fontsize=fontsize, labelpad=20)
plt.title('(b) With 40% Local cache',fontsize=fontsize, pad=25)







plt.savefig('speedup.eps',  bbox_inches='tight')
# Displaying the chart
plt.show()
