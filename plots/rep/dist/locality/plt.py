import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

fontsize=39
plt.rcParams["font.family"] = "Arial"
plt.rcParams["font.size"] = fontsize

fig = plt.gcf()
fig.set_dpi(100)


fig.set_size_inches(15, 6)

columns = ['nodes', 'DistCache', 'LocalityAware']

# ssd, 24 workers, 8 v100
df = pd.read_csv('data.csv', sep='\t', header=None, names=columns)

num_subcategories = 2
node_values = df['nodes'].tolist()
dist_values = df['DistCache'].tolist()
locality_values = df['LocalityAware'].tolist()

nodes = df['nodes'].tolist()
categories = nodes

bar_width = 0.2
index = np.arange(len(categories))

# Plotting the bar chart
plt.bar(index, dist_values, bar_width, label='Regular', color='red')
plt.bar(index + 1 * bar_width, locality_values, bar_width, label='Locality-aware', color='green')

# plt.legend(loc="upper center", bbox_to_anchor=(1.15, 1.5), ncol=3, markerfirst=False, columnspacing=1)
plt.legend(loc="upper right", markerfirst=False, columnspacing=1)


plt.ylim(0,250)
# Adding labels and title
plt.xticks(index + (bar_width * num_subcategories) / 2, categories, fontsize=fontsize)
plt.yticks(range(0,251,50), range(0,251,50), fontsize=fontsize)
plt.xlabel('# of nodes', fontsize=fontsize)
plt.ylabel('Epoch Time', fontsize=fontsize, labelpad=20)
plt.title('Distributed Cache w/wo Locality-aware',fontsize=fontsize, pad=25)






plt.savefig('speedup.eps',  bbox_inches='tight')
# Displaying the chart
plt.show()
