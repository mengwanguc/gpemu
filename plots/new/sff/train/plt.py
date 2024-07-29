import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

plt.rcParams["font.family"] = "Arial"

fontsize=39

fig = plt.gcf()

columns = ['cache', 'pytorch', 'minio', 'sff']

# ssd, 24 workers, 8 v100
df = pd.read_csv('data.csv', sep='\t', header=None, names=columns)

num_subcategories = 3
pytorch_values = df['pytorch'].tolist()
minio_values = df['minio'].tolist()
sff_values = df['sff'].tolist()
caches = df['cache'].tolist()
categories = caches

bar_width = 0.2
index = np.arange(len(categories))

# Plotting the bar chart
plt.bar(index, pytorch_values, bar_width, label='OS Page Cache', color='red')
plt.bar(index + 1 * bar_width, minio_values, bar_width, label='MinIO', color='green')
plt.bar(index + 2 * bar_width, sff_values, bar_width, label='SFF+MinIO', color='blue')


plt.ylim(0,600)
# Adding labels and title
plt.xticks(index + (bar_width * num_subcategories) / 2, categories, fontsize=fontsize)
plt.yticks(fontsize=fontsize)
plt.xlabel('% of dataset cached', fontsize=fontsize)
plt.ylabel('Epoch Time', fontsize=fontsize, labelpad=20)
plt.title('(c) Epoch time with different cache replacement algorithms',fontsize=fontsize, pad=25)

fig.set_size_inches(20, 8)
fig.set_dpi(100)
plt.legend(fontsize=fontsize,  markerfirst=False, frameon=False, bbox_to_anchor=(0.6, 0.55))
plt.savefig('epoch.eps',  bbox_inches='tight')
# Displaying the chart
plt.show()
