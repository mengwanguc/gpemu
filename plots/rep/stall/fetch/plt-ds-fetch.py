import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

plt.rcParams["font.family"] = "Helvetica"

fontsize=39

fig = plt.gcf()

columns = ['model', 'emu', 'paper']

# ssd, 24 workers, 8 v100
df = pd.read_csv('ds-fetch-emu.csv', sep='\t', header=None, names=columns)

num_subcategories = 2
emulated_values = df['emu'].tolist()
paper_values = df['paper'].tolist()
models = df['model'].tolist()
categories = models

bar_width = 0.2
emulated_values = [num * 100 for num in emulated_values]
paper_values = [num * 100 for num in paper_values]
index = np.arange(len(categories))

# Plotting the bar chart
plt.bar(index, emulated_values, bar_width, label='By GPEmu', color='blue')
plt.bar(index + 1 * bar_width, paper_values, bar_width, label='From DS Paper', color='green')


plt.ylim(0,100)
# Adding labels and title
plt.xticks(index + (bar_width * num_subcategories) / 2, categories, fontsize=fontsize)
plt.yticks(fontsize=fontsize)
plt.xlabel('Models', fontsize=fontsize)
plt.ylabel('Fetch Stall (% of epoch time)', fontsize=fontsize)
plt.title('(a) Fetch stalls across varying models',fontsize=fontsize, pad=20)

fig.set_size_inches(20, 8)
fig.set_dpi(100)
plt.legend(fontsize=fontsize,  markerfirst=False)
plt.savefig('ds-fetch.eps',  bbox_inches='tight')
# Displaying the chart
plt.show()
