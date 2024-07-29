import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

fontsize=40
plt.rcParams["font.family"] = "Arial"
plt.rcParams["font.size"] = fontsize

fig = plt.gcf()

columns = ['scheduler', 'emulator', 'paper']

# ssd, 24 workers, 8 v100
# df = pd.read_csv('ffcv.csv.dali', sep='\t', names=columns)
df = pd.read_csv('data.csv', sep='\t', names=columns)

emulated_values = df['emulator'].tolist()
paper_values = df['paper'].tolist()
categories = df['scheduler'].tolist()

num_subcategories = 2
bar_width = 0.3
index = np.arange(len(categories))

# Plotting the bar chart
print(emulated_values)
plt.bar(index, emulated_values, bar_width, label='w/ GPEmu', color='green')
plt.bar(index + 1 * bar_width, paper_values, bar_width, label='Paper', color='blue')


yticks=[0,1,2]
plt.ylim(0,3)
plt.xlim(-0.3,1.5)
# Adding labels and title
plt.xticks(index + (bar_width) / 2, categories, fontsize=fontsize)
plt.yticks(yticks, yticks, fontsize=fontsize)
plt.xlabel('Scheduler', fontsize=fontsize)
plt.ylabel('Avg JCT (normalized)', fontsize=fontsize)
title = plt.title('(b) Normalized average JCT',fontsize=fontsize, pad=20)
# title.set_position((0.4, 0))

fig.set_size_inches(8, 6)
fig.set_dpi(100)
plt.legend(fontsize=fontsize, frameon=False, markerfirst=False)
plt.savefig('fig.eps',  bbox_inches='tight')
# Displaying the chart
plt.savefig('fig.png',  bbox_inches='tight')