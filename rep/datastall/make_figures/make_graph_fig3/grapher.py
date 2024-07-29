import matplotlib.pyplot as plt
import pandas as pd

fig = plt.gcf()

columns = ['percentage']
df = pd.read_csv('time3.dat', sep='\t', header=None, names=columns)
percentage = df['percentage'].tolist()
# Sample data
cols = ['models','cache']
df = pd.read_csv('cache.dat', sep='\t', header=None, names=cols)
model = df['models'].tolist()
cache = df['cache'].tolist()
categories = []
for i in range(len(model)):
    categories.append(''+model[i]+'\n'+str(cache[i]))

values = [num * 100 for num in percentage]

# Plotting the bar chart
plt.bar(categories, values)
plt.ylim(0,100)
# Adding labels and title
plt.xticks(fontsize=12)
plt.yticks(fontsize=13)
plt.xlabel('Models', fontsize=17)
plt.ylabel('Fetch Stall (% of epoch time)', fontsize=17)
plt.title('Fetch stalls (Emulator): Stall wait for IO on SSD + 8 workers\n~33% dataset cached + 8GPUs (emulated by p100).',fontsize=17)

fig.set_size_inches(13, 6)
fig.set_dpi(100)
plt.savefig('Figure_3_SSD_p100_8w_more_models.png',  bbox_inches='tight')
# Displaying the chart
plt.show()
