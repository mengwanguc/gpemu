import matplotlib.pyplot as plt
import pandas as pd

fig = plt.gcf()

columns = ['percentage']
df = pd.read_csv('time6.dat', sep='\t', header=None, names=columns)
percentage = df['percentage'].tolist()
# Sample data
categories = ['alexnet','resnet18', 'mobilenet_v2', 'resnet50']
values = percentage

# Plotting the bar chart
plt.bar(categories, values)

plt.ylim(0, 100)
# Adding labels and title
plt.xlabel('Models', fontsize=20)
plt.ylabel('Prep Stall (% of epoch time)', fontsize=20)
plt.title('Prep Stall across DNNs (Emulator)\n All 100% cached + HDD + 8 workers + 8GPUs (emulated p100)', fontsize=20)
plt.xticks(fontsize=18)
plt.yticks(fontsize=18)

fig.set_size_inches(13, 10)
fig.set_dpi(100)
plt.savefig('Figure_6.png',  bbox_inches='tight')
# Displaying the chart
plt.show()
