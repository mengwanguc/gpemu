# import matplotlib.pyplot as plt

# # Data
# categories = ['FFCV' ,"Nvidia Dali",'ImageFolder']
# values = [69,149,337]

# # Create a bar chart
# plt.bar(categories, values)

# # Adding labels and a title
# plt.xlabel('Loader')
# plt.ylabel('Training Time (sec/epoch)')
# plt.suptitle('Comparison training time to train one epoch on ImageNet(GPU)')
# plt.title("(Model:Alexnet, Batch Size:256, Optimizer:SGD)", fontsize=10)

# # Save the chart as a JPG file
# plt.savefig('figure5(a)-gpu-new.jpg', format='jpg')

# Show the plot (optional)
# plt.show()


import matplotlib.pyplot as plt

# Data
categories = ['FFCV', 'Nvidia Dali','ImageFolder']
values = [67,272, 324]

# Create a bar chart
plt.bar(categories, values)

# Adding labels and a title
plt.xlabel('Loader')
plt.ylabel('Training Time (sec/epoch)')
plt.suptitle('Comparison training time to train one epoch on ImageNet(CPU)')
plt.title("(Model:Alexnet, Batch Size:256, Optimizer:SGD)", fontsize=10)

# Save the chart as a JPG file
plt.savefig('figure5(a)-cpu-new.jpg', format='jpg')

# Show the plot (optional)
# plt.show()
