import matplotlib.pyplot as plt

categories = ["ImageFolder Data Reading Only", "ImageFolder Reading + Processing", "ImageFolder Full Training", "Idealized Training","FFCV"]
values = [35, 310, 327, 21 ,83]  
# values = [34, 311, 330, 24 ,82] 

plt.figure(figsize=(8, 4))  # Set the figure size (optional)

plt.barh(categories, values)  # Create the horizontal bar chart

plt.xlabel("Time per epoch (sec)")
# plt.ylabel("Categories")
plt.title("Time taken per set of stages in ImageNet training (on CPU with emulator)")

# Save the chart as a JPEG image
plt.savefig("figure-2-cpu.jpg", format="jpg", dpi=300, bbox_inches="tight")

plt.show()  # Display the chart (optional)

# ~~~~~~~~~~~~~~~~~~~~

# categories = ["ImageFolder Data Reading Only", "ImageFolder Reading + Processing", "ImageFolder Full Training", "Idealized Training","FFCV"]
# # values = [35, 310, 327, 21 ,109] cpu 
# values = [34, 311, 330, 24 ,82] 

# plt.figure(figsize=(8, 4))  # Set the figure size (optional)

# plt.barh(categories, values)  # Create the horizontal bar chart

# plt.xlabel("Time per epoch (sec)")
# # plt.ylabel("Categories")
# plt.title("Time taken per set of stages in ImageNet training (on GPU)")

# # Save the chart as a JPEG image
# plt.savefig("figure-2-gpu.jpg", format="jpg", dpi=300, bbox_inches="tight")

# plt.show()  # Display the chart (optional)



