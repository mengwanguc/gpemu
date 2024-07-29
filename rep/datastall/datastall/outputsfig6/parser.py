import re
models = ["alexnet", "resnet18", "mobilenet_v2", "resnet50"]

list_files = []
for model in models:
    st = "output_"+model+".txt"
    list_files.append(st)

values = []
for file_name in list_files:
    with open(file_name, "r") as input_file:
        lines = input_file.readlines()
        last_two_lines = lines[-1:]
        for line in last_two_lines:
            values.append(line.strip())

# Open a new file for writing
with open("parsed_output.txt", "a") as output_file:
    for value in values:
        output_file.write(str(value) + "\n")
