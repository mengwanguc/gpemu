import re
sizes = ['6g','8g','10g','12g','14g','16g']
"10g" "12g" "14g" "16g" "18g" "20g"
list_files = []
for size in sizes:
    st = "output"+size+".txt"
    list_files.append(st)

values = []
for file_name in list_files:
    with open(file_name, "r") as input_file:
        lines = input_file.readlines()
        line690 = lines[689]  # Index 689 corresponds to line 690
        line1374 = lines[1373]  # Index 1373 corresponds to line 1374
        values.append(line690.strip())
        values.append(line1374.strip())

# Open a new file for writing
with open("parsed_output.txt", "a") as output_file:
    for value in values:
        output_file.write(str(value) + "\n")
