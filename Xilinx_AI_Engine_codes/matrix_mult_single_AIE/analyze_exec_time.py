import glob, os
from statistics import mean

# find from trace, and report in ns
initial_simulation_ns = 571

global_initial_tlast_list = []
global_average_diff_list = []

for filename in glob.glob("matC*.txt"):

    # print(filename)

    with open(filename) as file:

        # list that keeps the time of tlast on each file separetely
        file_tlast_list = []

        prevline = ""
        for line in file:
            if "TLAST" in line:
                time_list = prevline.split()

                if (time_list[2] == "ns"):
                    file_tlast_list.append(float(time_list[1]))
                elif (time_list[2] == "ps"):
                    file_tlast_list.append(float(time_list[1])/1000)
                elif (time_list[2] == "us"):
                    file_tlast_list.append(float(time_list[1])*1000)
                else:
                    print(time_list[2])
                    print("not ns or ps... exiting")
                    exit(1)

            prevline = line

        # print(file_tlast_list)
        global_initial_tlast_list.append(file_tlast_list[0])

        prev_time = file_tlast_list[0]
        file_diff_list = []
        for i in range(len(file_tlast_list)):
            
            if (i > 0):
                file_diff_list.append(file_tlast_list[i]-prev_time)

            prev_time = file_tlast_list[i]

        # print(round(mean(file_diff_list),2))
        global_average_diff_list.append(round(mean(file_diff_list),2))


print("Mean initial tlast:", (mean(global_initial_tlast_list) - initial_simulation_ns), "ns")
print("Steady state total time:" , mean(global_average_diff_list), "ns")