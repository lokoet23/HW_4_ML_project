# importing the module
import json
 
# Opening JSON file
with open('active_cores.json') as json_file:
    data = json.load(json_file)

    print("Active Cores:", len(data["ActiveCores"]))
    print("Active Memory:", len(data["ActiveMemory"]))
 