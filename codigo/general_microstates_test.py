from sneia_tools import SNEIATools

path = "/home/netropy/Documentos/Conectivity_tools/codigo/csvs"
path_general_microstates = "/home/netropy/Documentos/Conectivity_tools/codigo/Microstates"

tools = SNEIATools(freq=256)
tools.get_centroids(folder_path=path)
tools.get_general_microstates(folder_path=path_general_microstates)

print(tools.general_microstates)
#tools.plot_general_microstates()
