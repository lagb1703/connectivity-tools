from sneia_tools import SNEIATools
import os
path = os.path.join(".", "..","codigo", "csvs")
path_general_microstates = os.path.join(".", "..","codigo", "csvs")
print(path_general_microstates)

ch_names = ["Fp1","AF7","AF3","F1","F3","F5","F7","FT7","FC5","FC3","FC1","C1","C3","C5","T7","TP7","CP5","CP3","CP1","P1","P3","P5","P7","P9","PO7","PO3","O1","Iz","Oz","POz","Pz","CPz","Fpz","Fp2","AF8","AF4","AFz","Fz","F2","F4","F6","F8","FT8","FC6","FC4","FC2","FCz","Cz","C2","C4","C6","T8","TP8","CP6","CP4","CP2","P2","P4","P6","P8","P10","PO8","PO4","O2"]

tools = SNEIATools(freq=256, ch_names=ch_names)
tools.get_centroids(folder_path=path)
#tools.get_general_microstates(folder_path=path_general_microstates)

#print("GENERAL MICROSTATES: ", tools.general_microstates.shape, tools.general_microstates)
#tools.plot_microstates(path_general_microstates)


## TODO: Nos quedamos en que los microestados que estamos imprimiendo no son los correctos
