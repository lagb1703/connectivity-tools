import scipy.io
import numpy as np
import numpy as np
import matplotlib.pyplot as plt
from sneia_tools import SNEIATools

# Obtener los datos de los electrodos
tools = SNEIATools()
preprocess_path = tools.preprocess_data('codigo/MD5E-s111284160--f39bd1e24dbf93dd25b4df19d989baf7.set')
ch_names, indice_muestra, electrodos = tools.read_data(preprocess_path)

print("Path del csv", preprocess_path)
print("Nombres de los canales: ", ch_names)

# Calcular la gfp para los electrodos
gfp = tools.get_gfp(electrodos)

# Obtener los microestados (indices y valores en la gfp que superan el umbral)
indexes, values = tools.get_microstates(gfp)
print("microstates_indexes: (", len(indexes), ")\n", indexes)
print("microstates_values: (", len(values), ")\n", values)
electrodes_values = tools.get_electrodes_value(indexes, electrodos)
print("valores de los electrodos: (", len(electrodes_values), ")\n", electrodes_values)
print()

# Preparación de gráfico de gfp
plt.plot(indice_muestra, gfp, label="gfp")
plt.axhline(np.mean(gfp)+2*np.std(gfp), color='red', linestyle='--')

# Agregar etiquetas y leyenda
plt.xlabel('Índice de muestra')
plt.ylabel('Amplitud')
plt.legend()

# Mostrar el gráfico de la gfp
plt.show()

# Clusterizamos
df = tools.apply_kmeans(electrodes_values[0])

# Obtener series de tiempo
time_series = tools.get_time_series(df)

# Presentar mapas topográficos
print("Nombres de los canales (", len(ch_names), ")", ch_names)
tools.time_series_topomap(time_series[0],ch_names,256,"standard_1020")
tools.time_series_topomap(time_series[1],ch_names,256,"standard_1020")
tools.time_series_topomap(time_series[2],ch_names,256,"standard_1020")
tools.time_series_topomap(time_series[3],ch_names,256,"standard_1020")
