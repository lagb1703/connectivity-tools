import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import mne
import scipy.io


class SNEIATools():
    """ Esta clase se encarga de agrupar los métodos necesarios para el tratamiento de EEG """

    def read_data(self, path: str):
        """ Retorna los indices y los electrodos de un EEG (data)

        Args:
            path (string): Ruta de un archivo .csv que contiene los datos de un EEG

        Returns:
            tuple (np.array, np.array):
                (indices_muestra, electrodos)
                electrodos es un arreglo de arreglos, cada uno representa los datos de un electrodo
        """
        data_tuples = np.genfromtxt(path, delimiter=",", dtype=float, names=True)
        data = np.array([list(row) for row in data_tuples])
        print(data.shape)
        ch_names = list(data_tuples.dtype.names)
        ch_names.remove("0")
        print(ch_names, len(ch_names))

        indice_muestra = data[:, 0]

        electrodos = np.empty((data.shape[0], data.shape[1] - 1))
        for i in range(data.shape[1] - 1):
            electrodos[:,i] = data[:, i + 1]

        return ch_names, indice_muestra, electrodos

    def get_gfp(self, electrodes: np.ndarray):
        """ Retorna la GFP de un grupo de electrodos

        Args:
            electrodes (np.ndarray):
                Contiene los valores de un EEG, separado por electrodos

        Returns:
            GFP: Potencia de campo global de un grupo de electrodos
        """
        N = electrodes.shape[1]
        v_mean = np.mean(electrodes, axis=1)
        GFP = np.zeros(electrodes.shape[0])

        for i in range(0,N):
            GFP = GFP+(electrodes[:,i]-v_mean)**2
        GFP = np.sqrt(GFP/N)

        return GFP

    def get_electrodes_value(self, indices: np.array, electrodes: np.array): # type: ignore
        """ Retorna un array con los valores de cada electrodo para los indices dados.

        Args:
            indices (np.array): Indices separados por microestado
            electrodes (np.array): Array con los valores de los electrodos

        Returns:
            np.array:
                Array con los valores de los electrodos para los indices dados
        """
        electrodes_values = []

        for indice in indices:
            electrodes_values.append(electrodes[indice])

        return np.array(electrodes_values, dtype=object)

    def index_max_min(self, gfp):
        """ Retorna los indices de los maximos y minimos de la GFP
        
        Args:
            gfp (np.array): Potencia de campo global GFP

        """
        np.append(gfp, 0)
        derivative = np.diff(gfp)

        index_min_derivative = []
        index_max_derivative = []

        for i in range(len(derivative) - 1):
            if derivative[i] < 0 and derivative[i + 1] > 0:
                index_min_derivative.append(i)
            if derivative[i] > 0 and derivative[i + 1] < 0:
                index_max_derivative.append(i)

        index_max_gfp = []
        index_min_gfp = []

        for i in index_max_derivative:
            index_max_gfp.append(i+1)
        for i in index_min_derivative:
            index_min_gfp.append(i+1)

        return index_max_gfp, index_min_gfp

    def calcular_gev_cluster(self, puntos_asignados, centroide):
        GFP = self.get_gfp(puntos_asignados)
        num = 0
        den = 0
        for j in range(0, puntos_asignados.shape[0]):
            num += (GFP[j] * np.corrcoef(puntos_asignados[j, :], centroide)[0, 1]) ** 2
            den += (GFP[j]) ** 2
        return num / den

    def k_means_modificado(self, datos, k=4, iteraciones=10):
        # 1. Inicializar los centroides aleatoriamente.
        centroides = np.random.randint(-15, 15 + 1, size=(4, 64))

        GEV = []

        for i in range(iteraciones):
            # 2. Calcular la correlación entre cada centroide y el resto de los puntos.
            asignacion_centroides = []

            for punto in datos:
                correlaciones = [np.corrcoef(punto, centroide)[0, 1] for centroide in centroides]
                asignacion_centroides.append(np.argmax(correlaciones))

                # 3. Promediar los puntos asignados a cada centroide para crear nuevos centroides.
                GEVi = 0
                for j in range(k):
                    puntos_asignados = datos[np.array(asignacion_centroides) == i]
                    if len(puntos_asignados) > 0:  # Evitar dividir por cero.
                        centroides[j] = np.mean(puntos_asignados, axis=0)

                # 4. Con esos puntos asignados calculo la GEV de cada cluster
                GEVi += self.calcular_gev_cluster(puntos_asignados, centroides[i])
                GEV.append(GEVi)

        # 5. Añadir columna adicional con la asignación de los centroides.
        matriz_final = pd.DataFrame(
            np.hstack((datos, np.array(asignacion_centroides).reshape(-1, 1))),
            columns=[f'col_{i}' for i in range(64)] + ['centroide']
        )

        return matriz_final, centroides

    def get_topomap(self, serie, instant, channels, freq: int, standard: str):
        """ Imprime mapa topográfico a partir de una serie de tiempo en un instante dado

        Args:
            serie (np.array): 
                Serie de tiempo

            instant (int): 
                Instante de tiempo

            channels (list): 
                Lista de nombres de los canales

            freq (int): 
                Frecuencia de muestreo

            standard (str): 
                Tipo de montaje
        """
        ch_types_str = ['eeg']*len(channels)

        info = mne.create_info(ch_names=channels, sfreq=freq, ch_types=ch_types_str)
        raw_data = mne.io.RawArray(serie.T, info)
        raw_data.set_montage(standard)

        _, ax = plt.subplots(figsize=(5, 5))
        mne.viz.plot_topomap(instant, raw_data.info, axes=ax)
        plt.show()

    def preprocess_data(self, path: str):
        """ Preprocesamiento de los datos de un EEG en formato .set
            Crea un archivo en formato CSV con los datos preprocesados y retorna la ruta
            Para la base de datos: https://openneuro.org/datasets/ds003775/versions/1.2.1

        Args:
            path (string):
                Ruta del archivo .set a preprocesar

        Returns:
            string:
                Ruta del archivo .csv preprocesado
        """
        data = scipy.io.loadmat(path)
        chanlocs = data['chanlocs']
        ch_name = [chanlocs[0][i][0] for i in range(0,64)]
        ch_names = [item[0] for item in ch_name]

        signals = data['data']

        first_layer_signals = signals[:,:,0]

        transpose_signals = np.transpose(first_layer_signals)

        transpose_signals = np.vstack((ch_names, transpose_signals))

        index_sequence = np.arange(transpose_signals.shape[0]).reshape(-1, 1)

        dataset_csv = np.hstack((index_sequence, transpose_signals))

        csv_path = data["filename"][0].replace(".set", "_preprocess.csv")

        np.savetxt(csv_path, dataset_csv, delimiter=',', fmt='%s')

        return csv_path

    def get_occurrence(self, vector):
        """ Retorna la ocurrencia de cada letra en un vector

        Args:
            vector (np.array): Vector con letras correspondientes a los microestados

        Returns:
            ocurrence (dict): Diccionario con la ocurrencia de cada letra
        """
        occurrence = {}

        actual_letter = None
        for letter in vector:
            if letter != actual_letter:
                if letter in occurrence:
                    occurrence[letter] += 1
                else:
                    occurrence[letter] = 1
                actual_letter = letter

        return occurrence

    def get_coverage(self, vector):
        """ Retorna la cobertura de cada letra en un vector

        Args:
            vector (np.array): Vector con letras

        Returns:
            dict:
                Diccionario con la cobertura de cada letra
        """
        letters = ["A", "B", "C", "D"]

        coverage = {}
        for letter in letters:
            coverage[letter] = vector.count(letter) / len(vector)

        return coverage

    def get_duration(self, vector, freq):
        """ Retorna la duración promedio de cada letra en un vector

        Args:
            vector (np.array): Vector con letras correspondientes a los microestados
            freq (int): Frecuencia de muestreo

        Returns:
            durations (dict): Diccionario con la duración promedio de cada letra
        """
        letters = ["A", "B", "C", "D"]
        grouped_letters = []
        grouped_letters.append(vector[0])

        # Iteramos desde el segundo elemento del original hasta el último
        for i in range(1, len(vector)):
            # Si la letra actual es igual a la letra anterior en el nuevo vector,
            # las concatenamos en el nuevo vector
            if vector[i] == vector[i - 1]:
                grouped_letters[-1] += vector[i]
            else:
                grouped_letters.append(vector[i])

        durations = {}
        letter_durations = [[], [], [], []]

        for i in range(len(grouped_letters)):
            if grouped_letters[i][0]=='A':
                duracion_a=len(grouped_letters[i])*(1/freq)
                letter_durations[0].append(duracion_a)
            elif grouped_letters[i][0]=='B':
                duracion_b=len(grouped_letters[i])*(1/freq)
                letter_durations[1].append(duracion_b)
            elif grouped_letters[i][0]=='C':
                duracion_c=len(grouped_letters[i])*(1/freq)
                letter_durations[2].append(duracion_c)
            else:
                duracion_d=len(grouped_letters[i])*(1/freq)
                letter_durations[3].append(duracion_d)

        for letter in letters:
            durations[letter] = np.mean(letter_durations[letters.index(letter)])

        return durations

    def get_metrics(self, vector, freq):
        """ Retorna la cobertura, ocurrencia y duración de cada microestado
            Args:
                vector (np.array): Vector con letras correspondientes a los microestados
                freq (int): Frecuencia de muestreo
        """
        coverage = self.get_coverage(vector)
        occurrence = self.get_ocurrence(vector)
        durations = self.get_duration(vector, freq)

        return coverage, occurrence, durations
