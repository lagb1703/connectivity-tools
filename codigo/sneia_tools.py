import csv
import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import mne
import scipy.io
from scipy.stats import pearsonr
from sklearn.cluster import KMeans


class SNEIATools():
    """ Esta clase se encarga de agrupar los métodos necesarios para el tratamiento de EEG """

    def __init__(self, freq, ch_names):
        self.general_microstates = None
        self.freq = freq
        self.ch_names = ch_names
        mne.set_log_level("WARNING")

    def preprocess_from_sets(self, folder_path):
        """ Preprocesamiento de los datos de un EEG en formato .set
            Crea un archivo en formato CSV con los datos preprocesados y retorna la ruta
            Para la base de datos: https://openneuro.org/datasets/ds003775/versions/1.2.1
            Args:
                folder_path (string):
                    Ruta de la carpeta con los archivos .set a preprocesar
        """
        file_names = os.listdir(folder_path)

        for name in file_names:
            effective_path = f"{folder_path}/{name}"
            self.preprocess_data(effective_path)

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
        if not path.endswith(".set"):
            raise Exception("El archivo debe ser .set")

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

        np.savetxt(f"{path.rsplit('/', 1)[0]}/{csv_path}", dataset_csv, delimiter=',', fmt='%s')

        return csv_path

    def get_Max_electrodes_value(self, GFP, electrodes):
        np.append(GFP, 0)
        derivative = np.diff(GFP)
        #mejorar

        index_max_derivative = []

        for i in range(len(derivative) - 1):
            if derivative[i] > 0 and derivative[i + 1] < 0:
                index_max_derivative.append(electrodes[i + 1])

        return np.array(index_max_derivative).astype(np.float64)

    def get_microstates(self, path: str):
        """ Obtiene los microestados de un EEG en formato .csv
            Además, crea un archivo .csv con los microestados y retorna los valores de los microestados

            Args:
                path (string):
                    Ruta del archivo .csv a procesar

            Returns:
                tuple (pd.DataFrame, np.array):
                    matriz_final (pd.DataFrame):
                        DataFrame con los valores de los microestados
                    centroids (np.array):
                        Array con los centroides de los microestados
        """
        if not path.endswith(".csv"):
            raise Exception("El archivo debe ser .csv")

        _, electrodos = self.read_data(path)

        gfp = self.get_gfp(electrodos)

        electrodes_values = self.get_Max_electrodes_value(gfp, electrodos)

        np.random.seed(1)

        _, centroids = self.modified_k_means(electrodes_values)

        try:
            with (
                open(f"{path.rsplit('/', 2)[0]}/Microstates/{path.rsplit('/', 1)[1].replace('.csv', '_microstates.csv')}",
                    'w', newline='') as microstates_file
            ):
                csv_writer = csv.writer(microstates_file)

                for centroid in centroids:
                    csv_writer.writerow(centroid)
        except Exception as e:
            print(f"Ocurrió un error: {str(e)}")

        return f"{path.rsplit('/', 1)[0]}/Microstates"

    def get_centroids(self, folder_path: str):
        """ Obtiene los microestados de un grupo de EEGs en formato .csv

        Args:
            folder_path (str):
                Ruta de la carpeta con los archivos .csv a procesar

        Returns:
            general_microstates (list):
                Array con los microestados generales
        """
        file_names = os.listdir(folder_path)


        for file_name in file_names:
            self.get_microstates(os.path.join(folder_path, file_name))

    def get_general_microstates(self, folder_path: str):
        """ Obtiene los microestados generales de un grupo de EEGs en formato .csv

        Args:
            folder_path (str):
                Ruta de la carpeta con los archivos .csv a procesar

        Returns:
            general_microstates (list):
                Array con los microestados generales
        """
        file_names = os.listdir(folder_path)

        data = []
        centroids = []
        for file_name in file_names:
            with open(f"{folder_path}/{file_name}", newline='') as microstates_file:
                reader = csv.reader(microstates_file)
                for row in reader:
                    centroids.append(row)
            data += centroids

        self.general_microstates = self.cluster_of_clusters(data)
        with open(
            f"{folder_path}/general_microstates.csv", 'w', newline=''
        ) as general_microstates_file:
            csv_writer = csv.writer(general_microstates_file)

            for centroid in self.general_microstates:
                csv_writer.writerow(centroid)

        return self.general_microstates

    def cluster_of_clusters(self, data):
        """ Agrupa los microestados en 4 clusters

        Args:
            data (list):
                Lista con los microestados a clusterizar

        Returns:
            label, centroids (tuple):
                Etiquetas de los clusters y centroides
        """
        cluster_matrix = np.array(data).astype(np.float64)

        print("CLUSTERS: ", cluster_matrix.shape, cluster_matrix)

        kmeans = KMeans(n_clusters=4, random_state=1)
        kmeans.fit(cluster_matrix)
        centroids = kmeans.cluster_centers_

        return centroids

    def read_data(self, path: str):
        """ Retorna los indices y los electrodos de un EEG (data)

        Args:
            path (string):
                Ruta de un archivo .csv que contiene los datos de un EEG

        Returns:
            tuple (np.array, np.array):
                (indices_muestra, electrodos)
                electrodos es un arreglo de arreglos, cada uno representa los datos de un electrodo
        """
        data_tuples = np.genfromtxt(path, delimiter=",", dtype=float, names=True)
        electrodos = np.array([list(row)[1:] for row in data_tuples])
        #print(data.shape)
        ch_names = list(data_tuples.dtype.names[1:])
        #print(ch_names, len(ch_names))

        return ch_names, electrodos

    def get_gfp(self, electrodes: np.ndarray):
        """ Retorna la GFP de un grupo de electrodos

        Args:
            electrodes (np.ndarray):
                Contiene los valores de un EEG, separado por electrodos

        Returns:
            GFP: Potencia de campo global de un grupo de electrodos
        """
        #print(electrodes.shape)
        N = electrodes.shape[1]
        v_mean = np.mean(electrodes, axis=1)
        GFP = np.zeros(electrodes.shape[0])

        for i in range(0,N):
            GFP += (electrodes[:,i])**2
        GFP = np.sqrt(GFP/N - v_mean**2)
        return GFP

    def get_electrodes_value(self, indices: np.array, electrodes: np.array): # type: ignore
        """ Retorna un array con los valores de cada electrodo para los indices dados.

        Args:
            indices (np.array):
                Indices separados por microestado
            electrodes (np.array):
                Array con los valores de los electrodos

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

        Returns:|
            tuple (list, list):
                (index_max_gfp, index_min_gfp)
                index_max_gfp: Indices de los maximos de la GFP
                index_min_gfp: Indices de los minimos de la GFP
        """
        np.append(gfp, 0)
        derivative = np.diff(gfp)
        #mejorar

        index_min_derivative = []
        index_max_derivative = []

        for i in range(len(derivative) - 1):
            if derivative[i] < 0 and derivative[i + 1] > 0:
                index_min_derivative.append(i+1)
            if derivative[i] > 0 and derivative[i + 1] < 0:
                index_max_derivative.append(i+1)

        return index_min_derivative, index_max_derivative

    def calculate_gev_cluster(self, assigned_points, centroid):
        """ Calculates the GEV of a cluster

        Args:
            assigned_points (np.array):
                Matiz con los puntos asignados al cluster
            centroid (np.array):
                Centroide del cluster

        Returns:
            float:
                GEV del cluster
        """
        #print("assigned_points", assigned_points, " - ", assigned_points.shape)
        GFP = self.get_gfp(assigned_points)
        num = 0
        den = 0
        for j in range(0, assigned_points.shape[0]):
            num += (GFP[j] * np.corrcoef(assigned_points[j, :], centroid)[0, 1]) ** 2
            den += (GFP[j]) ** 2
        if den == 0:
            return None
        return num / den

    def modified_k_means(self, data, k=4, iterations=1000):
        """
        Obtiene los centroides y la matriz de electrodos etiquetada a partir de un EEG

        Args:
            data (np.array):
                Valores de los electrodos correspondientes a un EEG
            k (int, optional):
                Número de Clusters. Por defecto 4.
            iterations (int, optional):
                Número de iteraciones. Por defecto 10.

        Returns:
            final_matrix, centroids (tuple):
                final_matrix (pd.DataFrame):
                    Matriz con los valores de los electrodos etiquetados
                centroids (np.array):
                    Centroides de los clusters
        """
        centroids = np.random.uniform(-15, 15 + 1, size=(k, 64))
        while iterations > 0:
            #print("Iteración ", iterations)
            iterations -= 1
            dataCentroid = []
            i = k
            while i > 0:
                i -= 1
                dataCentroid.append([])

            for point in data:
                max = 0
                i = 0
                for pos, centroid in enumerate(centroids):
                    value = abs(np.corrcoef(point, centroid)[0, 1])
                    if value > max:
                        max = value
                        i = pos
                dataCentroid[i].append(point)
                assigned_points = np.array(dataCentroid[i])
                centroids[i] = np.mean(assigned_points, axis=0, dtype=np.float64)
            # GEVi = 0
            # for l in range(k):
            #     if len(dataCentroid[l]) > 0:
            #         GEVi += self.calculate_gev_cluster(np.array(dataCentroid[l]), centroids[l])
            # print(GEVi)
        final_matrix = 0 #= pd.DataFrame(
        #     np.hstack((data, np.array(centroid_assignment).reshape(-1, 1))),
        #     columns=[f'col_{i}' for i in range(64)] + ['centroid']
        # )

        return final_matrix, centroids
    def plot_general_microstates(self):
        """ Grafica los microestados generales

        Returns:
            None
        """
        for ms in self.general_microstates:
            self.get_topomap(self.general_microstates, ms, self.ch_names, self.freq, 'standard_1020')

    def plot_microstates(self, folder_path):
        """ Grafica los microestados del path recibido

        Args:
            folder_path (str):
                Ruta de la carpeta con los microestados en formato .csv a imprimir

        Returns:
            None
        """
        for file_name in os.listdir(folder_path):
            microstates = np.array(pd.read_csv(os.path.join(folder_path, file_name), header=0))[:, 1:]
            for ms in microstates:
                self.get_topomap(microstates, ms, self.ch_names, self.freq, 'standard_1020')


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
        ch_types_str = ['eeg']*(len(channels))#¿para que?
        # ch_types_str.extend(['eeg']*(len(channels)-1))
        # names.extend(channels)
        info = mne.create_info(ch_names=channels, sfreq=freq, ch_types=ch_types_str)
        # print(len(serie.T[0:-1, 1:-1]), '\n', info)
        raw_data = mne.io.RawArray(serie.T, info)
        raw_data.set_montage(standard)

        _, ax = plt.subplots(figsize=(5, 5))
        mne.viz.plot_topomap(instant, raw_data.info, axes=ax)
        plt.show()

    def print_topomap(self, ms):
        """ Imprime mapa topográfico a partir de un microestado

        Args:
            ms (np.array):
                Microestado
        """
        ch_types_str = ['eeg']*len(self.ch_names)

        info = mne.create_info(ch_names=self.ch_names, sfreq=self.freq, ch_types=ch_types_str)

        mne.viz.plot_topomap(ms, info, cmap="hot", show=True)

    def get_occurrence(self, vector):
        """ Retorna la ocurrencia de cada letra en un vector

        Args:
            vector (np.array):
                Vector con letras correspondientes a los microestados

        Returns:
            ocurrence (dict):
                Diccionario con la ocurrencia de cada letra
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
            vector (np.array):
                Vector con letras correspondientes a los microestados
            freq (int):
                Frecuencia de muestreo

        Returns:
            durations (dict): Diccionario con la duración promedio de cada letra
        """
        letters = ["A", "B", "C", "D"]
        grouped_letters = []
        grouped_letters.append(vector[0])

        for i in range(1, len(vector)):
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

    def get_metrics(self, vector):
        """
        Retorna la cobertura, ocurrencia y duración de cada microestado

        Args:
            vector (np.array):
                Vector con letras correspondientes a los microestados
            freq (int):
                Frecuencia de muestreo

        Return:
            tuple (dict, dict, dict):
                (coverage, occurrence, durations)
                coverage (dict):
                    Diccionario con la cobertura de cada letra
                occurrence (dict):
                    Diccionario con la ocurrencia de cada letra
                durations (dict):
                    Diccionario con la duración promedio de cada letra
        """
        coverage = self.get_coverage(vector)
        occurrence = self.get_ocurrence(vector)
        durations = self.get_duration(vector, self.freq)

        return coverage, occurrence, durations

    def correlate_microstates(self, eeg_matrix, microstates):
        """
        Calcula la correlación entre cada punto de tiempo de la matriz EEG y los microestados.
        Args:
            eeg_matrix (np.array):
                Matriz con los datos del EEG.
            microstates (list):
                Lista con los microestados.

        Returns:
            np.array:
                Matriz con las correlaciones de cada punto de tiempo con cada microestado.
        """
        num_timepoints = eeg_matrix.shape[1]
        matched_states = [''] * num_timepoints
        letter_mapping = ['A', 'B', 'C', 'D']

        for t in range(num_timepoints):
            correlation_aux = 0
            for idx, m in enumerate(microstates):
                correlation = pearsonr(eeg_matrix[:, t], m[:, 0])[0]
                if correlation > correlation_aux:
                    correlation_aux = correlation
                    best_microstate = letter_mapping[idx]
            matched_states[t] = best_microstate

        return matched_states

    def evaluate_a_pacient(self, path: str):
        """
        Evalua un paciente a partir de su matriz EEG y los microestados generales de la clase.

        Args:
            path (str): Ruta del archivo .csv a procesar

        Returns:
            tuple (dict, dict, dict):
                (coverage, occurrence, durations)
                coverage (dict):
                    Diccionario con la cobertura de cada letra
                occurrence (dict):
                    Diccionario con la ocurrencia de cada letra
                durations (dict):
                    Diccionario con la duración promedio de cada letra
        """
        _, electrodes = self.read_data(path)
        matched_states = self.correlate_microstates(
            electrodes, self.general_microstates)

        return self.get_metrics(matched_states)
