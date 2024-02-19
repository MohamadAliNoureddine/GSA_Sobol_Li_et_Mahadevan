import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.models import load_model
saved_model_directory_Standard= "/home/noureddine/codes/efispec3d.git/efispec3d-interne/test/Dumanoir/Neural-Network/StandardNN"
loaded_model_Standard = load_model(saved_model_directory_Standard)
freq_min=0
freq_max=0.3333333e+03
num_freq_points=16385
frequences=np.linspace(freq_min,freq_max,num_freq_points)

def model_standard(input_1,input_2,input_3,input_4,input_5,input_6):
    inputs = np.column_stack((input_1, input_2,input_3,input_4,input_5,input_6))
    sortie_standard=loaded_model_Standard.predict(inputs)
    return sortie_standard

######### Plan d'experience pour l'analyse de sensibilité globale en utilisant la suite de Sobol
from SALib.sample import saltelli
nombre_lignes =2048
nombre_colonnes = 12
problem = {
    'num_vars': 12,
    'names': [f'input_{i}' for i in range(12)],
    'bounds': [[1150, 1650],[2150,2650],[3750,4250],[14068,15068],[17553.33,18583.33],[-28520,-27520],[1150, 1650],[2150,2650],[3750,4250],[14068,15068],[17553.33,18583.33],[-28520,-27520]]
}
Uncertain_parameters= saltelli.sample(problem, nombre_lignes)
min_colon1= np.min(Uncertain_parameters[:, 0])
max_colon1=np.max(Uncertain_parameters[:, 0])
X11 = (Uncertain_parameters[:, 0] - min_colon1) / (max_colon1 - min_colon1)
min_colon2= np.min(Uncertain_parameters[:, 1])
max_colon2=np.max(Uncertain_parameters[:, 1])
X21= (Uncertain_parameters[:, 1] - min_colon2) / (max_colon2- min_colon2)
min_colon3= np.min(Uncertain_parameters[:, 2])
max_colon3=np.max(Uncertain_parameters[:, 2])
X31= (Uncertain_parameters[:, 2] - min_colon3) / (max_colon3 - min_colon3)
min_colon4= np.min(Uncertain_parameters[:, 3])
max_colon4=np.max(Uncertain_parameters[:, 3])
X41 = (Uncertain_parameters[:, 3] - min_colon4) / (max_colon4 - min_colon4)
min_colon5= np.min(Uncertain_parameters[:, 4])
max_colon5=np.max(Uncertain_parameters[:, 4])
X51 = (Uncertain_parameters[:, 4] - min_colon5) / (max_colon5- min_colon5)
min_colon6= np.min(Uncertain_parameters[:, 5])
max_colon6=np.max(Uncertain_parameters[:, 5])
X61 = (Uncertain_parameters[:, 5] - min_colon6) / (max_colon6 - min_colon6)
min_colon7= np.min(Uncertain_parameters[:, 6])
max_colon7=np.max(Uncertain_parameters[:, 6])
X12 = (Uncertain_parameters[:, 6] - min_colon7) / (max_colon7 - min_colon7)
min_colon8= np.min(Uncertain_parameters[:, 7])
max_colon8=np.max(Uncertain_parameters[:, 7])
X22= (Uncertain_parameters[:, 7] - min_colon8) / (max_colon8- min_colon8)
min_colon9= np.min(Uncertain_parameters[:, 8])
max_colon9=np.max(Uncertain_parameters[:, 8])
X32= (Uncertain_parameters[:, 8] - min_colon9) / (max_colon9 - min_colon9)
min_colon10= np.min(Uncertain_parameters[:, 9])
max_colon10=np.max(Uncertain_parameters[:, 9])
X42 = (Uncertain_parameters[:, 9] - min_colon10) / (max_colon10- min_colon10)
min_colon11= np.min(Uncertain_parameters[:, 10])
max_colon11=np.max(Uncertain_parameters[:, 10])
X52= (Uncertain_parameters[:, 10] - min_colon11) / (max_colon11- min_colon11)
min_colon12= np.min(Uncertain_parameters[:, 11])
max_colon12=np.max(Uncertain_parameters[:, 11])
X62= (Uncertain_parameters[:, 11] - min_colon12) / (max_colon12 - min_colon12)

###########Méthode de Sobol
Output = model_standard(X11,X21,X31,X41,X51,X61)
output_mean=np.mean(Output,axis=0)
square_output_mean=output_mean**2
Variance_total=np.var(Output,axis=0)
def indice_de_sobol_ordre1(input1,input2,input3,input4,input5,input6):
    conditional_output=model_standard(input1,input2,input3,input4,input5,input6)
    a=Output*conditional_output
    U=np.mean(a,axis=0)
    V=U-square_output_mean
    return V/Variance_total
V_s1=indice_de_sobol_ordre1(X11,X22,X32,X42,X52,X62)
V_s2=indice_de_sobol_ordre1(X12,X21,X32,X42,X52,X62)
V_s3=indice_de_sobol_ordre1(X12,X22,X31,X42,X52,X62)
X_source=indice_de_sobol_ordre1(X12,X22,X32,X41,X52,X62)
Y_source=indice_de_sobol_ordre1(X12,X22,X32,X42,X51,X62)
Z_source=indice_de_sobol_ordre1(X12,X22,X32,X42,X52,X61)
indices_sobol = np.array([V_s1, V_s2, V_s3, X_source, Y_source, Z_source])
sobol_indices_matrix = np.array(indices_sobol)
cumulative_influence = np.zeros_like(sobol_indices_matrix[0])
noms_parametres = ['Vs1', 'Vs2', 'Vs3', 'Xsource', 'Ysource', 'Zsource']
n_params = sobol_indices_matrix.shape[0]
plt.figure(figsize=(10, 8))
for param in range(n_params):
    influence = sobol_indices_matrix[param]
    plt.fill_between(frequences, cumulative_influence, cumulative_influence + influence, alpha=0.3, label=noms_parametres[param])
    cumulative_influence += influence
plt.title('Méthode de Sobol')
plt.xlabel('Fréquence [Hz]')
plt.ylabel('Indices de Sobol cumulées')
plt.xlim(0, 1.3)
plt.legend()
#plt.show()

############################Méthode de Li et mahadevan
X_in=Uncertain_parameters[:, :6]
y_in =Output
def compute_sobol_index(x, y, ninter, nminvar):
    min_x = np.min(x)
    max_x = np.max(x)
    dx = (max_x - min_x) / ninter
    pi_hist = np.linspace(min_x, max_x, ninter + 1)
    VectVariancePI = np.zeros(ninter)
    for i in range(ninter):
        mask = np.logical_and(x >= pi_hist[i], x <= pi_hist[i + 1])
        nmask = np.count_nonzero(mask)
        if nmask > nminvar:
            y_mask = y[mask]
            print(f"Interval {i}:")
            print(y_mask.shape)
            VectVariancePI[i] = np.var(y_mask)
        else:
            print('Erreur : Nombre insuffisant de points pour calculer la variance')
    S = 1 - np.mean(VectVariancePI) / np.var(y)
    return S
n_params = X_in.shape[1]  # Nombre de paramètres incertains
n_times = y_in.shape[1]   # Nombre de pas de temps dans les sorties
sobol_indices = np.zeros((n_params, n_times))  # Matrice pour stocker les indices de Sobol
for param in range(n_params):  # Itération sur chaque paramètre incertain
    x = X_in[:, param]  # Sélection du paramètre
    for time_step in range(n_times):  # Itération sur chaque pas de temps
        y = y_in[:, time_step]  # Données pour ce pas de temps
        # Calcul des indices de Sobol pour ce paramètre à ce pas de temps
        Sobol_index = compute_sobol_index(x, y, ninter=30,nminvar=20)
        # Stockage de l'indice de Sobol pour ce paramètre à ce pas de temps
        sobol_indices[param, time_step] = Sobol_index
plt.figure(figsize=(10, 8))
cumulative_influence = np.zeros_like(sobol_indices[0])
noms_parametres = ['Vs1', 'Vs2', 'Vs3', 'Xsource', 'Ysource', 'Zsource']
for param in range(n_params):
    influence = sobol_indices[param]
    plt.fill_between(frequences, cumulative_influence, cumulative_influence + influence, alpha=0.3,label=noms_parametres[param])
    cumulative_influence += influence
plt.title('Méthode de Li et Mahadevan')
plt.xlabel('Fréquence [Hz]')
plt.ylabel('Indices de sobol cumulées ')
plt.xlim(0, 1.3)
plt.legend()
plt.show()



################Sobol vs Li et Mahadevan
fig, axs = plt.subplots(2, 3, figsize=(15, 10))
axs[0, 0].plot(frequences,sobol_indices[0], label=f'Méthode de Li et Mahadevan')
axs[0,0].plot(frequences,V_s1,label="Méthode de Sobol")
axs[0,0].set_xlabel('Frequency[Hz]')
axs[0,0].set_xlim(0,1.25)
axs[0,0].set_ylabel('Indice de Sobol')
axs[0,0].set_title('Influence de Vs1')
axs[0,0].legend()
axs[0, 1].plot(frequences,sobol_indices[1], label=f'Méthode de Li et Mahadevan' )
axs[0, 1].plot(frequences, V_s2,label="Méthode de Sobol" )
axs[0, 1].set_xlabel('Frequency[Hz]')
axs[0, 1].set_xlim(0,1.25)
axs[0, 1].set_ylabel('Indice de Sobol')
axs[0,1].set_title('Influence de Vs2')
axs[0, 1].legend()
axs[0, 2].plot(frequences, sobol_indices[2], label=f'Méthode de Li et Mahadevan')
axs[0, 2].plot(frequences, V_s3, label="Méthode de Sobol")
axs[0,2].set_title('Influence de Vs3')
axs[0, 2].set_xlabel('Frequency[Hz]')
axs[0, 2].set_xlim(0,1.25)
axs[0, 2].set_ylabel('Indice de Sobol')
axs[0, 2].legend()
axs[1, 0].plot(frequences,sobol_indices[3], label=f'Méthode de Li et Mahadevan')
axs[1, 0].plot(frequences,X_source,label="Méthode de Sobol")
axs[1,0].set_title('Influence de Xsource')
axs[1, 0].set_xlabel('Frequency[Hz]')
axs[1, 0].set_xlim(0,1.25)
axs[1, 0].set_ylabel('Indice de Sobol')
axs[1, 0].legend()
axs[1, 1].plot(frequences, sobol_indices[4], label=f'Méthode de Li et Mahadevan')
axs[1, 1].plot(frequences, Y_source,label="Méthode de Sobol")
axs[1,1].set_title('Influence de Ysource')
axs[1, 1].set_xlabel('Frequency[Hz]')
axs[1, 1].set_xlim(0,1.25)
axs[1, 1].set_ylabel('Indice de Sobol')
axs[1, 1].legend()
axs[1, 2].plot(frequences, sobol_indices[5], label=f'Méthode de Li et Mahadevan')
axs[1, 2].plot(frequences, Z_source,label="Méthode de Sobol")
axs[1,2].set_title('Influence de Zsource')
axs[1, 2].set_xlabel('Frequency[Hz]')
axs[1, 2].set_xlim(0,1.25)
axs[1, 2].set_ylabel('Indice de Sobol')
axs[1, 2].legend()

plt.tight_layout()
plt.show()

