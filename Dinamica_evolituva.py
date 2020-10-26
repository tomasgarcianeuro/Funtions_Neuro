#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct  6 12:46:22 2020

@author: tomasg
"""
import sys
import numpy as np
import matplotlib.pyplot as plt
import pickle 
from scipy.ndimage import gaussian_filter as gauss_fil
import pandas as pd
import pickle
from My_Functions import *
import time
# =============================================================================
# Los modulos cargados realizaran todo el trabajo fuerte.
# Aquí unicamente se organizara el trabajo. Agrego funciones: "nuevo calculo de mapa disparo".
# Sub Dir. : Documentacion en ingles en proceso 
# =============================================================================

# =============================================================================
# Analisis completos (autocor, croscor, grid score, en funcion del tiempo.)
# Evaluar resultados (coherencia), tiempo requerido para un analisis de un par, overlap optimo [time and information]
# =============================================================================

def Cargar(name):
    """Cargar archivos"""
    with open (name, "rb") as hand:
        dic=pickle.load(hand)
    return dic


def Return_keys(dic, num_clu):
    """
    """
    Luz = dic["Cluster Numero {}".format(num_clu)]["Luz"].keys()
    Oscuridad = dic["Cluster Numero {}".format(num_clu)]["Oscuridad"].keys()
    return list(Luz), list(Oscuridad)

def Return_time_spike_sort(num_clu, dic):
    """Return a DataFrame with three columns Time Light  Cluster
    The DF is increasingly organized by the column time.
    The column time have the time where a cluster shot"""
    time = []
    light = []
    cluster = []
    keys_luz, keys_oscuridad = Return_keys(dic, num_clu)
    for luz in ["Luz", "Oscuridad"]:
        if luz == "Luz":
            keys=keys_luz
        else:
            keys = keys_oscuridad
        for key in keys:
            serie_times = dic["Cluster Numero {}".format(num_clu)][luz][key]
            list(map( time.append, serie_times ))
            list(map( light.append, [key[:2]]*len(serie_times)))
            list(map( cluster.append, [num_clu]*len(serie_times)))
    dic_prov = {"Time":time, "Light":light, "Cluster":cluster}        
    df1 = pd.DataFrame(dic_prov)
    df1.sort_values("Time", inplace = True)
    df1.set_index(pd.Index(range(len(df1))), inplace=True)
    return df1


def Time_axis(dic):
    Eje_Temporal=np.zeros(dic["Posicion"].T.shape[1])
    for i in range(Eje_Temporal.shape[0]):
        Eje_Temporal[i]=400*(i+1)
    return Eje_Temporal

def Map_Time(position, windows, bins = 25.0001):
    """Return the heat map about the time that the animal keep in some bins"""
    Tasa_disparo = np.zeros([int(900/bins),int(900/bins)])
    real_position = Cutout_position(position, windows)
    for x, y in real_position:
        if (y!=-1 or x!=-1):
            Tasa_disparo[int(y/bins)][int(x/bins)]+=400
    return Tasa_disparo/20000       #Se divide por 20k para obtener la matriz expresada en segundos

    
def Vector_shouts(spike):
    """Return a vector where eache value represent a index in matrix of position animal"""
    nex = lambda arg: round(arg/400)-1
    vector = list( map(nex, spike))
    return vector

def Map_shouts(vector_shouts, position, bins = 25.0001):
    """"""
    Tasa_disparo = np.zeros([int(900/bins),int(900/bins)])
    for index in vector_shouts:
        pos = position[index]
        if (pos[0]!=-1 or pos[1]!=-1):
            Tasa_disparo[int(pos[1]/bins)][int(pos[0]/bins)]+=1
    return Tasa_disparo

def Rate(map_shouts, map_time):
    
    division = map_shouts/map_time
    mapa = np.where(map_time != 0, division, 0)
    return mapa
    
    
def Map_Rate(spike, position, windows):
    """This funcion return a rate-spike mape"""
    map_time = Map_Time(position, windows)
    vector_shouts = Vector_shouts(spike)
    map_shouts = Map_shouts(vector_shouts, position)
    rate = Rate(map_shouts, map_time)
    return rate

def Cutout_spike(time_spike_sort, windows):
    """"""    
    time = time_spike_sort["Time"]
    time = time[(time>=windows[0]) & (time<=windows[1])]
    return time

def Cutout_position(position, windows):
    min_pos = int(windows[0]/400)-1
    max_pos = int(windows[1]/400)
    pos = position[min_pos : max_pos]
    return pos  
 
def Create_windows(dic):
    low_time = int(dic["Light_Trials"][0][2])
    high_time = int(dic["Light_Trials"][59][3])
    scale = high_time - low_time
    return low_time, high_time, scale

def Return_map_rate(dic, num_clu, size_windows = 120, overlap = 2):
    """Return map of spike rate with values in windows"""
    size_windows *= 20000
    overlap *= 20000
    time_spike_sort = Return_time_spike_sort(num_clu, dic)
    time_axis = Time_axis(dic)
    low_time, high_time, scale = Create_windows(dic)    
    num_windows = int((scale-size_windows)/overlap ) 
    high_time = low_time + size_windows
    for i in range(num_windows):
        windows = [low_time, high_time]
        spike = Cutout_spike(time_spike_sort, windows)
        mapa_rate = Map_Rate(spike, dic["Posicion"], windows)
        low_time += overlap
        high_time += overlap
        mapa_rate = gauss_fil(mapa_rate, 1)
        img = plt.imshow(mapa_rate)
        yield mapa_rate
        
def Process(autocor_1, autocor_2, cross_1, cross_2):
    """Returns the score of cluster 1, 2 and autocross"""
    score1 = Calculo_grid_not_origin_extern_radios_variables(autocor_1)
    score2 = Calculo_grid_not_origin_extern_radios_variables(autocor_2)
    scorecros1 = Calculo_grid_not_origin_extern_radios_variables(cross_1)
    scorecros2 = Calculo_grid_not_origin_extern_radios_variables(cross_2)
    return score1, score2, scorecros1, scorecros2 

def Percentage(windows, Light_Trials, size_windows):
    
    if windows[0]>=int(Light_Trials[0,2]) and windows[0]<=int(Light_Trials[0,3]):
        if windows[1]<=int(Light_Trials[0,3]):
            interval = windows[1]-windows[0]
            return [[str(Light_Trials[0,1]), interval/size_windows*100]]
        else:
            interval = int(Light_Trials[0,3]) - windows[0]
            windows = [int(Light_Trials[0,3]), windows[1]]
            return [[str(Light_Trials[0,1]), interval/size_windows*100]] + Percentage(windows, Light_Trials[1:], size_windows)
    else:
        if len(Light_Trials)>1:
            return Percentage(windows, Light_Trials[1:], size_windows)

    
def Determination_percentage_light_type(dic, size_windows, overlap):
    
    size_windows *= 20000
    overlap *= 20000
    low_time, high_time, scale = Create_windows(dic)
    high_time = low_time + size_windows    
    num_windows = int((scale-size_windows)/overlap )
    type_lights = ["l1", "l2", "l3", "l4", "d1", "d2", "d3", "d4"]
    # lights = [ arg  for arg in type_lights if arg in Dicc_clu["Light_Trials"][:,1]]
    for i in range(num_windows):
        windows = [low_time, high_time]
        lable_pertentage = Percentage(windows, dic["Light_Trials"], size_windows)
        low_time += overlap
        high_time += overlap
        dic_light = {arg:0 for arg in type_lights}
        for i in range(len(lable_pertentage)):
            dic_light[lable_pertentage[i][0]] += lable_pertentage[i][1]
        yield dic_light
        
def Main(dic, clu1, clu2, size_windows = 120, overlap = 2):
    
    Cluster_1 = Return_map_rate(dic, clu1, size_windows, overlap)
    Cluster_2 = Return_map_rate(dic, clu2, size_windows, overlap)
    Light_Percentage = Determination_percentage_light_type(dic, size_windows, overlap)
    score_1 = []
    score_2 = []
    score_cross1 = []
    score_cross2 = []
    light_percentage = {arg:[] for arg in ["l1", "l2", "l3", "l4", "d1", "d2", "d3", "d4"]}
    func_aux_ligt = lambda arg : light_percentage[arg[0]].append(arg[1])
    
    
    val = 0
    total = 120*59
    pequeno = overlap/ total*100
    
    while True:
        try:
            print("{}%".format(val))
            val += pequeno
            map_rate_1 = Cluster_1.__next__()
            map_rate_2 = Cluster_2.__next__()
            autocor_1 = Mapa_Corr(map_rate_1)
            autocor_2 = Mapa_Corr(map_rate_2)
            cross_1 = Mapa_Cross_Corr(map_rate_1, map_rate_2)
            cross_2 = Mapa_Cross_Corr(map_rate_2, map_rate_1)
            score1, score2, scorecross1, scorecross2= Process(autocor_1, autocor_2, cross_1, cross_2)
            score_1.append(score1)
            score_2.append(score2)
            score_cross1.append(scorecross1)
            score_cross2.append(scorecross2)
            list(map(func_aux_ligt, list(Light_Percentage.__next__().items())))
            # Save_images(autocor_1, autocor_2, cross_1, cross_2)
        except StopIteration:
            break
    dic = {"Score {}".format(clu1) : score_1, "Score {}".format(clu2) : score_2, 
           "Score_cros_{}-{}".format(clu1, clu2) : score_cross1, 
           "Score_cros_{}/{}".format(clu2, clu1) : score_cross2} 
    return dic, light_percentage

if __name__ == "__main__":
# =============================================================================
#     Carga manual (). Falta agregar el codigo que analiza todos los cluster de todos los animales automaticamente
#     No se agregro para analizar el resultados preliminar (y evaluar tiempo estimado de calculo)
# =============================================================================
    inicio=time.time()
    Dicc_clu=Cargar("/home/tomasg/Escritorio/Neuro/Lectura de Datos/Generacion de Dicc_Clu/Diccionarios_Datos/jp693/Diccionario_0506")                           #Elemento ya creado con Division_clu_
    Scor1=[]
    Score2=[]
    dic, dic_light = Main(Dicc_clu, 9, 10, 120, 8)
    fin=time.time()
    time_transcurrido=fin-inicio
    minutos=int(time_transcurrido/60)
    seg=time_transcurrido-minutos*60
    print("El programa tardo ", minutos, " minutos y ", seg, " segundos.")

