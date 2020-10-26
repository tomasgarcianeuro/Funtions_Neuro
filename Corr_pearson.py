import numpy as np
import matplotlib.pyplot as plt
import pickle
from scipy.ndimage import gaussian_filter as gauss_fil
from os import listdir


# =============================================================================
# NO CORRER_ USADO COMO PLANTILLA PARA DEMOSTRACION A SUB DIR ( EN PRESENCIA )  DEL PASO A PASO
#  SUB DIR SE REHUSA (AÚN) A CREER EN EL DEBUGGEAR ()
# =============================================================================
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""Inicializaciones fundamentales """""""""""""""""""""""""""""""""""""""""""""""""""""""""

"""INICIALIZANDO VECTOR TEMPORAL"""
Eje_Temporal=np.zeros(Diccionario["Posicion"].T.shape[1])
for i in range(Eje_Temporal.shape[0]):
    Eje_Temporal[i]=400*(i+1)


"""""""""""""""""""""""""""""""""""""""""""""""""""""""""Funciones """""""""""""""""""""""""""""""""""""""""""""""""""""""""
"""Division de vector disparo en cluster"""

def Division_luz_oscuridad(clu,Tiempo_Luz_oscuridad):
    """La funcion logra separar, a pártir de una columna clu (donde se encuentran los tiempos de los picos para un cluster), en dos columnas (dentro
    de un diccionario cada una con los tiempos en los que se disparo el cluster, pero ya discriminados en luz y oscuridad"""
    inicio=0    #Tiempo en que inicia el periodo de luz u oscuridad
    final=0     # Tiempo en que finaliza el periodo de luz u oscuridad
    cluster_aux={}
    cluster_aux['Luz']={}
    cluster_aux['Oscuridad']={}
    luz=0       #variable para individualizar los l1,l2,l3,l4 etc.
    oscuridad=0
    inicio=float(Tiempo_Luz_oscuridad[0][2])            #Determino donde inica el evento
    for i in range(Tiempo_Luz_oscuridad.shape[0]):
        final=float(Tiempo_Luz_oscuridad[i][3])         #Donde finaliza el evento 
        if Tiempo_Luz_oscuridad[i][1][0]=="l":          #Determino si estamos en luz "l" y oscuridad "d"
            tipo_luz=str(Tiempo_Luz_oscuridad[i][1][:2])+".{}"
            cluster_aux["Luz"][tipo_luz.format(str(luz))]=clu[np.argmax(clu>inicio):np.argmin(clu<final)]    #Agrego al cluster correspondiente los picos correspondienes
            inicio=final;luz+=1                                                                           # al periodo de luz
        else:
            tipo_oscuridad=str(Tiempo_Luz_oscuridad[i][1][:2])+".{}"
            cluster_aux["Oscuridad"][tipo_oscuridad.format(str(oscuridad))]=clu[np.argmax(clu>inicio):np.argmin(clu<final)]    #Agrego al cluster correspondiente los picos correspondienes
            inicio=final;oscuridad+=1
    return cluster_aux 
            

def Division_clu_(Cluster,tiempo):
    """"Funcion que crea un diccionario con el tiempo de disparo de cada cluster """
    num_clu=int(Cluster[0])                       #Numero de clusters
    Cluster=np.delete(Cluster,0)                  #Elimino la primera fila que indica el numero de cluster
    Cluster=Cluster.reshape([Cluster.shape[0] ,1])
    tiempo=tiempo.reshape([ tiempo.shape[0]  ,1])
    Cluster=np.append(Cluster,tiempo,1)
    CLUSTERS={}
    for i in range(num_clu+1):            #Creeo el diccionario de los clusters
        Nombre_Vector="Cluster Numero {}"
        CLUSTERS[Nombre_Vector.format(i)]=np.array([])
                                        #LLeno los culster
    for i in range(Cluster.shape[0]):
        Num_clu=Cluster[i][0]   #Asigno a Num_clu el identificador de cluster que se activo
        Num_time=Cluster[i][1]  #Asigno a Num_time el tiempo de cluster que se activo referido a Num_clu
        Nombre_Vector="Cluster Numero {}"
        CLUSTERS[Nombre_Vector.format(int(Num_clu))]=np.append(CLUSTERS[Nombre_Vector.format(int(Num_clu))],Num_time)
    return CLUSTERS


def Division_global(Cluster,tiempo,tiempo_l_o):
    """La funcion se apoya en las dos funciones anteriores (Division_clu_ y Division_luz_oscuridad) para devolver un diccionario, que a su vez contiene
    otros diccionarios en la cual tenemos una organizacion de la sigiente manera:
        Dentro del diccionario principal tenemos n diccionarios (siendo n el numero de cluster)
        Dentro de cada uno de esos diccionarios tenemos dos diccionarios más, cada uno con los valores temporales en los que se disparo dicho cluster
    cuando el sujeto se encontraba en luz u oscuridad (dependiendo del diccionario al que pertenece)"""
    num_clu=int(Cluster[0])
    Cluster_Principal=Division_clu_(Cluster, tiempo)    #Creo el diccionario principal
    clu_aux={}
    for i in range(num_clu):
        clu_aux=Division_luz_oscuridad(Cluster_Principal["Cluster Numero {}".format(i+1)],tiempo_l_o)
        Cluster_Principal["Cluster Numero {}".format(i+1)]=clu_aux
    return Cluster_Principal


def Cargar(name):
    """Cargar archivos"""
    with open (name, "rb") as hand:
        dic=pickle.load(hand)
    return dic


def Guardar(name,objecto):
    with open (name, "wb") as hand:
        dic=pickle.dump(objecto,hand)
 
"""Cálculo de tasa de disparo, etc."""       

def Disparos(clu,time):
    """Funcion que crea un vector donde cada elemento es un numero entero que representa el lugar donde estaba el SUJETO 
    cuando se registro el disparo. Esto es, los elementos del vector representan el INDICE 
    en el Eje_Temporal en el que sucedio el disparo."""
    posicion_disparo=np.array([])
    j=0
    for i in range(clu.shape[0]):
        Condicion=True
        while Condicion:
            if (clu[i]>=time[j] and clu[i]<time[j+1]):
                posicion_disparo=np.insert(posicion_disparo,posicion_disparo.shape[0],j)
                break
            j+=1
    return posicion_disparo


def Cuentas(disp_clu,posx,posy,bins):
    """Funcion que determina el número total de disparos den un determinado lugar fisico. Se representa por medio de 
    una matriz"""
    X=list(posx)
    Y=list(posy)
    Tasa_disparo=np.zeros([int(900/bins),int(900/bins)])
    for i in range(disp_clu.shape[0]):
        Tasa_disparo[int(Y[int(disp_clu[i])]/bins)][int(X[int(disp_clu[i])]/bins)]+=1
    return Tasa_disparo

def Tiempo(posx,posy,bins):
    """Funcion que determina el tiempo total que el animal paso en un determinado lugar fisico. Se representa por medio
    de una matriz"""
    X=list(posx)
    Y=list(posy)
    Tasa_disparo=np.zeros([int(900/bins),int(900/bins)])
    for i in range(posx.shape[0]):
        if (Y[i]!=-1 or X[i]!=-1):
            Tasa_disparo[int(Y[i]/bins)][int(X[i]/bins)]+=400
    return Tasa_disparo       #Se divide por 20k para obtener la matriz expresada en segundos

def Tiempo_2(Dicc,bins,type_fuente):
    """Funcion que determina el tiempo total que el animal paso en un determinado lugar fisico en unos determinados 
    intervalos. Se representa por medio de una matriz"""
    Tasa_disparo=np.zeros([int(900/bins),int(900/bins)])
    for i in range(60):
        if Dicc["Light_Trials"][i][1]==type_fuente:
            X=list(Dicc["Posicion"].T[0][int(int(Dicc["Light_Trials"][i][2])/400):int(int(Dicc["Light_Trials"][i][3])/400)+1])
            Y=list(Dicc["Posicion"].T[1][int(int(Dicc["Light_Trials"][i][2])/400):int(int(Dicc["Light_Trials"][i][3])/400)+1])
            for i in range(len(X)):
                if (Y[i]!=-1 or X[i]!=-1):
                    Tasa_disparo[int(Y[i]/bins)][int(X[i]/bins)]+=400
    return Tasa_disparo/20000       #Se divide por 20k para obtener la matriz expresada en segundos


def Tasa_disparo(cuenta,tiempo):
    """Funcion que determina la tasa de disparo"""
    #La variable tiempo es el Eje_temporal
    tasa_verdadera=np.zeros(cuenta.shape)
    for i in range(cuenta.shape[0]):
        for j in range(cuenta.shape[0]):
            if tiempo[i][j]!=0:
                tasa_verdadera[i][j]=cuenta[i][j]/tiempo[i][j]
    return tasa_verdadera

def Cluster_type_light_(Diccionario,bins,luz_oscuridad,type_fuente,num_clu,Eje_Temporal):
    Disparos_clu=np.array([])
    fuente="{}{}".format(str(type_fuente),".{}")
    for i in range(30):
        if fuente.format(i) in Diccionario["Cluster Numero {}".format(str(num_clu))][luz_oscuridad]:
            Disparos_clu=np.concatenate((Disparos_clu,Diccionario["Cluster Numero {}".format(str(num_clu))][luz_oscuridad][fuente.format(i)]))    
    Shots_clu=Disparos(Disparos_clu, Eje_Temporal) #Determino los disparos
    Cuenta=Cuentas(Shots_clu,Diccionario["Posicion"].T[0], Diccionario["Posicion"].T[1], bins)
    Time=Tiempo_2(Diccionario, bins,type_fuente) #Tiempo del individuo en una determinada posicion
    Tasa_de_disparo=Tasa_disparo(Cuenta, Time)
    Imagen_Filtrada=gauss_fil(Tasa_de_disparo, 3)
    plt.imshow(Imagen_Filtrada,cmap="jet")
   # plt.imshow(Tasa_de_disparo,cmap="jet")







def Mover_Imagen(image,x,y):
    
    Diaba=image
    if :
        Da=np.zeros([35,35+x])
        Diaba=np.append(Diaba, Da,axis=1)
        Diaba=np.append(Da[:,:35-x], Diaba,axis=1)
    """
    else:
        Da=np.zeros([35,35+x])
        Diaba=np.append(Diaba[:,:], Da,axis=1)
      #  Diaba=np.append(Da, Diaba,axis=1)
    """
    if (y<=18):
        Da=np.zeros([35+y,105])
        Diaba=np.append(Diaba, Da,axis=0)
        Diaba=np.append(Da[:35-y,:], Diaba,axis=0)
  """
    else :  
        Da=np.zeros([71-35+y,71])
        Diaba=np.append(Diaba[y:,:], Da,axis=0)
       # Diaba=np.append(Da, Diaba,axis=0)
    """
    plt.imshow(Diaba,cmap="jet")        
        
    return Diaba


def Mover_Imagen(image,x,y):
    
    Diaba=image
    if (x>=0):
        Da=np.zeros([35,35+x])
        Diaba=np.append(Diaba, Da,axis=1)
        Diaba=np.append(Da[:,:35-x], Diaba,axis=1)
    elif (x<0):
        Da=np.zeros([35,35-x])
        Diaba=np.append(Da, Diaba,axis=1)
        Diaba=np.append(Diaba,Da[:,:35+x] ,axis=1)
    if (y>=0):
        Da=np.zeros([35+y,105])
        Diaba=np.append(Diaba, Da,axis=0)
        Diaba=np.append(Da[:35-y,:], Diaba,axis=0)
    if (y>0):
        Da=np.zeros([35-y,105])
        Diaba=np.append(Diaba, Da,axis=0)
        Diaba=np.append(Da[:35+y,:], Diaba,axis=0)    
    #plt.imshow(Diaba,cmap="jet")        
        
    return Diaba

 def Corr(x,y):
    return ( np.sum(x*y)/x.shape[0] - np.average(x)*np.average(y) )/(np.std(x)*np.std(y))

def Mapar_Corr(image):
    Mapa=np.zeros([70,70])
    for i in range(70):
        for j in range(70):
            
            Mapa[i][j]=Corr(, )








# "COMENTADO" ASI UNICAMENTE POR DESEO DE SUB DIR _ (MIGUEL, NO TE RIAS QUE DESPUES VENIS A PEDIR AYUDA)
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""Calculo de tasa de disparo"""""""""""""""""""""""""""""""""""""""""""""""""""""""""
"""En esta parte del codigo determino la tasa de disparo de la siguiente manera:
    1) Determino una matriz con los disparos para UN cluster
        la matriz es una representacion cuadricular del espacio. Cada elemento de la matriz representa una porcion del espacio físico por donde el animal se mueve
    2) Determino una matriz del mismo tamaño con el tiempo que estuvo el animal en dicho lugar físico 
    3) Determino, con una division elemento a elemento, la tasa de disparo del animal en dicho lugar
    4) A la matriz resultante Tasa_de_disparo le aplico un filtro gaussiano para suavizar la imagen
    5) Grafico el resultado """

#Antes que nada se cargan los elementos a aplicar el proceso antes descripto
Name="jp5519"
Datos_Animal="{}/Diccionario_{}".format(Name,"0410")
Nombre_Carga="/home/tomasg/Escritorio/Neuro/Lectura de Datos/Generacion de Dicc_Clu/Diccionarios_Datos/{}".format(Datos_Animal)
Diccionario=Cargar(Nombre_Carga) #Cargo el diccionario generado anteriormente
"""CONCATENO LOS INTERVALOS EN LOS QUE SE UTILIZO LA MISMA LUZ l2 O l4"""
Disparos_clu=np.array([])
for i in range(30):
    #if fuente.format(i) in Diccionario["Cluster Numero {}".format(str(num_clu))][luz_oscuridad]:
    try:
        Disparos_clu=np.concatenate((Disparos_clu,Diccionario["Cluster Numero 2"]["Luz"]["l1.{}".format(i)]))    
    except KeyError:
        print("l1.{}".format(i)," No esta")
"""PROCESO LOS DATOS PARA LA MISMA LUZ l2 O l4"""
Shots_clu=Disparos(np.sort(Disparos_clu), Eje_Temporal) #Determino los disparos
Cuenta=Cuentas(Shots_clu,Diccionario["Posicion"].T[0], Diccionario["Posicion"].T[1], 25.0001)
Time=Tiempo_2(Diccionario, 25.0001,"l1") #Tiempo del individuo en una determinada posicion
Tasa_de_disparo=Tasa_disparo(Cuenta, Time)
Imagen_Filtrada=gauss_fil(Tasa_de_disparo, 1)
plt.imshow(Imagen_Filtrada,cmap="jet")
plt.imshow(Tasa_de_disparo,cmap="jet")
#plt.imshow(Tasa_de_disparo,cmap="jet",vmin=0,vmax=1)
#plt.imshow(Imagen_Filtrada,cmap="jet",vmin=0,vmax=1)


#Ver mapa de corre
Diaba=Imagen_Filtrada
Da=np.zeros([35,18])
Diaba=np.append(Diaba, Da,axis=1)
Diaba=np.append(Da, Diaba,axis=1)
Da=np.zeros([18,71])
Diaba=np.append(Diaba, Da,axis=0)
Diaba=np.append(Da, Diaba,axis=0)


Di=np.zeros(Diaba.shape)






Da=Da.reshape([40,35])
Di=np.zeros([70,5])
plt.imshow(Diaba,cmap="jet")
