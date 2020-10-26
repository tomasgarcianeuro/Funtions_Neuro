import numpy as np
import matplotlib.pyplot as plt
import pickle


# =============================================================================
# NO CORRER_ USADO COMO PLANTILLA PARA DEMOSTRACION A SUB DIR ( EN PRESENCIA )  DEL PASO A PASO
#  SUB DIR SE REHUSA A CREER EN EL DEBUGGEAR ()
# =============================================================================

#                            """LEVANTANDO DATOS """
"""Lectura de datos en brutos"""

#####Lectura de la posicion####
Posicion_X_Y=np.loadtxt('/home/tomasg/Escritorio/Neuro/Lectura de Datos/jp693-05062015-0108.whl')
#####Creo el vector que contiene el eje temporal de la posicion####
Eje_Temporal=np.zeros([Posicion_X_Y.shape[0]])
#####Lectura de los disparos (Clusers)####
Disparo_Clusters=np.loadtxt("/home/tomasg/Escritorio/Neuro/Lectura de Datos/jp693-05062015-0108.clu")
#####Tiempo en el que suceden los disparos####
Tiempo_Disparo_Clusters=np.loadtxt("/home/tomasg/Escritorio/Neuro/Lectura de Datos/jp693-05062015-0108.res")
#####Tiempo en el que se da inicio y fina de la luz:::Indica el tiempo en los que hay luz y oscuridad####
Tiempo_Luz_oscuridad=np.loadtxt("/home/tomasg/Escritorio/Neuro/Lectura de Datos/jp693-05062015-0108.light_trials_intervals",dtype=str)
#                            """Primeros procesos"""

"""INICIALIZANDO VECTOR TEMPORAL"""

0.for i in range(Eje_Temporal.shape[0]):
    Eje_Temporal[i]=400*(i+1)
    
"""Reestructuro los datos posicion"""

Posicion_X_Y=Posicion_X_Y.reshape([2,427125])
pos_X=Posicion_X_Y[0]
pos_Y=Posicion_X_Y[1]
del Posicion_X_Y        #Elimino el vector para liberar espacio de memoria RAM

 #                           """Funciones """
"""Division de vector disparo en cluster"""
def Division_luz_oscuridad(clu,tiempo_l_o):
    """La funcion logra separar, a pártir de una columna clu (donde se encuentran los tiempos de los picos para un cluster), en dos columnas (dentro
    de un diccionario cada una con los tiempos en los que se disparo el cluster, pero ya discriminados en luz y oscuridad"""
    inicio=0    #Tiempo en que inicia el periodo de luz u oscuridad
    final=0     # Tiempo en que finaliza el periodo de luz u oscuridad
    cluster_aux={}
    cluster_aux['Luz']=np.array([])
    cluster_aux['Oscuridad']=np.array([])
    inicio=float(Tiempo_Luz_oscuridad[0][2])              #Determino donde inica el evento
    for i in range(tiempo_l_o.shape[0]):
        final=float(Tiempo_Luz_oscuridad[i][3])         #Donde finaliza el evento 
        if Tiempo_Luz_oscuridad[i][1][0]=="l":          #Determino si estamos en luz "l" y oscuridad "d"
            cluster_aux["Luz"]=np.append(cluster_aux["Luz"],clu[np.argmax(clu>inicio):np.argmin(clu<final)])    #Agrego al cluster correspondiente los picos correspondienes
            inicio=final                                                                                        # al periodo de luz
        else:
            cluster_aux["Oscuridad"]=np.append(cluster_aux["Oscuridad"],clu[np.argmax(clu>inicio):np.argmin(clu<final)])    #Agrego al cluster correspondiente los picos correspondienes
            inicio=final 
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
    return dic

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

def Graficar_Disparo(disp_clu,posx,posy):
    """Funcion que grafica los puntos donde hubo disparo"""
    X=list(posx)
    Y=list(posy)
    for i in disp_clu:
        plt.plot(X[int(i)],Y[int(i)],"Dr")


def Graficar_selectivo(disp_clu,posx,posy,ma,mi):
    """Funcion que grafica los puntos donde hubo disparo"""
    X=list(posx)
    Y=list(posy)
    for i in disp_clu:
        if X[int(i)]<ma and X[int(i)]>mi and Y[int(i)]<ma and Y[int(i)]>mi:
            plt.plot(X[int(i)],Y[int(i)],"Dr")


def Graficar_posicion(posx,posy,disp_clu):
    """"Funcion que grafica la trayectoria del SUJETO"""
    maximo=int(max(disp_clu))
    minimo=int(min(disp_clu))
    pos_x=[];pos_y=[]
    for i in range(minimo,maximo):
        pos_x.append(posx[i]),pos_y.append(posy[i])
    plt.plot(pos_x,pos_y,linestyle="--",color="grey")               
            
#                            """Procesamiento"""
        
Dicc_clu=Cargar("/home/tomasg/Escritorio/Neuro/Lectura de Datos/Generacion de Dicc_Clu/Diccionarios_Datos/jp693/Diccionario_0506")                           #Elemento ya creado con Division_clu_
Posicion_disparo=Disparos(Dicc_clu["Cluster Numero 10"],Eje_Temporal)
np.save("dicc.pickle",Dicc_clu,allow_pickle=True)
da=np.array([])
da=Guardar("Archivo_Donny",Dicc_clu)
