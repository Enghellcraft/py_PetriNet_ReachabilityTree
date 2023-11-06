import numpy as np
from numpy.linalg import svd
from sympy import *
import networkx as nx
import matplotlib.pyplot as plt

NumberOfIt = 0
Trans = 0
MaxMarking = 0
MarkingList = []
Cyclic = False
Dead = False
TabIndex = 0

def main(input, output, state):
    global NumberOfIt
    global MarkingList
    global Dead
    global Cyclic
    global TabIndex
    global Trans
    global MaxMarking

    if np.shape(input) != np.shape(output):
        print("Error: Matrices de Input y Output deben tener las mismas dimensiones")
        return
    if np.shape(input)[0] != np.shape(state)[0]:
        print("Error: El Estado Inincial de la Matriz es erroneo, debe ser de- " + str(np.shape(input)[0]))
        return

    # Matriz de incidencia
    A = output - input

    transitions = GetTransitions(input, state)
    
    NumberOfIt = NumberOfIt + 1
    if NumberOfIt > 20:
        transitions[0, Trans] = 0
    if NumberOfIt > 30:
        return
    if sum(transitions[0, :]) == 0:
        # Marking DEAD
        Dead = True
        print("Dead")
    elif sum(transitions[0, :]) > 1:
        # Multiples Ramas
        for count in range(0, np.shape(transitions)[1]):
            u = np.zeros([1, np.shape(transitions)[1]])
            if transitions[0, count] == 1:
                Trans = count
                u[0, count] = 1
                # print("Se produce una rama")
                NM = NextMarking(A, state, u.T)
                MaxMarking = CheckMaxMarking(NM, MaxMarking)
                found = False
                for elm in MarkingList:
                    if np.array_equal(elm, NM):
                        found = True
                        break
                if found:
                    Cyclic = True
                    for i in range(TabIndex):
                        print('    ', end=' ')
                    print("Ciclo Encontrado" + str(NM.T))
                else:
                    MarkingList.append(NM)
                    for i in range(TabIndex):
                        print('    ', end=' ')
                    print(str(NM.T) + "--> Realizada Transición: " + str(Trans + 1))
                    TabIndex = TabIndex + 1
                    main(input, output, NM)
                    TabIndex = TabIndex - 1

    else:
        # Camino Único
        Trans += 1
        if(Trans == 1):
            print("Transitions Nro:", Trans, ": ",transitions)
        NM = NextMarking(A, state, transitions.T)
        MaxMarking = CheckMaxMarking(NM, MaxMarking)
        found = False
        for elm in MarkingList:
            if np.array_equal(elm, NM):
                found = True
                break

        if found:
            Cyclic = True
            for i in range(TabIndex):
                print('    ', end=' ')
            print("Ciclo encontrado" + str(NM.T))
        else:
            MarkingList.append(NM)
            for i in range(TabIndex):
                print('    ', end=' ')
            print("Transitions Nro:", (Trans+1), ": ", NM.T)
            main(input, output, NM)      
            
def CheckMaxMarking(nextMarking, MaxMarking):
    if max(nextMarking) > MaxMarking:
        return int(np.max(nextMarking, axis=None))
    return MaxMarking

def GetTransitions(input, state):
    # Determina que transiciones pueden dispararse
    u = np.zeros([1, np.shape(input)[1]])
    for i in range(0, np.shape(input)[1]):
        if np.amin(state.T - input[:, i]) > -1:
            u[0, i] = 1
    return u

def NextMarking(A, M, u):
    MPrime = M + np.dot(A, u)
    # print(MPrime.T)
    return MPrime

def InvarientSolver(input, output):
    A = Matrix(output - input)

    x = A.nullspace()
    tInvarient = len(x) > 0

    temp_A = Matrix(A.T)
    x = temp_A.nullspace()
    pInvarient = len(x) > 0

    return tInvarient, pInvarient

def nullspace(A, atol=1e-13, rtol=0):
    A = np.atleast_2d(A)
    u, s, vh = svd(A)
    tol = max(atol, rtol * s[0])
    nnz = (s >= tol).sum()
    ns = vh[nnz:].conj().T
    return ns



def draw_petri_net(input, output):
    # Crea grafo dirigido
    G = nx.DiGraph()

    # Agrega Nodos Lugar
    for i in range(np.shape(input)[1]):
        G.add_node(f'p{i}', node_type='Lugar')

    # Agrega Nodos Transicion
    for i in range(np.shape(input)[0]):
        G.add_node(f't{i}', node_type='Transicion')

    # Agrega los arcos
    for i in range(np.shape(input)[0]):
        for j in range(np.shape(input)[1]):
            if input[i, j] != 0:
                G.add_edge(f'p{j}', f't{i}', edge_type='Input')
            if output[i, j] != 0:
                G.add_edge(f't{i}', f'p{j}', edge_type='Output')

    # Gráfico Circular
    pos = nx.circular_layout(G)

    # Grafica de Nodos
    nx.draw_networkx_nodes(G, pos, nodelist=[n for n, d in G.nodes(data=True) if d['node_type'] == 'Lugar'], node_shape='o', node_color='lightblue')
    nx.draw_networkx_nodes(G, pos, nodelist=[n for n, d in G.nodes(data=True) if d['node_type'] == 'Transicion'], node_shape='s', node_color='lightgreen')

    # Grafica de arcos
    nx.draw_networkx_edges(G, pos, edgelist=[(u, v) for u, v, d in G.edges(data=True) if d['edge_type'] == 'Input'], edge_color='pink')
    nx.draw_networkx_edges(G, pos, edgelist=[(u, v) for u, v, d in G.edges(data=True) if d['edge_type'] == 'Output'], edge_color='purple')

    nx.draw_networkx_labels(G, pos)
    plt.show()

#input = np.asarray([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
#output = np.asarray([[0, 1, 0, 0], [1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]])
#initialState = np.asarray([[1], [0], [1], [1], [1]])

#input = np.asarray([[1, 0, 0],[1, 0, 0], [1, 0, 1], [0, 1, 0]])
#output = np.asarray([[1, 0, 0], [0, 1, 0], [0, 1, 0],[0, 0, 1]])
#initialState = np.asarray([[1], [0], [1], [0]]) 

# T Invarient 
#input = np.asarray([[0, 1, 0, 2], [1, 1, 0, 0], [0, 0, 1, 0]])
#output = np.asarray([[1, 0, 0, 2], [0, 0, 1, 0], [0, 2, 0, 0]])
#initialState = np.asarray([[1], [1], [0]])

# P Invarient 
# input = np.asarray([[2, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [1, 0, 0, 1]])
# output = np.asarray([[0, 2, 0, 0], [1, 0, 0, 0], [0, 0, 0, 1], [1, 0, 1, 0]])
# initialState = np.asarray([[3], [0], [1], [0]])

input = np.asarray([[1,0,0],[0,1,0],[0,0,1]])
output = np.asarray([[0,1,0],[0,0,1],[1,0,0]])
initialState = np.asarray([[1],[0],[0]])

main(input, output, initialState)
draw_petri_net(input, output)
tInvarient, pInvarient = InvarientSolver(input, output)
print("T-Invarient = " + str(tInvarient))
print("P-Invarient = " + str(pInvarient))
print("Ciclo encontrado = " + str(Cyclic))
print("Dead Encontrado = " + str(Dead))
if MaxMarking > 6:
    print("La red de Petri es NO Acotada")
else:
    print("La red de Peri es " + str(MaxMarking) + " acotada")

"""
 REDES DE PETRI
Las Redes de Petri son una metodología para el modelado y el estudio de diversos 
sistemas, ya que a diferencia de otros modelos gráficos de comportamiento
dinámico, son una herramienta matemática que admite una representación gráfica 
que facilita el análisis y las modificaciones locales del modelo; permitiendo la
representación clara y condensada del paralelismo y la sincronización, llevando 
el modelo a condiciones límite, las cuales en un modelo real serían difíciles de 
lograr o con alto costo de implementación.

ÁRBOL DE ALCANZABILIDAD
Dada una red de Petri (N,M_0) desde una marca inicial M_0 podemos obtener tantas nuevas
marcas como transiciones habilitadas. Así, de cada nueva marca podemos obtener más marcas. 
Este proceso genera un árbol de marcas o árbol de Alcanzabilidad.
Los nodos representan las marcas generadas a partir de M_0 (la raíz) y sus sucesores, y 
cada arco representa un disparo de una transición, la cual transforma una marca en otra.

Algunas de las propiedades que se pueden estudiar utilizando el árbol de 
alcanzabilidad T para una red de Petri (N,M_0) son las siguientes:
  * Una red (N,M_0) es acotada y así R(N,M_0) es finito si y solo si \omega no aparece
    en nigún nodo etiquetado en T.
  * Una red (N,M_0) es libre si y solo si solo aparecen ceros y unos en las etiquetas de 
    los nodos de T.
  * Una transición t es muerta si y solo si no aparece como etiqueta de un arco en T.
  * Si M es alcanzable desde M_0, entonces existe un nodo etiquetado M'$tal que M <= M'.


La Matriz de Incidencia es una herramienta matemática que se utiliza para representar 
relaciones entre dos conjuntos de elementos. Esta matriz se compone de una estructura
rectangular formada por filas y columnas, donde cada fila representa un elemento del
primer conjunto y cada columna representa un elemento del segundo conjunto.

    * Matriz de incidencia binaria: Esta matriz se utiliza cuando solo se quiere
    conocer si hay o no una relación entre los elementos de los dos conjuntos.
    * Matriz de incidencia ponderada: Esta matriz se utiliza cuando se quiere asignar
    un valor numérico a la relación entre los elementos de los dos conjuntos.

En la funcion Main se ingresa una matriz de input y otra de output, cuya
diferencia resulta en la matriz de incidencia. esta matriz describe el efecto
de disparo de cada transicion por cada lugar de la red.
El State refiere al estado inicial de los tokens en la red, a partir de alli
se simula el disparo de las transiciones, hasta encontrar deadlocks o ciclos
y renueva los markings de la red.

La función check max marking, toma las marcas siguientes y máximas, para
compararlas y agregar la proxima máxima marca.

La función transitions toma el input como referencia y los markings actuales
para comprobar que transiciones pueden ser disparadas.

Next Marking toma la matriz de incidencia, la marca inicial y un vector
constante u, que contiene las marcas necesarias para llevar a cabo la 
secuencia. Donde NextMarking = M0 + u * A (Ecuación de estado), devolviendo
así la siguiente marca cuando se dispare la transicion.

El sistema calcula tambien las invariantes de la red:
el invariante de lugar es un vector de ponderación n x 1 γ=transpose([γ0 γ1 ... γN])
tal que A*u=0, donde A es la matriz de incidencia de la Red de Petri.
La ecuación de estado de alguna red se escribe como M = M0 + vA (siendo M0 la marca 
del estado inicial, M la marca de algún otro estado, u la suma de los vectores de disparo
para alcanzar la marca M). A partir de ahí, podemos escribir M * γ = M0 * γ + u * A * γ ==> M * γ = M0*γ.
La última ecuación se deriva por definición, como A * γ = 0. Como la ecuación M = M0 + u * A es 
válida para cualquier estado posterior alcanzable desde M0, esto significa que el número de 
tokens ponderados con la invariante de lugar seguirá siendo el mismo (es una constante) para 
todos los estados alcanzables. Sin embargo, hay que tener en cuenta que para diferentes marcas 
iniciales, esta constante normalmente no será la misma.
Las invariantes son afirmaciones que se garantizan como verdaderas en 
todos los estados alcanzables de una red determinada. Las invariantes 
estructurales son útiles para derivar ciertas propiedades, como conservación,
vivacidad, estados de origen y consistencia, entre otras. 
Aquí veremos dos tipos de invariantes:
* Invariante T:  identifica un conjunto de disparos de transición que 
pueden devolver la red a la misma marca, lo que indica un posible bucle.
* Invariante P: muestra un conjunto de lugares en los que la suma ponderada 
de sus tokens permanece constante en cualquier marca posible alcanzable, 
independientemente de la marca inicial. Estas invariantes se pueden 
utilizar para demostrar la exclusión mutua y pueden verse como un 
componente neto que preserva el token. 

Por ultimo Draw Petri toma el input y output para cancular un grafo dirigido
donde peuden verse los nodos: lugares y transiciones, además de los arcos de 
entrada y salida de los nodos.
    """
