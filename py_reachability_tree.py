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

    # Find Incidence Matrix
    A = output - input

    

    transitions = GetTransitions(input, state)
    
    NumberOfIt = NumberOfIt + 1
    if NumberOfIt > 20:
        transitions[0, Trans] = 0
    if NumberOfIt > 30:
        return
    if sum(transitions[0, :]) == 0:
        # DEADLOCK
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
            print("Transition " + str(Trans + 1) + ": " + str(NM.T))
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

def NextMarking(A, M, u):
    MPrime = M + np.dot(A, u)
    # print(MPrime.T)
    return MPrime

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

    # Draw the graph with the spring_layout algorithm
    pos = nx.circular_layout(G)

    # Draw the nodes with different shapes and colors based on their node_type
    nx.draw_networkx_nodes(G, pos, nodelist=[n for n, d in G.nodes(data=True) if d['node_type'] == 'Lugar'], node_shape='o', node_color='lightblue')
    nx.draw_networkx_nodes(G, pos, nodelist=[n for n, d in G.nodes(data=True) if d['node_type'] == 'Transicion'], node_shape='s', node_color='lightgreen')

    # Draw the edges with different colors based on their edge_type
    nx.draw_networkx_edges(G, pos, edgelist=[(u, v) for u, v, d in G.edges(data=True) if d['edge_type'] == 'Input'], edge_color='pink')
    nx.draw_networkx_edges(G, pos, edgelist=[(u, v) for u, v, d in G.edges(data=True) if d['edge_type'] == 'Output'], edge_color='purple')

    nx.draw_networkx_labels(G, pos)
    plt.show()

#input = np.asarray([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
#output = np.asarray([[0, 1, 0, 0], [1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]])
#initialState = np.asarray([[1], [0], [1], [1], [1]])

input = np.asarray([[1, 0, 0],
                    [1, 0, 0], 
                    [1, 0, 1], 
                    [0, 1, 0]])
output = np.asarray([[1, 0, 0], 
                     [0, 1, 0], 
                     [0, 1, 0], 
                     [0, 0, 1]])
initialState = np.asarray([[1], [0], [1], [0]])

# T Invarient 
# input = np.asarray([[0, 1, 0, 2], [1, 1, 0, 0], [0, 0, 1, 0]])
# output = np.asarray([[1, 0, 0, 2], [0, 0, 1, 0], [0, 2, 0, 0]])
# initialState = np.asarray([[1], [1], [0]])

# P Invarient 
# input = np.asarray([[2, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [1, 0, 0, 1]])
# output = np.asarray([[0, 2, 0, 0], [1, 0, 0, 0], [0, 0, 0, 1], [1, 0, 1, 0]])
# initialState = np.asarray([[3], [0], [1], [0]])


main(input, output, initialState)
draw_petri_net(input, output)
tInvarient, pInvarient = InvarientSolver(input, output)
print("T-Invarient = " + str(tInvarient))
print("P-Invarient = " + str(pInvarient))
print("Ciclo encontrado = " + str(Cyclic))
print("Muerta = " + str(Dead))
if MaxMarking > 6:
    print("La red de Petri es NO Acotada")
else:
    print("La red de Peri es " + str(MaxMarking) + " acotada")
    