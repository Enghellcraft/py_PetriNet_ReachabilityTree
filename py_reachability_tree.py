# instalar pip install pnet

import numpy as np
from numpy.linalg import svd
from sympy import *
from pnet import *


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
    global TokenList

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
            
    draw_petri_net(input, output)


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
    # Create places
    places = [Place("p" + str(i)) for i in range(np.shape(input)[0])]

    # Create transitions
    transitions = [Transition("t" + str(i)) for i in range(np.shape(input)[0])]

    # Create arcs
    arcs = []
    for i in range(np.shape(input)[0]):
        if output[i, i] == 1:  # If there is an arc from place i to place i
            arcs.append(Arc(places[i], transitions[i]))
            arcs.append(Arc(transitions[i], places[i]))
        for j in range(np.shape(input)[0]):
            if output[i, j] == 1 and i != j:  # If there is an arc from place i to place j
                arcs.append(Arc(places[i], transitions[j]))
                arcs.append(Arc(transitions[j], places[j]))

    # Create Petri net
    net = PetriNet("net", places, transitions, arcs)

    # Create marking
    marking = Marking({places[i]: 1 for i in range(np.shape(input)[0])})

    # Visualize Petri net
    net.visualize(marking)

input = np.asarray([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
output = np.asarray([[0, 1, 0, 0], [1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]])
initialState = np.asarray([[1], [0], [1], [1], [1]])

# input = np.asarray([[1, 0, 0], [1, 0, 0], [1, 0, 1], [0, 1, 0]])
# output = np.asarray([[1, 0, 0], [0, 1, 0], [0, 1, 0], [0, 0, 1]])
# initialState = np.asarray([[1], [0], [1], [0]])

# T Invarient 
# input = np.asarray([[0, 1, 0, 2], [1, 1, 0, 0], [0, 0, 1, 0]])
# output = np.asarray([[1, 0, 0, 2], [0, 0, 1, 0], [0, 2, 0, 0]])
# initialState = np.asarray([[1], [1], [0]])

# P Invarient 
# input = np.asarray([[2, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [1, 0, 0, 1]])
# output = np.asarray([[0, 2, 0, 0], [1, 0, 0, 0], [0, 0, 0, 1], [1, 0, 1, 0]])
# initialState = np.asarray([[3], [0], [1], [0]])


main(input, output, initialState)
tInvarient, pInvarient = InvarientSolver(input, output)
print("T-Invarient = " + str(tInvarient))
print("P-Invarient = " + str(pInvarient))
print("Ciclo encontrado = " + str(Cyclic))
print("Muerta = " + str(Dead))
if MaxMarking > 6:
    print("La red de Petri es NO Acotada")
else:
    print("La red de Peri es " + str(MaxMarking) + " acotada")
    