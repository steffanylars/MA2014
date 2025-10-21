"""
Hidden Markov Model - Robot Localization
Implementación completa basada en Russell & Norvig Cap. 15
"""

import numpy as np
import random
import matplotlib.pyplot as plt
import seaborn as sns

# ============================================================================
# DEFINICIÓN DE ESTADOS DEL SISTEMA
# ============================================================================

S = {
    0: (0, 0), 1: (1, 0), 2: (2, 0), 3: (3, 0), 4: (5, 0), 5: (6, 0), 
    6: (7, 0), 7: (8, 0), 8: (9, 0), 9: (11, 0), 10: (12, 0), 11: (13, 0), 
    12: (15, 0), 13: (2, 1), 14: (3, 1), 15: (5, 1), 16: (8, 1), 17: (10, 1), 
    18: (12, 1), 19: (1, 2), 20: (2, 2), 21: (3, 2), 22: (5, 2), 23: (8, 2), 
    24: (9, 2), 25: (10, 2), 26: (11, 2), 27: (12, 2), 28: (15, 2), 29: (0, 3), 
    30: (1, 3), 31: (3, 3), 32: (4, 3), 33: (5, 3), 34: (7, 3), 35: (8, 3), 
    36: (9, 3), 37: (10, 3), 38: (12, 3), 39: (13, 3), 40: (14, 3), 41: (15, 3)
}

# ============================================================================
# MODELO DE TRANSICIÓN
# ============================================================================

def neighbors_states(states, location=True):
    """
    Genera un diccionario de estados con listas de estados vecinos o ubicaciones.

    Parameters:
    - states (dict): Diccionario que mapea nombres de estado a posiciones 2D.
    - location (bool): Si es True, devuelve ubicaciones vecinas; si es False, estados vecinos.

    Returns:
    - states_neighbors (dict): Diccionario con cada estado y su lista de vecinos.
    """
    states_neighbors = {}
    
    for state, (x, y) in states.items():
        neighbors = []
        for other_state, (ox, oy) in states.items():
            if state == other_state:
                continue
            # Verificar si son adyacentes (Manhattan distance = 1)
            if (abs(x - ox) == 1 and y == oy) or (abs(y - oy) == 1 and x == ox):
                if location:
                    neighbors.append((ox, oy))
                else:
                    neighbors.append(other_state)
        states_neighbors[state] = neighbors
                    
    return states_neighbors


def transition_matrix(states):
    """
    Calcula la matriz de transición basada en el modelo de movimiento del robot.
    El robot se mueve a cualquier celda vecina con igual probabilidad.

    Parameters:
    - states (dict): Diccionario de estados del sistema.

    Returns:
    - t_matrix (numpy.ndarray): Matriz de transición 42x42.
    """
    n_states = len(states)
    t_matrix = np.zeros((n_states, n_states))
    
    states_neighbors = neighbors_states(states, location=False)
    
    for state, state_neighbors in states_neighbors.items():
        if len(state_neighbors) == 0:
            # Si no hay vecinos, permanece en el mismo estado
            t_matrix[state, state] = 1.0
        else:
            # Probabilidad uniforme de moverse a cualquier vecino
            prob = 1.0 / len(state_neighbors)
            for neighbor in state_neighbors:
                t_matrix[state, neighbor] = prob
                    
    return t_matrix


# ============================================================================
# MODELO DEL SENSOR
# ============================================================================

def expected_measurements(states):
    """
    Calcula las mediciones esperadas para cada estado basándose en sus vecinos.
    Formato: [arriba, abajo, izquierda, derecha]
    1 = espacio libre (sin obstáculo), 0 = obstáculo

    Parameters:
    - states (dict): Diccionario de estados del sistema.

    Returns:
    - measurements (dict): Diccionario con mediciones esperadas para cada estado.
    """
    measurements = {}
    states_neighbors = neighbors_states(states, location=True)
    
    for state, (x, y) in states.items():
        # [arriba, abajo, izquierda, derecha]
        m = [0, 0, 0, 0]
        
        directions = {
            "up":    (x, y - 1),
            "down":  (x, y + 1),
            "left":  (x - 1, y),
            "right": (x + 1, y)
        }
        
        for i, dir_ in enumerate(["up", "down", "left", "right"]):
            if directions[dir_] in states_neighbors[state]:
                m[i] = 1
        
        measurements[state] = tuple(m)
    
    return measurements


def compute_discrepancy(measure_1, measure_2):
    """
    Calcula la discrepancia entre dos mediciones binarias.
    La discrepancia es el número de bits diferentes.

    Parameters:
    - measure_1 (str): Representación binaria de la primera medición.
    - measure_2 (str): Representación binaria de la segunda medición.

    Returns:
    - discrepancy (int): Número de bits diferentes.
    """
    discrepancy = 0
    
    for i in range(4):
        if measure_1[i] != measure_2[i]:
            discrepancy += 1
    
    return discrepancy


def observation_matrix(epsilon, measure, measurements):
    """
    Calcula la matriz de observación para una medición específica.
    P(E_k = e_k | X_k = s_i) = (1-ε)^(4-d) * ε^d

    Parameters:
    - epsilon (float): Probabilidad de error en cada sensor.
    - measure (str): Medición binaria actual (formato: '0101').
    - measurements (dict): Mediciones esperadas para cada estado.

    Returns:
    - o_matrix (numpy.ndarray): Matriz de observación diagonal 42x42.
    """
    o_matrix = np.zeros((42, 42))

    for state, state_measure in measurements.items():
        state_measure_str = ''.join(map(str, state_measure))
        d = compute_discrepancy(measure, state_measure_str)
        o_matrix[state, state] = ((1 - epsilon) ** (4 - d)) * (epsilon ** d)

    return o_matrix


def observation_matrices(epsilon, states):
    """
    Genera todas las matrices de observación posibles (16 en total).
    Formato de medición: ESWN (Este, Sur, Oeste, Norte)
    pero implementado como: [arriba, abajo, izquierda, derecha]

    Parameters:
    - epsilon (float): Probabilidad de error del sensor.
    - states (dict): Diccionario de estados del sistema.

    Returns:
    - o_matrices (dict): Diccionario con las 16 matrices de observación.
    """
    o_matrices = {}
    measurements = expected_measurements(states)

    # Generar las 16 posibles mediciones binarias
    for i in range(16):
        measure = bin(i)[2:].zfill(4)  # Convierte a binario de 4 dígitos
        o_matrices[measure] = observation_matrix(epsilon, measure, measurements)

    return o_matrices


# ============================================================================
# SIMULACIÓN DEL ROBOT
# ============================================================================

def robot_measurements(state, states, epsilon=0.1):
    """
    Simula las mediciones del robot con errores aleatorios.
    Cada sensor tiene probabilidad ε de fallar independientemente.

    Parameters:
    - state (int): Estado actual del robot.
    - states (dict): Diccionario de estados del sistema.
    - epsilon (float): Probabilidad de error del sensor.

    Returns:
    - measured (str): Medición binaria simulada.
    """
    measured = ''
    measurements = expected_measurements(states)
    expected = measurements[state]

    for bit in expected:
        if random.random() < epsilon:
            # Error: invertir el bit
            measured += '0' if bit == 1 else '1'
        else:
            # Medición correcta
            measured += str(bit)

    return measured


def random_robot_walk(states, steps, epsilon=0.1):
    """
    Simula una caminata aleatoria del robot con mediciones en cada paso.

    Parameters:
    - states (dict): Diccionario de estados del sistema.
    - steps (int): Número de pasos de la caminata.
    - epsilon (float): Probabilidad de error del sensor.

    Returns:
    - path (list): Lista de posiciones visitadas.
    - evidence (list): Lista de mediciones obtenidas.
    """
    path = []
    evidence = []

    state_neighbors = neighbors_states(states, location=False)
    current_state = random.choice(list(states.keys()))

    for _ in range(steps):
        path.append(states[current_state])
        measured = robot_measurements(current_state, states, epsilon)
        evidence.append(measured)
        
        neighbors = state_neighbors[current_state]
        if neighbors:
            current_state = random.choice(neighbors)

    return path, evidence


# ============================================================================
# ALGORITMOS DE INFERENCIA
# ============================================================================

def filtering(E, O, T, X0):
    """
    Algoritmo de filtrado (Forward Algorithm).
    Calcula P(X_k | E_1:k = e_1:k)

    Parameters:
    - E (list): Secuencia de mediciones.
    - O (dict): Diccionario de matrices de observación.
    - T (numpy.ndarray): Matriz de transición.
    - X0 (numpy.ndarray): Distribución inicial.

    Returns:
    - forward (numpy.ndarray): Distribución de probabilidad sobre estados.
    """
    forward = X0
    
    for evidence in E:
        forward = O[evidence] @ T.T @ forward
        forward = forward / forward.sum()  # Normalización
        
    return forward


def smoothing(E, O, T, X0, k):
    """
    Algoritmo de suavizado (Forward-Backward Algorithm).
    Calcula P(X_k | E_1:n = e_1:n) donde k < n

    Parameters:
    - E (list): Secuencia completa de mediciones.
    - O (dict): Diccionario de matrices de observación.
    - T (numpy.ndarray): Matriz de transición.
    - X0 (numpy.ndarray): Distribución inicial.
    - k (int): Paso de tiempo para el cual calcular la distribución.

    Returns:
    - smooth (numpy.ndarray): Distribución suavizada sobre estados.
    """
    # Forward pass hasta el tiempo k
    forward = filtering(E[:k], O, T, X0)
    
    # Backward pass desde el final hasta k
    backward = np.ones((T.shape[0], 1))
    
    for evidence in E[-1:k-1:-1]:
        backward = T @ O[evidence] @ backward
    
    # Combinar forward y backward
    smooth = forward * backward
    smooth = smooth / smooth.sum()  # Normalización
    
    return smooth


def most_likely_sequence(E, O, S, T, X0):
    """
    Algoritmo de Viterbi para encontrar la secuencia más probable de estados.
    Calcula argmax_{x_1:k} P(x_1:k | E_1:k = e_1:k)

    Parameters:
    - E (list): Secuencia de mediciones.
    - O (dict): Diccionario de matrices de observación.
    - S (dict): Diccionario de estados.
    - T (numpy.ndarray): Matriz de transición.
    - X0 (numpy.ndarray): Distribución inicial.

    Returns:
    - best_sequence (list): Secuencia más probable de ubicaciones.
    """
    n_states = T.shape[0]
    n_steps = len(E)
    
    sequences = np.zeros((n_states, n_steps))
    states = np.zeros((n_states, n_steps))
    ones = np.ones((n_states, n_states))
    
    # Inicialización
    sequences[:, 0] = (O[E[0]] @ X0).reshape((n_states,))
    message = sequences[:, 0].reshape((n_states, 1))
    
    # Recursión hacia adelante
    for i, evidence in enumerate(E[1:]):
        message = (T @ O[evidence]) * (message * ones)
        states[:, i+1] = np.argmax(message, axis=0).reshape((n_states,))
        message = np.max(message, axis=0).reshape((n_states, 1))
        sequences[:, i+1] = message.reshape((n_states,))
    
    # Backtracking para encontrar la secuencia óptima
    states = states.astype('int32')
    s = np.argmax(sequences[:, -1], axis=0)
    best_sequence = [S[s]]
    
    for i in range(n_steps-1, 0, -1):
        s = states[s, i]
        best_sequence.append(S[s])
    
    best_sequence = best_sequence[::-1]
    
    return best_sequence


# ============================================================================
# VISUALIZACIÓN
# ============================================================================

def view_heatmap(state, states, title="Robot Localization Heatmap"):
    """
    Crea y muestra un mapa de calor basado en la distribución de probabilidad.

    Parameters:
    - state (numpy.ndarray): Vector de 42 componentes con probabilidades.
    - states (dict): Diccionario de estados del sistema.
    - title (str): Título del gráfico.
    """
    # Crear una matriz 4x16 para el mapa de calor
    heatmap_data = np.zeros((4, 16))
    
    # Llenar la matriz con las probabilidades
    for state_idx, (x, y) in states.items():
        heatmap_data[y, x] = state[state_idx, 0]
    
    # Crear el mapa de calor
    plt.figure(figsize=(14, 4))
    sns.heatmap(heatmap_data, annot=True, fmt='.4f', cmap='YlOrRd', 
                cbar_kws={'label': 'Probability'}, linewidths=0.5)
    plt.title(title)
    plt.xlabel('X Position')
    plt.ylabel('Y Position')
    plt.tight_layout()
    plt.show()


# ============================================================================
# PROGRAMA PRINCIPAL
# ============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("HIDDEN MARKOV MODEL - ROBOT LOCALIZATION")
    print("=" * 70)
    
    # Configuración
    epsilon = 0.1  # Error rate del sensor
    n_steps = 20   # Número de pasos para la simulación
    
    # Calcular matriz de transición
    print("\n[1/6] Calculando matriz de transición...")
    T = transition_matrix(S)
    print(f"    Matriz de transición: {T.shape}")
    
    # Calcular matrices de observación
    print("\n[2/6] Calculando matrices de observación...")
    O = observation_matrices(epsilon, S)
    print(f"    Matrices de observación: {len(O)} matrices")
    
    # Distribución inicial (uniforme)
    X0 = np.ones((42, 1)) / 42
    print(f"\n[3/6] Distribución inicial uniforme: P(X_0) = 1/42")
    
    # Simular caminata aleatoria del robot
    print(f"\n[4/6] Simulando caminata aleatoria del robot ({n_steps} pasos)...")
    path, E = random_robot_walk(S, n_steps, epsilon)
    print(f"    Path real: {path[:5]}... (mostrando primeros 5)")
    print(f"    Evidencia: {E[:5]}... (mostrando primeros 5)")
    
    # FILTERING
    print("\n[5/6] Ejecutando FILTERING...")
    filtered = filtering(E, O, T, X0)
    print(f"    Estado más probable (filtering): {S[np.argmax(filtered)]}")
    print(f"    Estado real final: {path[-1]}")
    view_heatmap(filtered, S, f"Filtering - Después de {n_steps} pasos")
    
    # SMOOTHING
    print("\n[6/6] Ejecutando SMOOTHING...")
    k = n_steps // 2  # Punto medio
    smoothed = smoothing(E, O, T, X0, k)
    print(f"    Estado más probable en k={k} (smoothing): {S[np.argmax(smoothed)]}")
    print(f"    Estado real en k={k}: {path[k-1]}")
    view_heatmap(smoothed, S, f"Smoothing - Estado en paso {k}")
    
    # VITERBI (Most Likely Sequence)
    print("\n[7/7] Ejecutando VITERBI (secuencia más probable)...")
    sequence = most_likely_sequence(E, O, S, T, X0)
    
    # Crear vector para visualización
    vector = np.zeros((42, 1))
    for loc in sequence:
        for state_idx, state_loc in S.items():
            if state_loc == loc:
                vector[state_idx, 0] = 1.0
                break
    
    print(f"    Secuencia predicha: {sequence[:5]}... (mostrando primeros 5)")
    print(f"    Secuencia real:     {path[:5]}... (mostrando primeros 5)")
    
    # Calcular exactitud
    accuracy = sum(1 for i in range(len(path)) if sequence[i] == path[i]) / len(path)
    print(f"    Exactitud: {accuracy*100:.2f}%")
    
    view_heatmap(vector, S, f"Viterbi - Secuencia más probable ({n_steps} pasos)")
    
    print("\n" + "=" * 70)
    print("ANÁLISIS COMPLETADO")
    print("=" * 70)