import time
import time
import random
from openpyxl import Workbook
from .constants import *

# init
Sc = 0
Scb = 0
Sb = 0
edge_num = 0
ver_num = 0
iter_count = 0  # Iteration count
pro = 50  # Probability of selecting neighbors

def random_int(n):      
    """Randomly generate an integer from 0 to (n-1)."""
    new_var = np.random.randint
    return new_var(n)   

def read_initial(in_file):
    """Read the initial graph data from the given input file."""
    global ver_num, edge_num, edge, adj, color, p
    global adjlen, tabu, v2color, current_best_color, best_color
    global initial_color, last_color, v2idx, conect, colorlen, bian

    with open(in_file, 'r') as f:
        ver_num, edge_num = map(int, f.readline().strip().split()) #number of vertecies and number of edges
        edge = np.zeros((ver_num, ver_num), dtype=int)  # Edge connection matrix
        adj = [[] for _ in range(ver_num)]  # Adjacency list
        color = np.zeros((2, ver_num), dtype=int)  # Color assignment matrix
        p = np.zeros((ver_num, 2), dtype=float)  # Restart probabilities
        adjlen = np.zeros(ver_num, dtype=int)  # Length of adjacency list
        tabu = np.zeros(ver_num, dtype=int)  # Tabu list
        v2color = np.zeros(ver_num, dtype=int)  # Current coloring of vertices
        current_best_color = np.zeros(ver_num, dtype=int)  # Current best coloring
        best_color = np.zeros(ver_num, dtype=int)  # Global best coloring
        initial_color = np.zeros(ver_num, dtype=int)
        last_color = np.zeros(ver_num, dtype=int)
        v2idx = np.zeros(ver_num, dtype=int)  # Vertex indices
        conect = np.zeros(ver_num, dtype=int)  # Edges in current color
        colorlen = np.zeros(2, dtype=int)  # Color length count
        bian = np.zeros(2, dtype=int)  # Edge counts between colors

        for _ in range(edge_num):
            a, b = map(int, f.readline().strip().split())
            a -= 1  # Convert to zero-based index
            b -= 1  # Convert to zero-based index
            edge[a][b] = edge[b][a] = 1  # Mark the edge connection

        # Update adjacency list based on edge connections

        edge_num=0
        for i in range(ver_num - 1):
            for j in range(i + 1, ver_num):
                if edge[i][j]:  #when there is a edge between i and j (where is no 0)
                    edge_num+=1
                    adj[i].append(j)
                    adj[j].append(i)
                    adjlen[i] += 1
                    adjlen[j] += 1


def random_initial():
    """Construct an initial solution by randomly assigning one of two colors to each vertex."""
    global Sc, Scb, Sb, colorlen, bian, tabu, conect, p, v2color, initial_color
    global current_best_color, best_color, color, v2idx, ver_num, edge

    # Initialize counters and structures
    colorlen[0] = colorlen[1] = 0
    bian[0] = bian[1] = 0

    for i in range(ver_num):
        tabu[i] = 0
        conect[i] = 0
        p[i][0] = p[i][1] = 0.5  # Reset probabilities

    # Assign random colors to each vertex
    for i in range(ver_num):
        k = np.random.randint(2)  # Random color
        color[k][colorlen[k]] = i
        v2color[i] = k
        initial_color[i] = k
        current_best_color[i] = k
        best_color[i] = k
        v2idx[i] = colorlen[k]
        colorlen[k] += 1

    # Calculate connections within the same color
    for i in range(ver_num):
        for j in range(adjlen[i]):  # Iterate over neighbors
            neighbor = adj[i][j]
            if v2color[i] == v2color[neighbor]:
                conect[i] += 1

    # Update the number of same-color edges (bian)
    for i in range(ver_num):
        bian[v2color[i]] += conect[i]

    bian[0] //= 2
    bian[1] //= 2

    # Calculate Sc, Scb, and Sb
    Sc = min(bian[0], bian[1])
    Scb = Sc
    Sb = Sc


def one_flip(n):
    """..."""
    global Sc, bian, color, v2color, v2idx, colorlen, conect, adj, adjlen

    # Current color and index of vertex n
    org_color = v2color[n]
    new_color = 1 - org_color
    org_idx = v2idx[n]

    print(f"FLIP: Vertex {n}, org_color={org_color}, new_color={new_color}")

    # Step 1: Move vertex n to the new color group
    color[new_color][colorlen[new_color]] = n
    v2color[n] = new_color
    v2idx[n] = colorlen[new_color]
    colorlen[new_color] += 1

    # Step 2: Remove vertex n from the original color group
    if org_idx != colorlen[org_color] - 1:
        last_vertex = color[org_color][colorlen[org_color] - 1]
        color[org_color][org_idx] = last_vertex
        v2idx[last_vertex] = org_idx
    colorlen[org_color] -= 1

    # Step 3: Update `bian` and `conect` for vertex n
    print(f"Before Update: bian={bian}, conect[n]={conect[n]}")
    bian[org_color] -= conect[n]
    conect[n] = adjlen[n] - conect[n]
    bian[new_color] += conect[n]
    print(f"After Update: bian={bian}, conect[n]={conect[n]}")

    # Step 4: Update the connection counts of n's neighbors
    for i in range(adjlen[n]):
        neighbor = adj[n][i]
        if v2color[neighbor] == org_color:
            conect[neighbor] -= 1
        else:
            conect[neighbor] += 1

    # Step 5: Update the Sc value
    old_Sc = Sc
    Sc = min(bian[0], bian[1])
    print(f"Updated Sc: {old_Sc} -> {Sc} (bian={bian})")
    

def pertubation():
    """applying a small random perturbation to the current solution
    pertubation_depth = 0.01 (controls how deep the perturbation goes)"""
    global tabu, Sc
    k = max(int(pertubation_depth * ver_num), 1)  # Calculate number of perturbations based on depth (at least one)
    #print(f"iter: {iter} prima，sc: {Sc}")
    # Perform k random flips
    for i in range(k):
        j = random_int(ver_num)  # Randomly select a vertex to move
        one_flip(j)  # Flip the color of the selected vertex
        tabu[j] = 0  # Reset the tabu status of the flipped vertex
    #print(f"iter: {iter} dopo，sc: {Sc}")

#these two #line are to log the value of Sc before and after the perturbation process.

def updateP():
    """adjusts probabilities based on the algorithm's reinforcement learning mechanism
    mechanism depends on whether the current coloring matches the initial one,
    rewarding consistency or penalizing changes"""
    global p, initial_color, current_best_color, last_color, alpha, beta, gamma1

    len_0_0 = 0    #vertices initially had color 0 and still have color 0 in the best solution
    len_0_1 = 0

    # Copy currentBestColor to lastColor
    for i in range(ver_num):
        last_color[i] = current_best_color[i]

    # Fix symmetry by comparing initial and last colors (analyzes how many vertices have remained the same or changed colors between the initial coloring )
    for i in range(ver_num):
        if initial_color[i] == 0 and last_color[i] == 0:
            len_0_0 += 1
        elif initial_color[i] == 0 and last_color[i] == 1:
            len_0_1 += 1

    # If len_0_1 (initial 0 to last 1) is greater than len_0_0 (initial 0 to last 0), flip all colors
    if len_0_1 > len_0_0:
        for i in range(ver_num):
            last_color[i] = 1 if last_color[i] == 0 else 0

    # Compare initialColor and lastColor and update probabilities in p
    for i in range(ver_num):
        if last_color[i] == initial_color[i]:  # Direct reinforcement
            p[i][initial_color[i]] = alpha + ((1 - alpha) * p[i][initial_color[i]])
            p[i][1 - last_color[i]] = (1 - alpha) * p[i][1 - last_color[i]]
        else:  # Penalize the old color, reward the new one
            p[i][last_color[i]] = gamma1 + ((1 - gamma1) * beta) + ((1 - gamma1) * (1 - beta) * p[i][last_color[i]])
            p[i][initial_color[i]] = (1 - gamma1) * (1 - beta) * p[i][initial_color[i]]

    # Uncomment this section to print the updated probabilities for debugging purposes
    # for i in range(1, verNum):
    #     print(i, p[i][0], p[i][1])


def restart():
    """Reinitializes variables.Resets the color assignment of vertices based
    on updated probabilities using a technique called roulette wheel selection"""
    global colorlen, bian, tabu, conect, v2color, initial_color, v2idx, Sc, Scb, edge

    # Reset the lengths of the two color sets and bian counters
    colorlen[0] = 0
    colorlen[1] = 0
    bian[0] = 0
    bian[1] = 0

    # Reset tabu and connection counts for each vertex
    for i in range(ver_num):
        tabu[i] = 0
        conect[i] = 0

    # This part of coloring needs improvement
    updateP()       #probabilities that will be used in the roulette wheel selection

    # roulette wheel selection
    for i in range(ver_num):
        l = int(p[i][0] / (p[i][0] + p[i][1]) * 100)  # Probability of selecting color 0 (converted to a percentage)
        if random_int(101) < l:
            k = 0               #assigne color 0
        elif random_int(101) == l:
            k = random_int(2)  # Equal probability for either color
        else:
            k = 1              #assigne color 1

        # Assign vertex i to color k
        color[k][colorlen[k]] = i
        v2color[i] = k  # Assign the vertex to color k
        initial_color[i] = k  # Store initial color
        v2idx[i] = colorlen[k]  # Track the index of the vertex in this color
        colorlen[k] += 1  # Increment the number of vertices assigned to this color

    # Count the number of connections between vertices of the same color
    for i in range(ver_num):
        for j in range(ver_num):
            if edge[i][j]:  # If there is an edge between vertex i and vertex j
                if v2color[i] == v2color[j]:  # If they share the same color
                    conect[i] += 1  # Increase the conflict count for vertex i

    # Calculate the conflict score for each color set
    for i in range(ver_num):
        k = v2color[i]
        bian[k] += conect[i]  # Accumulate connection conflicts

    bian[0] //= 2  # Divide by 2 to avoid double counting connections
    bian[1] //= 2
    Sc = min(bian[0], bian[1])  # Update Sc with the minimum conflict score
    Scb = Sc  # Store the best score so far


def judge_best(t):
    """tracking the best solution found so far"""
    global Sc, Scb, Sb, no_improve, current_best_color, best_color, bestTime, pertubation_time, pro, startTime, endTime


    print(f"Debug: Sc = {Sc}, Scb = {Scb}, Sb = {Sb}")

    # If the current solution is better than the best solution so far (Sc > Scb)
    if Sc > Scb:
        Scb = Sc  # Update the best solution found in the current search
        no_improve = 0  # Reset the counter for no improvements
        for i in range(ver_num):  #Save the current coloring as the best found so far in this iteration
            current_best_color[i] = v2color[i]

        # If the current solution is better than the global best solution (Sc > Sb)
        if Sc > Sb:
            endTime = time.time()  # Record the end time
            bestTime = endTime - t # Calculate elapsed time
            Sb = Sc  # Update the global best score
            for i in range(ver_num):     # Save the current coloring as the global best solution
                best_color[i] = v2color[i]

            # Optionally print information
            # print(f"iter: {iter} Sc: {Sc} time: {bestTime}s")
    else:
        no_improve += 1  # Increase the no-improvement counter
        if no_improve > pertubation_threshold:      # no improvement made after a threshold number (pertubation_threshold = 500) of iterations
            pro = 50  # Reset the probability of neighborhood selection
            no_improve = 0  # Reset the no-improvement counter
            pertubation_time += 1  # Increase the perturbation count
            if pertubation_time < restart_threshold:   #number of perturbations is less than the restart threshold
                pertubation()  # Apply a small perturbation
            else:
                # Restart the search if the restart threshold is reached
                pertubation_time = 0  # Reset the perturbation count
                restart()  # Perform a full restart of the algorithm


def local_search(t):
    """Optimizing the current solution by either flipping the color of
    a vertex or swapping the colors of two vertices to improve
    the balance of edge counts between the two color groups."""
    global pro, Sc, Sb

    temp_delta = swap_delta = temp_delta_2 = swap_delta_2 = -MAX_VAL
    mark = swap_0 = swap_1 = -1

    if np.random.randint(0, 99) < pro:  # FLIP operation
        for i in range(ver_num):
            if tabu[i] > iter_count:  # Skip vertices in the tabu list
                continue

            k = bian[1 - v2color[i]] + adjlen[i] - conect[i]  # Gain from the new color
            m = bian[v2color[i]] - conect[i]  # Loss from the old color
            delta = min(k, m) - Sc  # Calculate improvement (delta)

            print(f"Vertex {i}: k={k}, m={m}, delta={delta}, tabu={tabu[i]}")

            if delta > temp_delta:
                temp_delta = delta
                mark = i
            elif delta == temp_delta: #use a secondary criterion (The idea here is to potentially favor a move that has larger overall change even if the initial delta value is the same)
                if k + m > temp_delta_2:
                    temp_delta_2 = k + m
                    mark = i

        if mark == -1:
            print("Warning: No valid vertex found for FLIP (mark == -1). Skipping FLIP.")
        else:
            print(f"Performing FLIP on vertex {mark}. Delta: {temp_delta}.")
            tabu[mark] = iter_count + tt  # Add the vertex to the tabu list
            one_flip(mark)  # Perform the flip
            if Sc > Sb:
                print(f"New best score found with FLIP. Sc: {Sc}, Sb: {Sb}.")
                pro += 1  # Increase probability threshold if solution improves

    else:  # SWAP operation
        for i in range(colorlen[0]):  # Iterate over vertices in color group 0
            org_0 = color[0][i]
            if tabu[org_0] > iter_count:
                continue
            for j in range(colorlen[1]):  # Iterate over vertices in color group 1
                org_1 = color[1][j]
                if tabu[org_1] > iter_count:
                    continue

                k = bian[1] + adjlen[org_0] - conect[org_0] - conect[org_1] - edge[org_0][org_1]  # Balance for color 1
                m = bian[0] + adjlen[org_1] - conect[org_1] - conect[org_0] - edge[org_1][org_0]  # Balance for color 0
                delta = min(k, m) - Sc

                if delta > swap_delta:
                    swap_delta = delta
                    swap_0 = org_0
                    swap_1 = org_1
                elif delta == swap_delta:
                    if k + m > swap_delta_2:
                        swap_delta_2 = k + m
                        swap_0 = org_0
                        swap_1 = org_1

        if swap_0 == -1 or swap_1 == -1:
            print("Warning: No valid vertices found for SWAP (swap_0 == -1 or swap_1 == -1). Skipping SWAP.")
        else:
            print(f"Performing SWAP between vertices {swap_0} and {swap_1}. Delta: {swap_delta}.")
            # Perform the SWAP move (flipping both vertices)
            tabu[swap_0] = iter_count + tt
            tabu[swap_1] = iter_count + tt
            one_flip(swap_0)
            one_flip(swap_1)

        if Sc > Sb:
            print(f"New best score found with SWAP. Sc: {Sc}, Sb: {Sb}.")
            pro -= 1  # Decrease probability threshold if solution improves

    # Check if this is the best solution so far
    judge_best(t)
    print(f"Judge Best: Sc: {Sc}, Sb: {Sb}.")