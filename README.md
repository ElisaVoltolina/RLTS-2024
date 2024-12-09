# RLTS-2024
This repository contains the implementation of methods described in the article “Reinforcement learning based tabu search for the minimum load coloring Problem” by Zhe Sun a, Una Benlic b, Mingjie Li a, Qinghua Wuadeveloped, for the Mathematical Optimisation exam in December 2024. The focus is on solving the Minimum Load Coloring Problem (MLCP), which involves partitioning the vertex set V of a graph into two disjoint subsets, corresponding to vertices labeled as red and blue. The objective is to maximize the minimum number of edges whose endpoints share the same color.
Structure of the Repository: 
1.	Exact Solution: An integer programming formulation is implemented using Gurobi to solve the MLCP optimally. The related code can be found in the file “guribi-nb”
2.	Heuristic Approach: The article also proposes a reinforcement learning-based Tabu Search algorithm to approximately solve the MLCP. This implementation is organized into the following folders:
   - main: Contains the main function that runs the algorithm.
   - src: Includes all utility and constants
3.	Sensitivity analysis: in the folder named “sensitivity-analysis-nb”
The dataset used for experiments is from the publicly available DIMACS collection.
