# Nature-Inspired-Computation-Problem-Solving-with-an-Evolutionary-Algorithm-Bank-Problem-

This project aims to develop an evolutionary algorithm system to maximize the amount of money that can be packed into a security van (0/1 Knapsack Problem), given a set of 100 bags of money with different denominations, weight, and value. 

Dataset:
The algorithm is tested on a dataset containing items with assigned weights and values. The goal is to select items whose total weight does not exceed a specified limit (285 kg), while maximizing the total value.

Implementation Details:
1. Generates initial population randomly
2. Uses binary tournament selection for parents
3. Applies single-point crossover and mutation
4. Implements weakest replacement strategy
5. Terminates after maximum fitness evaluations (10,000)
   
Key Findings:
1. Best results with small population size (2), tournament size (2), and mutation rate (1)
2. Removing mutation or crossover operations impairs solution quality
3. Suggests using Pareto tournament selection for multi-objective extension
   
Results:
The best solution found has a total value of 4353£ with a total weight of 284.0 kg, very close to the 285 kg limit.
