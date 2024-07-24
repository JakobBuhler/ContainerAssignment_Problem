import numpy as np
from typing import Any, Dict, List, Literal, Optional, Tuple
from pydantic import Field


class ContainerAssignment2Route():
    """
    # Container Assignment Problem

    Description
    -----------
    The Container Assignment Problem involves assigning freight containers to transportation modes to minimize
    costs while meeting delivery deadlines. This problem is critical in logistics for optimizing the use of multimodal
    transport networks, which include various transportation modes like trucks and barges. Efficient container
    assignment can lead to significant cost savings and improved service levels, especially in dynamic environments
    where real-time decision-making is essential.
    The Container Assignment Problem tries to find the best mode of transportation for each container, in this case
    the truck or barge mode while recognizing capacity constraints for each transportation route.
    The Problem creates a QUBO-Matrix with the size
    N+M*K x N+M*K
    with N containers, M tracks of transportation, and K amount of Slack Variables for each track. This produces a
    solution vector where my decision variables represent if the container was transported by truck x = 1 or by
    barge mode x = 0.

    Links
    -----

    [Transformation](https://arxiv.org/pdf/2007.01730)

    Attributes
    ----------

    ##Example Parameters with 3 containers, 2 tracks, and each track has capacity 2

    N = 3               Amount of Containers
    M = 2               Amount of tracks
    cap = [2,2]         Capacity of each track
    c_b = [3,2,4]       Cost to ship each container in barge mode
    c_t = [13,11,15]    Cost to ship each container in truck mode
    K = 2               Amount of slack variables
    P =  max(c_b) + 1   Penalty value for capacity constraint

    routes = {}         Routes each describing one container route.  = 1 if route i uses track j
    routes[0] = [0,1]   #Example:  routes[0][1]= 1 because Container 0 uses track 1
    routes[1] = [1,0]
    routes[2] = [1,1]

    With the above parameters we can now decode our QUBO containing:

    1. Objective function to minimize costs

    2. Capacity constraint ensuring all containers are being transported properly

    Note: If gen_problem_own_params wants to be used for a specific problem, additional parameters such
    as track_capacity, barge_mode_cost, truck_mode_cost, and
    routes have to be supplied.
    """

    name: Literal["CA"] = "CA"
    N: int = Field(default=0)
    M: int = Field(default=0)
    K: int = Field(default=0)
    c_b: list = Field(default_factory=list)
    c_t: list = Field(default_factory=list)
    cap: list = Field(default_factory=list)
    routes: dict = Field(default_factory=dict)
    P: int = Field(default=0)
    vars: int = Field(default=0)


    def gen_problem_random_params(self,N,M,cap_value):
        self.N = N
        self.M = M
        self.c_b = np.random.randint(1, 10, size=N).tolist()
        self.c_t = np.random.randint(14, 25, size=N).tolist()
        self.cap = [cap_value] * M
        self.K = int(np.log2(max(self.cap)) + 1)
        self.routes = {i: np.random.randint(0, 2, size=M).tolist() for i in range(N)}
        self.P =  max(self.c_b) +1
        self.vars = N+M*self.K

    def gen_problem_own_params(self,N,M, cap_value, c_b, c_t, routes):
        self.N = N
        self.M = M
        self.c_b = c_b
        self.c_t = c_t
        self.cap = [cap_value] * M
        self.K = int(np.log2(max(self.cap)) + 1)
        self.routes = routes
        self.P = max(self.c_b) + 1
        self.vars = N+M*self.K

    def gen_qubo_matrix(self):

        #Objective Function:

        Q1 = np.zeros((self.vars, self.vars), dtype=int)
        for i in range(self.N):
            idx = i
            Q1[idx, idx] += self.c_t[idx] - self.c_b[idx]

        #Capacity Constraint:
        Q2 = np.zeros((self.vars, self.vars), dtype=int)

        for j in range(self.M):

            for i in range(self.N):
                for l in range(self.N):
                    idx1 = i
                    idx2 = l
                    Q2[idx1, idx2] += self.routes[i][j] * self.routes[l][j]

            for k in range(self.K):
                for p in range(self.K):
                    idx3 = self.N + j * self.K + k
                    idx4 = self.N + j * self.K + p
                    Q2[idx3, idx4] += (2 ** k) * (2 ** p)

            for i in range(self.N):
                for l in range(self.N):
                    idx5 = l
                    Q2[idx5, idx5] += -2 * self.routes[i][j] * self.routes[l][j]

            for i in range(self.N):
                for k in range(self.K):
                    idx6 = self.N + j * self.K + k
                    Q2[idx6, idx6] += 2 * self.routes[i][j] * (2 ** k)

            for i in range(self.N):
                for k in range(self.K):
                    idx7 = i
                    idx8 = self.N + j * self.K + k
                    Q2[idx7, idx8] -=  self.routes[i][j] * (2 ** k)
                    Q2[idx8, idx7] -=  self.routes[i][j] * (2 ** k)

            for i in range(self.N):
                idx9 = i
                Q2[idx9, idx9] += 2 * self.cap[j] * self.routes[i][j]

            for k in range(self.K):
                idx10 = self.N + j * self.K + k
                Q2[idx10, idx10] += -2 * self.cap[j] * (2 ** k)

        Q = Q1 + self.P * Q2
        return Q

    def decode_solution(self, best_sample):
        truck_mode = []
        barge_mode = []
        for container, activation in best_sample.items():
            if container == self.N:
                break
            elif activation == 1:
                truck_mode.append(container)
            elif activation == 0:
                barge_mode.append(container)
        print(f"Containers transported by truck: {truck_mode}")
        print(f"Containers transported by barge mode: {barge_mode}")