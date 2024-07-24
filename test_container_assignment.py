import unittest
from container_assignment_2route import ContainerAssignment2Route


class TestContainerAssignment(unittest.TestCase):

    def test_gen_qubo_matrix(self):
        """
        Test whether a correct QUBO is generated.

        """
        # Params

        N = 3
        M = 2
        c_b = [4, 1, 7]
        c_t = [17, 24, 15]
        cap_value = 2
        routes = {0: [1, 0], 1: [0, 1], 2: [0, 0]}

        problem = ContainerAssignment2Route()
        problem.gen_problem_own_params(N, M, cap_value, c_b, c_t, routes)
        matrix = problem.gen_qubo_matrix()

        want = [
            [37, 0, 0, -8, -16, 0, 0],
            [0, 47, 0, 0, 0, -8, -16],
            [0, 0, 8, 0, 0, 0, 0],
            [-8, 0, 0, -8, 16, 0, 0],
            [-16, 0, 0, 16, 0, 0, 0],
            [0, -8, 0, 0, 0, -8, 16],
            [0, -16, 0, 0, 0, 16, 0],
        ]
        self.assertCountEqual(matrix.tolist(), want)
