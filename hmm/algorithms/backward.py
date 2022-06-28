import numpy as np
from .forward import ForwardScaled

class BackwardScaled(ForwardScaled):
    def __init__(self, input_seq, output, init_trainsition, transition):
        super().__init__(input_seq, output, init_trainsition, transition)
        self.backward_vars = np.zeros((self.K, self.L))
        self.run_forward_scaled()

    def backward_initialize(self):
        for k in range(self.K):
            self.backward_vars[k, -1] = 1

    def backward_calculate_DP(self):
        for l in range(self.L-1)[::-1]:
            output_index = self.input_seq[l+1] - 1
            for prev in range(self.K):
                paths = []
                for next in range(self.K):
                    paths.append(self.transition[prev, next] * self.output[next, output_index] * self.backward_vars[next, l+1])
                self.backward_vars[prev, l] = np.sum(paths) / self.scaling_params[l+1]

    def backward_terminate(self):
        options = []
        initial_output_index = self.input_seq[0] - 1
        for k in range(self.K):
            options.append(self.init_transition[k, 0] * self.output[k, initial_output_index] * self.backward_vars[k, 0])
        self.optimal_expectation = sum(options) * np.prod(self.scaling_params[1:])

    def run_backward_scaled(self):
        self.backward_initialize()
        self.backward_calculate_DP()
        self.backward_terminate()