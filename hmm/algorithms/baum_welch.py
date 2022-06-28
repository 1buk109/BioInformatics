import numpy as np
from .backward import BackwardScaled

class BaumWelch(BackwardScaled):
    def __init__(self, input_seq, output, init_trainsition, transition):
        super().__init__(input_seq, output, init_trainsition, transition)
        self.ll_update_diff = 0
        self.loglikelihood = 0
    
    def calculate_transition_expectation(self):
        self.expected_transition = np.zeros(self.transition.shape)
        for prev in range(self.K):
            for next in range(self.K):
                for l in range(self.L - 1):
                    output_index = self.input_seq[l + 1] - 1
                    self.expected_transition[prev, next] += (1 / self.scaling_params[l + 1]) * self.forward_vars[prev, l] * self.transition[prev, next] * self.output[next, output_index] * self.backward_vars[next, l + 1]

    def calculate_output_expectation(self):
        self.expected_output  = np.zeros(self.output.shape)
        fb = self.forward_vars * self.backward_vars
        for l in range(self.L):
            output_index = self.input_seq[l] - 1
            self.expected_output[:, output_index] += fb[:, l]

    def update_parameters(self):
        self.transition = self.expected_transition / self.expected_transition.sum(axis=1).reshape(-1, 1)
        self.output = self.expected_output / self.expected_output.sum(axis=1).reshape(-1, 1)
    
    def calculate_loglikelihood(self):
        ll = np.sum(np.log(self.scaling_params))
        self.ll_update_diff = abs(self.loglikelihood - ll)
        self.loglikelihood = ll


    def run_baum_welch(self, n: int = 10000, threshold: float = 1e-8):        
        for _ in range(n):            
            self.run_forward_scaled()
            self.run_backward_scaled()
            self.calculate_transition_expectation()
            self.calculate_output_expectation()
            self.update_parameters()
            self.calculate_loglikelihood()
            if (self.ll_update_diff < threshold):                
                break