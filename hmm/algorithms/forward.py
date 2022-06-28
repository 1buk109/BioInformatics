import numpy as np

class ForwardScaled:
    def __init__(self, input_seq, output, init_trainsition, transition):
        self.const = 1e-300
        self.optimal_expectation = 0
      
        self.K                  = len(transition)
        self.L                  = len(input_seq)
        self.forward_vars       = np.zeros((self.K, self.L))
        self.scaling_params     = np.zeros((self.L,))
        self.input_seq          = input_seq
        self.output             = output
        self.init_transition    = init_trainsition
        self.transition         = transition

    def logarithmize(self, list):
        return np.log(np.array(list) + self.const)

    def forward_initialize(self):
        for k in range(self.K):
            initial_output_index = self.input_seq[0] - 1
            self.forward_vars[k, 0] = self.output[k, initial_output_index] * self.init_transition[k, 0]

        scale_param = sum(self.forward_vars[:, 0])
        self.scaling_params[0] = scale_param
        self.forward_vars[:, 0] /= scale_param

    def forward_calculate_DP(self):
        for l in range(1, self.L):
            output_index = self.input_seq[l] - 1
            for next in range(self.K):
                paths = []
                for prev in range(self.K):
                    paths.append(self.forward_vars[prev, l-1] * self.transition[prev, next])
                self.forward_vars[next, l] = self.output[next, output_index] * np.sum(paths)

            scale_param = np.sum(self.forward_vars[:, l])
            self.scaling_params[l] = scale_param
            self.forward_vars[:, l] /= scale_param

    def forward_terminate(self):
        self.optimal_expectation = np.prod(self.scaling_params)

    def run_forward_scaled(self):
        self.forward_initialize()
        self.forward_calculate_DP()
        self.forward_terminate()