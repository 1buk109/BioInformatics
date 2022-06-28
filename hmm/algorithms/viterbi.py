import numpy as np

class Viterbi:
    def __init__(self, input_seq, output, init_trainsition, transition):
        self.const = 1e-300
        
        self.K                  = len(transition)
        self.L                  = len(input_seq)
        self.path               = np.full((self.K, self.L), self.K)
        self.input_seq          = input_seq
        self.hidden_seq         = np.array([-1] * self.L)
        self.output             = self.logarithmize(output)
        self.viterbi            = np.zeros((self.K, self.L))
        self.init_transition    = self.logarithmize(init_trainsition)
        self.transition         = self.logarithmize(transition)

    def logarithmize(self, list):
        return np.log(np.array(list) + self.const)

    def inverse_logarithmization(self, x):
        return np.exp(x) - self.const

    def initialize(self):
        initial_output_index = self.input_seq[0] - 1
        for k in range(self.K):
            self.viterbi[k, 0] = self.init_transition[k, 0] + self.output[k, initial_output_index]

    def calculate_DP(self):
        for l in range(1, self.L):
            output_index = self.input_seq[l] - 1
            for next in range(self.K):
                options = []
                for prev in range(self.K):
                    options.append(self.output[next, output_index] + self.viterbi[prev, l-1] + self.transition[prev, next])
                self.viterbi[next, l] = max(options)
                self.path[next, l] = np.argmax(options)

    def traceback(self):
        hidden_state = np.argmax(self.viterbi[:, -1])
        self.hidden_seq[-1] = hidden_state
        for l in range(1, self.L)[::-1]:
            hidden_state = self.path[hidden_state, l]
            self.hidden_seq[l-1] = hidden_state

    def calculate_joint_prob(self):
        return self.inverse_logarithmization(max(self.viterbi[:, -1]))

    def run_viterbi_algorithm(self):
        self.initialize()
        self.calculate_DP()
        self.traceback()