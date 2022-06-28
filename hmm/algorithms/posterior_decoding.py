import numpy as np
from .backward import BackwardScaled

class PosteriorDecoding(BackwardScaled):
    def __init__(self, input_seq, output, init_trainsition, transition):
        super().__init__(input_seq, output, init_trainsition, transition)
        self.hidden_sequence = np.array([-1] * self.L)
        self.run_backward_scaled()

    def run_posterior_decoding(self):
        posterior_decoding_options = self.forward_vars * self.backward_vars
        for l in range(self.L):
            self.hidden_sequence[l] = np.argmax(posterior_decoding_options[:, l])