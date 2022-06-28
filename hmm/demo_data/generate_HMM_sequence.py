import numpy as np

class GenerateHMMSequence:
    def __init__(self, sequence_length, init_transition, transition, output):
        self.init_transition = np.array(init_transition)
        self.transition = np.array(transition)
        self.output = np.array(output)
        self.sequence_length = sequence_length
        self.K = len(transition)

        self.hidden_sequence = np.array([-1] * self.sequence_length)
        self.observed_sequence = np.array([-1] * self.sequence_length)

    def generate_hidden_sequence(self):
        self.hidden_sequence[0] = np.random.choice(self.K, p=self.init_transition.reshape(-1,))
        for i in range(1, self.sequence_length):
            self.hidden_sequence[i] = np.random.choice(self.K, p=self.transition[self.hidden_sequence[i-1], :])

    def generate_observed_sequence(self):
        dice = range(1, self.output.shape[1] + 1)
        for i in range(self.sequence_length):
            self.observed_sequence[i] = np.random.choice(dice, p=self.output[self.hidden_sequence[i], :])

    def generate_sequences(self):
        self.generate_hidden_sequence()
        self.generate_observed_sequence()
        return self.hidden_sequence, self.observed_sequence