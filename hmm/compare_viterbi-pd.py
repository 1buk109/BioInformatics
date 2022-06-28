import numpy as np

from demo_data.generate_HMM_sequence import GenerateHMMSequence
from algorithms.viterbi import Viterbi
from algorithms.posterior_decoding import PosteriorDecoding

def main():
    sequence_length = 10000
    rounds = 10

    output = np.array([
        [1/6, 1/6, 1/6, 1/6, 1/6, 1/6],
        [1/10, 1/10, 1/10, 1/10, 1/10, 1/2]
    ])
    init_trainsition = np.array([[0.5], [0.5]])
    transition = np.array([
        [0.95, 0.05],
        [0.1, 0.9]
    ])

    generator = GenerateHMMSequence(sequence_length, init_trainsition, transition, output)

    precision_viterbi = [0] * rounds
    precision_pd = [0] * rounds

    for round in range(rounds):
        hidden_sequence, observed_sequence = generator.generate_sequences()
        viterbi = Viterbi(observed_sequence, output, init_trainsition, transition)
        viterbi.run_viterbi_algorithm()
        pd = PosteriorDecoding(observed_sequence, output, init_trainsition, transition)
        pd.run_posterior_decoding()
        precision_viterbi[round] = sum(viterbi.hidden_seq == hidden_sequence) / sequence_length
        precision_pd[round] = sum(pd.hidden_sequence == hidden_sequence) / sequence_length
    print(f"veterbi\n{np.mean(precision_viterbi)}")
    print(f"posterior_decoding\n{np.mean(precision_pd)}")

if __name__ == '__main__':
    main()