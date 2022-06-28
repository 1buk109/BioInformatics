import argparse
import numpy as np
from algorithms.baum_welch import BaumWelch
from algorithms.posterior_decoding import PosteriorDecoding
from algorithms.viterbi import Viterbi
from demo_data.generate_HMM_sequence import GenerateHMMSequence

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-l', '--sequence_length', type=int, required=True, help='The length of the observation sequence to be generated')
    return parser.parse_args()

def main(sequence_length):
    # validation_params
    output_val = np.array([
        [1/6, 1/6, 1/6, 1/6, 1/6, 1/6],
        [1/10, 1/10, 1/10, 1/10, 1/10, 1/2]
    ])
    init_trainsition = np.array([[0.5], [0.5]])
    transition_val = np.array([
        [0.95, 0.05],
        [0.1, 0.9]
    ])
    print('\n==validation parameters==')
    print('\n > output_probability')
    print(*output_val, sep='\n')
    print('\n > transition_probability')
    print(*transition_val, sep='\n')

    # initialize_params
    np.random.seed(7)
    output_init = np.random.random(output_val.shape)
    transition_init = np.random.random(transition_val.shape)

    # generate_sequences
    generator = GenerateHMMSequence(sequence_length, init_trainsition, transition_val, output_val)
    hidden_sequence, observed_sequence = generator.generate_sequences()

    # execute baum_welch
    em = BaumWelch(observed_sequence, output_init, init_trainsition, transition_init)
    em.run_baum_welch()
    print('\n==estimated parameters with baum_welch==')
    print('\n > output_probability')
    print(*em.output, sep='\n')
    print('\n > transition_probability')
    print(*em.transition, sep='\n')

    # compare_precision with viterbi and posterior_decoding
    viterbi = Viterbi(observed_sequence, em.output, init_trainsition, em.transition)
    viterbi.run_viterbi_algorithm()
    pd = PosteriorDecoding(observed_sequence, em.output, init_trainsition, em.transition)
    pd.run_posterior_decoding()
    precision_viterbi = sum(viterbi.hidden_seq == hidden_sequence) / sequence_length
    precision_pd = sum(pd.hidden_sequence == hidden_sequence) / sequence_length
    print('\n==precision of estimated sequences with estimated parameters==')
    print(f" > veterbi\n    {np.mean(precision_viterbi)}")
    print(f" > posterior_decoding\n    {np.mean(precision_pd)}")

if __name__ == '__main__':
    args = parse_args()
    main(args.sequence_length)