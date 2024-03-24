from evaluation import evaluate_functional_correctness

HUMAN_EVAL = ''

if __name__ == '__main__':
    sample_file = HUMAN_EVAL + 'samples.jsonl'
    evaluate_functional_correctness(sample_file, k=[1])