import random

def main():
    base = ["A", "T", "G", "C"]
    sample_lengths = [2**l for l in range(3, 14)]
    random.seed(42)

    for length in sample_lengths:
        X = "".join(random.choices(base, k=length))
        Y = "".join(random.choices(base, k=length))
        print(f"{X} {Y}")


if __name__ == "__main__":
    main()
