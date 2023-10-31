import argparse
parser = argparse.ArgumentParser()

def decode(state):
    pos = 1 if state%2 == 0 else 2 
    state //= 2 
    b1 = (state // 256) + 1
    state = state % 256
    b2 = (state // 16) + 1
    r = state % 16 + 1
    string = f"{'0'*(2 - len(str(b1)))}{b1}{'0'*(2 - len(str(b2)))}{b2}{'0'*(2 - len(str(r)))}{r}{pos}"
    return string

if __name__ == "__main__":
    parser.add_argument("--opponent", type=str)
    parser.add_argument("--value-policy", type=str)
    args = parser.parse_args()

    try:
        state = 0
        with open(f"{args.value_policy}", 'r') as value_policy_file:
            for state, line in enumerate(value_policy_file.readlines()[:-2]):
                line = line.strip().split()
                print(decode(state), line[1], line[0]) 

    except IOError:
        print("Error: unable to read file")