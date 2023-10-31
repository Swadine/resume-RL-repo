import sys,subprocess,os
import numpy as np
import matplotlib.pyplot as plt

def check(output, start):
    output = output.split('\n')
    output.remove('')
    # print(output)

    for out in output:
        line = out.split()
        # print(line)
        if line[0] == start:
            return float(line[2])

def run(opponent, p, q, start):
    cmd_encoder = "python","encoder.py","--opponent", opponent, "--p", str(p), "--q", str(q)
    print("\n","Generating the MDP encoding using encoder.py")
    f = open('verify_attt_mdp','w')
    subprocess.call(cmd_encoder,stdout=f)
    f.close()

    cmd_planner = "python","planner.py","--mdp","verify_attt_mdp"
    print("\n","Generating the value policy file using planner.py using default algorithm")
    f = open('verify_attt_planner','w')
    subprocess.call(cmd_planner,stdout=f)
    f.close()

    cmd_decoder = "python","decoder.py","--value-policy","verify_attt_planner","--opponent", opponent 
    print("\n","Generating the decoded policy file using decoder.py")
    cmd_output = subprocess.check_output(cmd_decoder,universal_newlines=True)
    ret = check(cmd_output, start)

    os.remove('verify_attt_mdp')
    os.remove('verify_attt_planner')
    return ret

if __name__ == "__main__":
    opponent = 'data/football/test-1.txt'
    g1_p = [0, 0.1, 0.2, 0.3, 0.4, 0.5]
    g1_q = 0.7
    g2_p = 0.3
    g2_q = [0.6, 0.7, 0.8, 0.9, 1]
    start = "0509081"

    fig, (ax1, ax2) = plt.subplots(2)
    fig.tight_layout(pad=1.8)
    # ax1.set_title('Expected goals vs p for fixed q = 0.7')
    ax1.set_xlabel('p for q = 0.7')
    ax1.xaxis.set_label_coords(0.5, -0.2)
    ax1.set_ylabel('Expected goals')

    # ax2.set_title('Expected goals vs q for fixed p = 0.3')
    ax2.set_xlabel('q for p = 0.3')
    ax2.set_ylabel('Expected goals')
    l1 = []
    l2 = []
    for p in g1_p:
        val = run(opponent, p, g1_q, start)
        l1.append(val)

    for q in g2_q:
        val = run(opponent, g2_p, q, start)
        l2.append(val)

    ax1.plot(g1_p, l1, '--bo', color = 'red')
    ax2.plot(g2_q, l2, '--bo', color = 'green')
    fig.savefig('plot.png')
    plt.show()

    print(l1)
    print(l2)

