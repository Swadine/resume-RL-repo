import numpy as np
from pulp import *
import copy, argparse, sys
parser = argparse.ArgumentParser()

EPSILON = np.float64(1E-10)  # Epsilon Optimality 

def value_iteration_slower(epsilon=EPSILON):
    if discount == 1.0:
        tolerance = np.float64(epsilon)
    else:
        tolerance = np.float64(epsilon * ((1 - discount)/(2 * discount)))

    values = np.zeros(numStates, dtype=np.float64)
    while True:
        new_values = np.amax(expected_rewards + discount *
                             np.dot(transitions, values), axis=1)
        if np.amax(abs(new_values - values)) < tolerance:
            break
        values = copy.deepcopy(new_values)
        # print(values[1146], file=sys.stderr)

    best_policy = np.argmax(expected_rewards + discount *
                            np.dot(transitions, values), axis=1)

    for value, policy in zip(values, best_policy):
        print(value, policy)

def value_iteration_faster(epsilon=EPSILON):
    # if discount == 1.0:
    #     tolerance = np.float64(epsilon)
    # else:
    #     tolerance = np.float64(epsilon * ((1 - discount)/(2 * discount)))

    values = np.zeros(numStates, dtype=np.float64)
    while True:
        new_values = np.zeros(numStates, dtype=np.float64)
        for s in range(numStates):
            mx = float('-inf')
            for a in range(numActions):
                trans = transitions_lp[s][a]
                val = 0
                for term in trans[1:]:
                    val += term[2] * ( term[1] + discount * values[term[0]])
                mx = max(mx, val)
            new_values[s] = mx

        if np.max(abs(new_values - values)) < EPSILON:
            break
        values = copy.deepcopy(new_values)
        # print(values[68], file=sys.stderr)

    # best_policy = np.argmax(expected_rewards + discount *
    #                         np.dot(transitions, values), axis=1)

    best_policy = []
    for state in range(numStates):
        mx = float('-inf')
        a = 0
        for action in range(numActions):
            val = 0
            trans = transitions_lp[state][action]
            for term in trans[1:]:
                val += term[2] * (term[1] + discount * values[term[0]])

            if val > mx:
                a = action
                mx = val

        best_policy.append(a)

    for value, policy in zip(values, best_policy):
        print(value, policy)

def policy_iteration_veryslow(epsilon=EPSILON):

    pi = np.zeros(numStates, dtype=int)
    values = np.zeros(numStates, dtype=np.float64)

    while True:
        rewards_pi = np.choose(pi, expected_rewards.T)
        # transitions_pi = np.ndarray.choose(pi, np.transpose(transitions, axes=(1, 0, 2)))
        transitions_pi = np.array([transitions[I, pi[I]]
                                  for I in range(len(pi))])
        values_pi = np.dot(np.linalg.inv(np.identity(
            numStates) - discount * transitions_pi), rewards_pi)
        new_pi = np.argmax(expected_rewards + discount *
                           np.dot(transitions, values_pi), axis=1)
        if (new_pi == pi).all() and np.amax(abs(values_pi - values)) < epsilon: 
            break
        pi = copy.deepcopy(new_pi)
        values = copy.deepcopy(values_pi)

    for i in range(len(values)):
        print(values[i], pi[i])

def policy_iteration_faster(epsilon=EPSILON):

    pi = np.zeros(numStates, dtype=int)
    values = np.zeros(numStates, dtype=np.float64)
    values_pi = np.zeros(numStates, dtype=np.float64)

    stable = False

    while not stable:
        while True:
            delta = 0
            for s in range(numStates):
                trans = transitions_lp[s][pi[s]]
                val = 0
                for term in trans[1:]:
                    val += term[2] * (term[1] + discount * values[term[0]])
                values_pi[s] = val # In-place policy evaluation
            
            if np.amax(abs(values_pi - values)) < epsilon:
                break
            values = copy.deepcopy(values_pi)

        stable = True
        
        for state in range(numStates):
            old_pi = pi[state]
            mx = float('-inf')
            for action in range(numActions):
                trans = transitions_lp[state][action]
                val = 0
                for term in trans[1:]:
                    val += term[2] * ( term[1] + discount * values[term[0]])
                pi[state] = action if val > mx else pi[state]
                mx = max(mx, val)
                
            stable &= (old_pi == pi[state])      
        
    for value, policy in zip(values, pi):
        print(value, policy)

def value_function_policy():

    values = np.zeros(numStates, dtype=np.float64)

    while True:
        new_values = np.zeros(numStates, dtype=np.float64)
        for state in range(numStates):
            val = 0
            trans = transitions_lp[state][policy[state]]
            for term in trans[1:]:
                val += term[2] * (term[1] + discount * values[term[0]])
            new_values[state] = val
        
        if(np.amax(abs(new_values - values)) < EPSILON):
            break

        values = copy.deepcopy(new_values)

    for value, action in zip(values, policy):
        print(value, action)


def linear_programming():
    problem = LpProblem('Value_Function', LpMinimize)
    digits = len(str(numStates))
    V = [pulp.LpVariable(("0"*( digits - len(str(i)) ) + str(i)), cat='Continuous') for i in range(numStates)]
    problem += lpSum([V[i] for i in range(numStates)])

    for s in range(numStates):
        for a in range(numActions): 
            trans = transitions_lp[s][a]
            value = 0
            for term in trans[1:]:
                value += term[2] * (term[1] + discount * V[term[0]])
            problem += V[s] >= value
        # print(s, file=sys.stderr)

    problem.solve(PULP_CBC_CMD(msg=0))

    values = [v.varValue for v in problem.variables()]
    # best_policy = np.argmax(expected_rewards + discount *
    #                         np.dot(transitions, values), axis=1)

    best_policy = []
    for state in range(numStates):
        mx = float('-inf')
        a = 0
        for action in range(numActions):
            val = 0
            trans = transitions_lp[state][action]
            for term in trans[1:]:
                val += term[2] * (term[1] + discount * values[term[0]])

            if val > mx:
                a = action
                mx = val

        best_policy.append(a)

    for value, policy in zip(values, best_policy):
        print(value, policy)

if __name__ == "__main__":
    parser.add_argument("--mdp", type=str)
    parser.add_argument("--algorithm", type=str)
    parser.add_argument("--policy", type=str)
    args = parser.parse_args()

    try:
        with open(f"{args.mdp}", 'r') as mdp_file:
            for line in mdp_file.readlines():
                line = " ".join(line.split()).split()
                if (line[0] == 'numStates'):
                    numStates = int(line[1])
                elif (line[0] == 'numActions'):
                    numActions = int(line[1])
                    # s, a, s'
                    # rewards = np.zeros(
                    #     (numStates, numActions, numStates), np.float64)
                    # # s, a, s'
                    # transitions = np.zeros(
                    #     (numStates, numActions, numStates), np.float64)
                    transitions_lp = transitions_lp = [ [ [[0,0.0,0.0]] for i in range(numActions)] for j in range(numStates) ]
                    # transitions_lp = []
                    # for state in range(numStates):
                    #     row = []
                    #     for action in range(numActions):
                    #         col = [[0.0,0.0,0.0]] # s', r, p
                    #         row.append(col)
                    #     transitions_lp.append(row)

                elif (line[0] == 'end'):
                    end = np.array(line[1:], dtype = np.int32)
                elif (line[0] == 'discount'):
                    discount = float(line[1])
                elif (line[0] == 'transition'):
                    transitions_lp[int(line[1])][int(line[2])].append([int(line[3]),float(line[4]),float(line[5])]) # s, a, s', r, p
                    # rewards[int(line[1]), int(line[2]), int(line[3])
                    #         ] = np.float64(line[4])  # s, a, s'
                    # transitions[int(line[1]), int(line[2]), int(line[3])
                    #         ] = np.float64(line[5])  # s, a, s'

        # expected_rewards = np.zeros((numStates, numActions), np.float64)
        # for state in range(numStates):
        #     for action in range(numActions):
        #         expected_rewards[state, action] = np.dot(transitions[state, action], rewards[state, action])
        if (args.algorithm == 'vi'):
            value_iteration_faster()
        elif (args.algorithm == 'hpi'):
            policy_iteration_faster()
        elif (args.algorithm == 'lp'):
            linear_programming()
        elif(type(args.policy) != type(None)):
            policy = []
            with open(f"{args.policy}", 'r') as policy_file:
                for line in policy_file.readlines():
                    policy.append(int(line))
            value_function_policy()
        else:
            value_iteration_faster()

    except IOError:
        print("Please check the path.")

    
