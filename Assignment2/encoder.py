import numpy as np
from enum import Enum
import argparse, sys
parser = argparse.ArgumentParser()

R = {} # opponent policy
p = 0
q = 0

# states -> 0 - 8191 normal game position, 8192 - end state
end = 8192
goal = 8193

def coord(b):
    return ((b-1) % 4), ((b-1) // 4) # x -> horizontal, y -> vertical

def encode(b1, b2, r, pos):
    # zero -> even -> 1
    # one -> odd -> 2

    x = ((b1-1)*256 + (b2-1)*16 + r-1) * 2
    x = x + 1 if pos == 2 else x
    return x

def lexico_state(b1, b2, r, pos):
    return (b1*10000 + b2*100 + r) * 10 + pos

def next_state(bx, by, ba, rx, ry, ra):
    bx_ = bx
    rx_ = rx
    by_ = by
    ry_ = ry

    if ba == 0:
        bx_ = bx - 1
    elif ba == 1:
        bx_ = bx + 1
    elif ba == 2:
        by_ = by - 1
    elif ba == 3:
        by_ = by + 1
    
    if ra == 0:
        rx_ = rx - 1
    elif ra == 1:
        rx_ = rx + 1
    elif ra == 2:
        ry_ = ry - 1
    elif ra == 3:
        ry_ = ry + 1

    return ((bx_ == rx_ and by_ == ry_) or (bx_ == rx and by_ == ry 
                    and rx_ == bx and ry_ == by)), (4*by_ + bx_ + 1), (4*ry_ + rx_ + 1) 

# actions -> 0,1,2,3 for B1, 4,5,6,7 for B2, 8 - PASS, 9 - SHOOT
# action -> 0,1,2,3 for L,R,U,D
def calc_transition(b1, b2, r, pos):
    x1, y1 = coord(b1)
    x2, y2 = coord(b2)
    xr, yr = coord(r)
    curr_state = encode(b1, b2, r, pos)
    for action in range(10):
        end_prob = 0
        if pos == 1:
            if (action == 0 and x1 > 0) or (action == 1 and x1 < 3) or \
                (action == 2 and y1 > 0) or (action == 3 and y1 < 3):
                for move in range(4):
                    opp = R[lexico_state(b1, b2, r, pos)][move] # probabilty
                    if(opp == 0): 
                        continue
                    tackle, b1_, r_ = next_state(x1, y1, action, xr, yr, move)
                    if(tackle == True):
                        end_prob += opp*(0.5+p)
                        print("transition", curr_state, action, encode(b1_, b2, r_, pos), 0, opp*(0.5-p))
                    else:
                        print("transition", curr_state, action, encode(b1_, b2, r_, pos), 0, (1.0-2*p)*opp)
                        end_prob += 2*p*opp
                print("transition", curr_state, action, end, 0, end_prob)


            elif (action == 4 and x2 > 0) or (action == 5 and x2 < 3) or \
                (action == 6 and y2 > 0) or (action == 7 and y2 < 3):
                for move in range(4):
                    opp = R[lexico_state(b1, b2, r, pos)][move] # probabilty
                    if(opp == 0): 
                        continue
                    _, b2_, r_ = next_state(x2, y2, (action-4), xr, yr, move) # no tackle
                    print("transition", curr_state, action, encode(b1, b2_, r_, pos), 0, (1.0-p)*opp)
                    end_prob += p*opp
                print("transition", curr_state, action, end, 0, end_prob)
            
            elif action == 8:
                for move in range(4):
                    opp = R[lexico_state(b1, b2, r, pos)][move]
                    if(opp == 0): 
                        continue
                    _, _, r_ = next_state(0, 0, 0, xr, yr, move)
                    rx_, ry_ = coord(r_)
                    prob = q - 0.1 * max(abs(x1 -  x2), abs(y1 - y2))

                    epsilon = 1e-6
                    if (rx_ >= min(x1, x2) and rx_ <= max(x1, x2)) and (ry_ >= min(y1, y2) and ry_ <= max(y1, y2)):
                        if x1 == x2:
                            prob /= 2
                        else:
                            m = (y2 - y1)/(x2 - x1)
                            line = y1 + m*(rx_ - x1)
                            prob = prob/2 if ( (abs(ry_ - line) < epsilon) and ((abs(m) == 1.0) or m == 0) ) else prob
                            
                    print("transition", curr_state, action, encode(b1, b2, r_, 3-pos), 0, prob*opp)
                    end_prob += (1.0-prob)*opp
                print("transition", curr_state, action, end, 0, end_prob)
            
            elif action == 9:
                goal_prob = 0
                for move in range(4):
                    opp = R[lexico_state(b1, b2, r, pos)][move]
                    if(opp == 0): 
                        continue
                    _, _, r_ = next_state(0, 0, 0, xr, yr, move)
                    rx_, ry_ = coord(r_)
                    prob = q - 0.2 * (3 - x1)
                    if (rx_ == 3 and (ry_ == 1 or ry_ == 2)):
                        prob /= 2
                    goal_prob += prob*opp
                    end_prob += (1.0-prob)*opp
                print("transition", curr_state, action, goal, 1, goal_prob)
                print("transition", curr_state, action, end, 0, end_prob)

            else:
                print("transition", curr_state, action, end, 0, 1.0) # zero reward

        elif pos == 2:
            if (action == 0 and x1 > 0) or (action == 1 and x1 < 3) or \
                (action == 2 and y1 > 0) or (action == 3 and y1 < 3):
                for move in range(4):
                    opp = R[lexico_state(b1, b2, r, pos)][move] # probabilty
                    if(opp == 0): 
                        continue
                    _, b1_, r_ = next_state(x1, y1, action, xr, yr, move) # no tackle
                    print("transition", curr_state, action, encode(b1_, b2, r_, pos), 0, (1.0-p)*opp)
                    end_prob += p*opp
                print("transition", curr_state, action, end, 0, end_prob)

            elif (action == 4 and x2 > 0) or (action == 5 and x2 < 3) or \
                (action == 6 and y2 > 0) or (action == 7 and y2 < 3):
                for move in range(4):
                    opp = R[lexico_state(b1, b2, r, pos)][move] # probabilty
                    if(opp == 0): 
                        continue
                    tackle, b2_, r_ = next_state(x2, y2, (action-4), xr, yr, move)
                    if(tackle == True):
                        end_prob += opp*(0.5+p)
                        print("transition", curr_state, action, encode(b1, b2_, r_, pos), 0, opp*(0.5-p))
                    else:
                        print("transition", curr_state, action, encode(b1, b2_, r_, pos), 0, (1.0-2*p)*opp)
                        end_prob += 2*p*opp
                print("transition", curr_state, action, end, 0, end_prob)

            elif action == 8:
                for move in range(4):
                    opp = R[lexico_state(b1, b2, r, pos)][move]
                    if(opp == 0): 
                        continue
                    _, _, r_ = next_state(0, 0, 0, xr, yr, move)
                    rx_, ry_ = coord(r_)
                    prob = q - 0.1 * max(abs(x2 -  x1), abs(y2 - y1))

                    epsilon = 1e-6
                    if (rx_ >= min(x1, x2) and rx_ <= max(x1, x2)) and (ry_ >= min(y1, y2) and ry_ <= max(y1, y2)):
                        if x1 == x2:
                            prob /= 2
                        else:
                            m = (y2 - y1)/(x2 - x1)
                            line = y1 + m*(rx_ - x1)
                            prob = prob/2 if ( (abs(ry_ - line) < epsilon) and ((abs(m) == 1.0) or m == 0) ) else prob

                    print("transition", curr_state, action, encode(b1, b2, r_, 3-pos), 0, prob*opp)
                    end_prob += (1.0-prob)*opp
                print("transition", curr_state, action, end, 0, end_prob)
            
            elif action == 9:
                goal_prob = 0
                for move in range(4):
                    opp = R[lexico_state(b1, b2, r, pos)][move]
                    if(opp == 0): 
                        continue
                    _, _, r_ = next_state(0, 0, 0, xr, yr, move)
                    rx_, ry_ = coord(r_)
                    prob = q - 0.2 * (3 - x2)
                    if (rx_ == 3 and (ry_ == 1 or ry_ == 2)):
                        prob /= 2
                    end_prob += (1.0-prob)*opp
                    goal_prob += prob*opp
                print("transition", curr_state, action, end, 0, end_prob)
                print("transition", curr_state, action, goal, 1, goal_prob)

            else:
                print("transition", curr_state, action, end, 0, 1.0) # zero reward


if __name__ == "__main__":
    parser.add_argument("--opponent", type=str)
    parser.add_argument("--p", type=float)
    parser.add_argument("--q", type=float)
    args = parser.parse_args()

    p = args.p
    q = args.q

    try:
        with open(f"{args.opponent}", 'r') as opponent_policy_file:
            for line in opponent_policy_file.readlines()[1:]:
                line = line.strip().split()
                R[int(line[0])] = [float(line[1]), float(line[2]), float(line[3]), float(line[4])] # L R U D Possession

        print("numStates", 8194)
        print("numActions", 10)
        print("end", end, goal)

        for b1 in range(1, 17):
            for b2 in range(1, 17):
                for r in range(1, 17):
                    for pos in [1, 2]:
                        calc_transition(b1, b2, r, pos)

        print("mdptype episodic")
        print("discount", 1.0)
        
        
    except IOError:
        print("Error: unable to read file")