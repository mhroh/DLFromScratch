import numpy as np

def And (x1, x2):
    x = np.array([x1, x2])
    w = np.array([0.5, 0.5])
    b = -0.5

    y = np.sum(x*w) + b

    if y <= 0:
        return 0
    else:
        return 1

def NAnd(x1, x2):
    if not And(x1, x2):
        return 1
    else:
        return 0

if __name__ == "__main__":
    print(And(1,1))
    print(And(0,1))
    print(And(1,0))
    print(And(0,0))

    print (NAnd(1,1))
    print (NAnd(0,1))
    print (NAnd(1,0))
    print (NAnd(0,0))
