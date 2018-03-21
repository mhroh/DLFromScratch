from chapter2.PerceptronAndWithNP import And, NAnd
from chapter2.PerceptronOrWithNP import Or

def XOR(x1, x2):
    S1 = not And(x1, x2)
    S2 = Or(x1, x2)
    Y = And(S1, S2)
    return Y

def XOR2(x1, x2):
    S1 = NAnd(x1, x2)
    S2 = Or(x1, x2)
    Y = And(S1, S2)
    return Y

if __name__ == '__main__':
    print(XOR(1,1))
    print(XOR(0,1))
    print(XOR(1,0))
    print(XOR(0,0))

    print(XOR2(1,1))
    print(XOR2(0,1))
    print(XOR2(1,0))
    print(XOR2(0,0))
