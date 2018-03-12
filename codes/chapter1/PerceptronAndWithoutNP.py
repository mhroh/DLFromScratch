
def And (x1, x2) :
    w1, w2, t = 0.5, 0.5, 0.5
    temp = w1*x1 + w2*x2

    if t < temp:
        return 1
    else:
        return 0



if __name__ == '__main__':
    print (And(1,1))
    print (And(0,0))
    print (And(1,0))
    print (And(0,1))
