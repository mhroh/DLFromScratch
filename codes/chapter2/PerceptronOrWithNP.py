import numpy as np

def Or (x1, x2) :
   x = np.array([x1, x2])
   w = np.array([0.5, 0.5])
   b = -0.4

   py = np.sum(x*w) + b

   if py > 0:
      return 1
   else:
      return 0


if __name__ == "__main__":
   print(Or(1,1))
   print(Or(0,1))
   print(Or(1,0))
   print(Or(0,0))

