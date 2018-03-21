import sys, os
sys.path.append(os.pardir)
from dataset.mnist import load_mnist

'''
flatten = True는 입력되는 숫자 이미지가 Grey 1channel로 구성되어 있다고 하였으니
그리고 28 * 28 인 pixel을 1차원 배열로 리턴한다는 것이다. 
즉 길이가 784인 array가 나온다는 의미, x_train은 0 ~ 9까지 길이가 각각 784인 array를 즉
n by 784 인 이미지와 n by 1 인 데이터가 로드된다는 것.
'''
#(훈련 이미지, 훈련 레이블), (시험 이미지, 시험 레이블)
(x_train, t_train), (x_test, t_test) = load_mnist(flatten=True, normalize=False)

print("X-Tr Shape ", x_train.shape, "X-Tr Rank ", x_train.ndim)
print("T-Tr Shape ", t_train.shape, "T-Tr Rank ", t_train.ndim)
print("X-Te Shape ", x_test.shape, "X-Te Rank ", x_test.ndim)
print("T-Te Shape ", t_test.shape, "T-Te Rank ", t_test.ndim)
