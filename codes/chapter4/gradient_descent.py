from chapter4.differential2 import numerical_gradient
import numpy as np

def gradient_descent(f, init_x, lr=0.01, step_num=100):
    x = init_x
    for i in range(step_num):
        grad = numerical_gradient(f, x)
        #learning rate는 gradient_descent가 좀 더 빠르게 최소값에 근사한 값에
        #도달하게 만들기 위해 취하는 하이퍼파라미터 이다. 만약 이 값을 적용하지 않는다면
        #최소값에 근사할 때 까지 걸리는 시간이 너무 오래 걸릴 것이다.
        #반면에 이 값이 너무 크다면 발산해 버리는 경우(overshoot)가 생기게 된다.
        #명싱하자 우리가 실제 공학(수학이 아닌)에서 구할 수 있는 최소값은 근사값이라는 것을.
        x -= lr * grad

        print("step : ", i, " grad : ", grad, " x : ", x)
    return x

if __name__ == "__main__":
    def function_2(x):
        return x[0]**2 + x[1]**2

    x = np.array([[-3.0, 4.0], [-2.0, 3.0]])

    result1 = gradient_descent(function_2, init_x=x, step_num=1000)
    result2 = gradient_descent(function_2, init_x=x, lr=10.0, step_num=1000)
    result3 = gradient_descent(function_2, init_x=x, lr=1e-10, step_num=1000)

    print("result1=", result1, " result2=", result2, " result3=", result3)



