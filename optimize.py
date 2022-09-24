from scipy.optimize import minimize, Bounds
import numpy as np

PRECISION = 1e-6


def vars(x: np.ndarray, k: int, type: str) -> np.ndarray:
    """
    变量一共有3*(k+1)个，分别为：
    - 白色秘密像素的生成器选择概率 p0 = [p0_0, p0_1,..., p0_k]
    - 黑色秘密像素的生成器选择概率 p1 = [p1_0, p1_1,..., p1_k]
    - 生成器的参数               r  = [r_0, r_1,..., r_k]
    在x中排列方式如下: x = [p0_0, ..., p0_k, p1_0, ..., p1_k, r_0, ..., r_k]
    :param x: 变量列表
    :param type: 变量类型
    :return: `x`中所有`type`类型的变量
    """
    if type == 'p0':
        return x[0:k + 1]
    elif type == 'p1':
        return x[k + 1:2 * (k + 1)]
    else:
        return x[2 * (k + 1):]


def optimize_nonliner(k: int):
    # randomly initialize vars
    p0 = np.random.dirichlet(np.ones(k + 1))  # 不知道什么分布，可以产生k+1个和为1的随机值
    p1 = np.random.dirichlet(np.ones(k + 1))
    assert abs(sum(p0) - 1) < PRECISION and abs(sum(p1) - 1) < PRECISION
    r = np.random.uniform(0, 1, k + 1)
    # merge into one list
    x0 = []
    x0.extend(p0)
    x0.extend(p1)
    x0.extend(r)

    # 范围： 所有变量都始终满足 0<=x<=1
    bounds = Bounds(0, 1, True)

    # 约束：
    # p0(resp. p1)之和为1
    cons = [{'type': 'eq', 'fun': lambda x: 1 - sum(vars(x, k, 'p0'))},
            {'type': 'eq', 'fun': lambda x: 1 - sum(vars(x, k, 'p1'))}]

    def ppx_cons(x: np.ndarray, order: int) -> int:
        xn = np.power(vars(x, k, 'r'), order)
        return sum(vars(x, k, 'p0') * xn) - sum(vars(x, k, 'p1') * xn)

    # 安全性条件
    for n in range(1, k):
        cons.append({'type': 'eq', 'fun': lambda x: ppx_cons(x, n)})

    # 对比度条件
    cons.append({'type': 'ineq', 'fun': lambda x: ppx_cons(x, k)})
    result = minimize(fun=lambda x: -ppx_cons(x, k), x0=x0, bounds=bounds, constraints=cons)
    return {'success': result['success'], 'func': result['fun'],
            'x': {'p0': vars(result['x'], k, 'p0'), 'p1': vars(result['x'], k, 'p1'),
                  'r': vars(result['x'], k, 'r')}}


if __name__ == '__main__':
    k = 6
    result = optimize_nonliner(k)
    print(result)
