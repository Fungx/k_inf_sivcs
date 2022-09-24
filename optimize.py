from scipy.optimize import minimize, Bounds
import numpy as np

EPSILON = 1e-6


class OptimizedResult:
    def __init__(self, success, nitr, contrast, vars):
        self.success = success
        self.contrast = contrast
        self.nitr = nitr
        self.vars = vars

    def __str__(self) -> str:
        return str(self.__dict__)


"""
非线性规划
"""


def vars_of(x: np.ndarray, k: int, type: str) -> np.ndarray:
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
    assert abs(sum(p0) - 1) < EPSILON and abs(sum(p1) - 1) < EPSILON
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
    cons = [{'type': 'eq', 'fun': lambda x: 1 - sum(vars_of(x, k, 'p0'))},
            {'type': 'eq', 'fun': lambda x: 1 - sum(vars_of(x, k, 'p1'))}]

    def ppx_cons(x: np.ndarray, order: int) -> int:
        xn = np.power(vars_of(x, k, 'r'), order)
        return sum(vars_of(x, k, 'p0') * xn) - sum(vars_of(x, k, 'p1') * xn)

    # 安全性条件
    for n in range(1, k):
        cons.append({'type': 'eq', 'fun': lambda x: ppx_cons(x, n)})

    # 对比度条件
    cons.append({'type': 'ineq', 'fun': lambda x: ppx_cons(x, k)})
    result = minimize(fun=lambda x: -ppx_cons(x, k), x0=x0, bounds=bounds, constraints=cons)
    variables = np.array([vars_of(result['x'], k, 'p0'), vars_of(result['x'], k, 'p1'), vars_of(result['x'], k, 'r')])
    optres = OptimizedResult(result['success'], result.nit, _contrast(variables), variables)
    return optres


"""
SA
"""


def _contrast(vars: np.ndarray) -> float:
    """
    求对比度
    """
    k = len(vars[0]) - 1
    rk = np.power(vars[2], k)
    return sum(vars[0] * rk) - sum(vars[1] * rk)


def _penalty(vars: np.ndarray, ws, wc) -> float:
    k = len(vars[0]) - 1
    # security penalty
    ps = [0] * (k - 1)
    for order in range(1, k):
        rn = np.power(vars[2], order)
        ps[order - 1] = max(0, abs(sum(vars[1] * rn) - sum(vars[0] * rn)) - EPSILON)
    sum_ps = sum(ps)
    # contrast penalty
    rn = np.power(vars[2], k)
    sum_pc = max(0, sum(vars[1] * rn) - sum(vars[0] * rn) + EPSILON)

    return 1 / ((1 + ws * sum_ps) * (1 + wc * sum_pc))


def _energy(vars: np.ndarray, ws, wc) -> float:
    return _contrast(vars) * _penalty(vars, ws, wc)


def optimize_sa1(k: int, init_vars=None, maxitr=1_000_000_000, esp_p=0.5, esp_r=0.5, initial_temp=2500.0,
                 terminated_temp=0.0001, alpha=0.99, ws=25, wc=1):
    """
    模拟退火
    :param k: 阈值
    :param init_vars: 初始变量
    :param maxitr: 最大迭代次数
    :param esp_p: 更新p0,p1的参数
    :param esp_r: 更新r的参数
    :param initial_temp: 初始温度
    :param terminated_temp: 结束温度
    :param alpha: 降温值，越大降温越慢
    :param ws: 惩罚函数的安全性系数
    :param wc: 惩罚函数的对比度系数
    :return:
    """
    # randomly initialize
    # [[p0],[p1],[r]]
    prev_vars = np.array([np.random.dirichlet(np.ones(k + 1)), np.random.dirichlet(np.ones(k + 1)),
                          np.random.uniform(0, 1, k + 1)]) if init_vars is None \
        else init_vars

    temp = initial_temp
    itr_cnt = 0
    best_energy = _energy(prev_vars, ws, wc)
    best_vars = prev_vars.copy()
    prev_energy = best_energy

    # start sa
    while itr_cnt < maxitr and temp > terminated_temp:
        new_vars = prev_vars.copy()
        # randomly alter new_vars (k+1)//2 times
        for _ in range((k + 1) // 2):
            # alter p0,p1
            i1, i2 = np.random.choice(k + 1, 2, False)
            p_color = new_vars[np.random.choice(2)]  # randomly choose p0 or p1
            tmp_esp = np.random.uniform(0, esp_p)  # [0,esp_p)
            chosen_esp = min(tmp_esp, 1 - p_color[i1], p_color[i2])
            p_color[i1] += chosen_esp
            p_color[i2] -= chosen_esp
            # alter r
            j1 = np.random.choice(k + 1)
            tmp_esp = np.random.uniform(-esp_r, esp_r)
            r_j1 = new_vars[2][j1]
            r_j1 += tmp_esp
            if r_j1 > 1:
                r_j1 = 1
            elif r_j1 < 0:
                r_j1 = 0
            new_vars[2][j1] = r_j1

        # calc new_energy
        new_energy = _energy(new_vars, ws, wc)

        # update best vars
        if new_energy > best_energy:
            best_vars = new_vars.copy()
            best_energy = new_energy

        # set up next search
        delta = new_energy - prev_energy
        if delta > 0 or np.random.random() < np.exp(delta / temp):
            # receive new vars as the next start point of searching
            prev_vars = new_vars
        # else use previous vars to search again

        # update params
        itr_cnt += 1
        temp = temp * alpha
        # check if all vars still in bounds
        assert abs(sum(prev_vars[0]) - 1) < EPSILON and abs(sum(prev_vars[1]) - 1) < EPSILON and \
               (0 <= prev_vars[2].all() <= 1)

    return OptimizedResult(True, itr_cnt, _contrast(best_vars), best_vars)


if __name__ == '__main__':
    k = 2
    vs = np.array([[1 / 3, 0, 2 / 3], [0, 1, 0], [0, 2 / 3, 1]])
    result = optimize_sa1(k)
    # result = optimize_sa1(k, init_vars=vs)
    # print(_contrast(vs))
    # print(_penalty(vs, 1, 1))
    # print(_energy(vs))
    print(result)
