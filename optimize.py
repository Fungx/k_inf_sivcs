import math

from scipy.optimize import minimize, Bounds
import numpy as np

EPSILON = 1e-5


class OptimizedResult:
    def __init__(self, success, nitr, contrast, variables,safety=None):
        self.success = success
        self.contrast = contrast
        self.nitr = nitr
        self.variables = variables
        self.safety = safety

    def __str__(self) -> str:
        return str(self.__dict__)


"""
非线性规划
"""


def variables_of(x: np.ndarray, k: int, type: str) -> np.ndarray:
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
    """
    使用`scipy`的非线性规划
    :param k:
    :return:
    """
    # randomly initialize variables
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
    cons = [{'type': 'eq', 'fun': lambda x: 1 - sum(variables_of(x, k, 'p0'))},
            {'type': 'eq', 'fun': lambda x: 1 - sum(variables_of(x, k, 'p1'))}]

    def ppx_cons(x: np.ndarray, order: int) -> int:
        xn = np.power(variables_of(x, k, 'r'), order)
        return sum(variables_of(x, k, 'p0') * xn) - sum(variables_of(x, k, 'p1') * xn)

    # 安全性条件
    for n in range(1, k):
        cons.append({'type': 'eq', 'fun': lambda x: ppx_cons(x, n)})

    # 对比度条件
    cons.append({'type': 'ineq', 'fun': lambda x: ppx_cons(x, k)})
    result = minimize(fun=lambda x: -ppx_cons(x, k), x0=x0, bounds=bounds, constraints=cons)
    variables = np.array(
        [variables_of(result['x'], k, 'p0'), variables_of(result['x'], k, 'p1'), variables_of(result['x'], k, 'r')])
    optres = OptimizedResult(result['success'], result.nit, contrast(variables), variables)
    return optres


"""
SA
"""


def contrast(variables: np.ndarray) -> float:
    """
    对比度
    """
    k = len(variables[0]) - 1
    rk = np.power(variables[2], k)
    return sum(variables[0] * rk) - sum(variables[1] * rk)


def safety_penalty_list(variables: np.ndarray) -> list:
    """
    安全惩罚值，也就是不足k张堆叠时的对比度
    :param variables:
    :return: 长度为`n-1`的列表，分别代表 1,...,n-1张堆叠的对比度
    """
    k = len(variables[0]) - 1
    # security penalty
    ps = [0] * (k - 1)
    for order in range(1, k):
        rn = np.power(variables[2], order)
        ps[order - 1] = max(0, abs(sum(variables[1] * rn) - sum(variables[0] * rn)) - EPSILON)
    return ps


def penalty(variables: np.ndarray, ws, wc) -> float:
    """
    总惩罚
    :param variables:
    :param ws:
    :param wc:
    :return:
    """
    k = len(variables[0]) - 1
    # security penalty
    sum_ps = sum(safety_penalty_list(variables))
    # contrast penalty
    rn = np.power(variables[2], k)
    sum_pc = max(0, sum(variables[1] * rn) - sum(variables[0] * rn) + EPSILON)

    return 1 / ((1 + ws * sum_ps) * (1 + wc * sum_pc))


def energy(variables: np.ndarray, ws, wc) -> float:
    """
    能量值
    :param variables:
    :param ws:
    :param wc:
    :return:
    """
    return contrast(variables) * penalty(variables, ws, wc)


# def energy_sa3(variables: np.ndarray, ws, wc) -> float:
#     return contrast(variables) - 1 / penalty(variables, ws, wc)


def combination_num(n, i) -> int:
    """组合数"""
    return math.factorial(n) / (math.factorial(i) * math.factorial(n - i))


def optimize_sa1(k: int, init_variables=None, maxitr=1_000_000, esp_p=0.5, esp_r=0.5, initial_temp=2000.0,
                 terminated_temp=0.0001, alpha=0.99, ws=25, wc=1):
    """
    模拟退火1，一层循环
    :param k: 阈值
    :param init_variables: 初始变量
    :param maxitr: 最大迭代次数
    :param esp_p: 更新p0,p1的参数
    :param esp_r: 更新r的参数
    :param initial_temp: 初始温度
    :param terminated_temp: 结束温度
    :param alpha: 降温系数，越大降温越慢
    :param ws: 惩罚函数的安全性系数
    :param wc: 惩罚函数的对比度系数
    :return:
    """
    # randomly initialize
    # [[p0],[p1],[r]]
    prev_variables = np.array([[combination_num(k, i) / (2 ** (k - 1)) if i % 2 == 0 else 0 for i in range(k + 1)],
                               [combination_num(k, i) / (2 ** (k - 1)) if i % 2 != 0 else 0 for i in range(k + 1)],
                               np.random.uniform(0, 1, k + 1)]) if init_variables is None \
        else init_variables

    assert abs(sum(prev_variables[0]) - 1) < EPSILON and abs(sum(prev_variables[1]) - 1) < EPSILON and \
           (0 <= prev_variables[2].all() <= 1)

    temp = initial_temp
    itr_cnt = 0
    best_energy = energy(prev_variables, ws, wc)
    best_variables = prev_variables.copy()
    prev_energy = best_energy

    # start sa
    while itr_cnt < maxitr and temp > terminated_temp:
        new_variables = prev_variables.copy()
        # randomly alter new_variables (k+1)//2 times
        for _ in range((k + 1) // 2):
            # alter p0,p1
            i1, i2 = np.random.choice(k + 1, 2, False)
            p_color = new_variables[np.random.choice(2)]  # randomly choose p0 or p1
            tmp_esp = np.random.uniform(0, esp_p)  # [0,esp_p)
            chosen_esp = min(tmp_esp, 1 - p_color[i1], p_color[i2])
            p_color[i1] += chosen_esp
            p_color[i2] -= chosen_esp
            # alter r
            j1 = np.random.choice(k + 1)
            tmp_esp = np.random.uniform(-esp_r, esp_r)
            r_j1 = new_variables[2][j1]
            r_j1 += tmp_esp
            if r_j1 > 1:
                r_j1 = 1
            elif r_j1 < 0:
                r_j1 = 0
            new_variables[2][j1] = r_j1

        # calc new_energy
        new_energy = energy(new_variables, ws, wc)

        # update best variables
        if new_energy > best_energy:
            best_variables = new_variables.copy()
            best_energy = new_energy

        # set up next search
        delta = new_energy - prev_energy
        if delta > 0 or np.random.random() < np.exp(delta / temp):
            # receive new variables as the next start point of searching
            prev_variables = new_variables
        # else use previous variables to search again

        # update params
        itr_cnt += 1
        temp = temp * alpha
        # check if all variables still in bounds
        assert abs(sum(prev_variables[0]) - 1) < EPSILON and abs(sum(prev_variables[1]) - 1) < EPSILON and \
               (0 <= prev_variables[2].all() <= 1)

    return OptimizedResult(True, itr_cnt, contrast(best_variables), best_variables)


def optimize_sa2(k: int, init_variables=None, maxitr=10, markov=2000, esp_p=0.5, esp_r=0.5, initial_temp=0.1,
                 terminated_temp=0.1/2000, alpha=0.95, ws=25, wc=1):
    """
    模拟退火2，两层循环，每次迭代不更新`esp_p`,`esp_r`和`markov`
    :param k: 阈值
    :param init_variables: 初始变量
    :param maxitr: 最大迭代次数
    :param markov: 马尔可夫链长度，也就是每次迭代，在同一温度下搜索的次数
    :param esp_p: 更新p0,p1的参数
    :param esp_r: 更新r的参数
    :param initial_temp: 初始温度
    :param terminated_temp: 结束温度
    :param alpha: 降温系数，越大降温越慢
    :param ws: 惩罚函数的安全性系数
    :param wc: 惩罚函数的对比度系数
    :return:
    """
    # randomly initialize
    # [[p0],[p1],[r]]
    prev_variables = np.array([[combination_num(k, i) / (2 ** (k - 1)) if i % 2 == 0 else 0 for i in range(k + 1)],
                               [combination_num(k, i) / (2 ** (k - 1)) if i % 2 != 0 else 0 for i in range(k + 1)],
                               np.random.uniform(0, 1, k + 1)]) if init_variables is None \
        else init_variables
    assert abs(sum(prev_variables[0]) - 1) < EPSILON and abs(sum(prev_variables[1]) - 1) < EPSILON and \
           (0 <= prev_variables[2].all() <= 1)
    temp = initial_temp
    itr_cnt = 0
    best_energy = energy(prev_variables, ws, wc)
    best_variables = prev_variables.copy()
    prev_energy = best_energy

    search_cnt = 0
    # start sa
    while itr_cnt < maxitr and temp > terminated_temp:
        for _ in range(markov):
            new_variables = prev_variables.copy()
            # randomly alter new_variables (k+1)//2 times
            for _ in range((k + 1) // 2):
                # alter p0,p1
                i1, i2 = np.random.choice(k + 1, 2, False)
                p_color = new_variables[np.random.choice(2)]  # randomly choose p0 or p1
                tmp_esp = np.random.uniform(0, esp_p)  # [0,esp_p)
                chosen_esp = min(tmp_esp, 1 - p_color[i1], p_color[i2])
                p_color[i1] += chosen_esp
                p_color[i2] -= chosen_esp
                # alter r
                j1 = np.random.choice(k + 1)
                tmp_esp = np.random.uniform(-esp_r, esp_r)
                r_j1 = new_variables[2][j1]
                r_j1 += tmp_esp
                if r_j1 > 1:
                    r_j1 = 1
                elif r_j1 < 0:
                    r_j1 = 0
                new_variables[2][j1] = r_j1

            # calc new_energy
            new_energy = energy(new_variables, ws, wc)

            # set up next search
            delta = new_energy - prev_energy
            if delta > 0 or np.random.random() < np.exp(delta / temp):
                # receive new variables as the next start point of searching
                prev_variables = new_variables
                prev_energy = new_energy
                # update best variables
                if prev_energy > best_energy:
                    best_variables = prev_variables.copy()
                    best_energy = prev_energy
                    itr_cnt = 0
            # else use previous variables to search again

            # check if all variables still in bounds
            assert abs(sum(prev_variables[0]) - 1) < EPSILON and abs(sum(prev_variables[1]) - 1) < EPSILON and \
                   (0 <= prev_variables[2].all() <= 1)
            search_cnt += 1
        # update params
        itr_cnt += 1
        temp = temp * alpha
    return OptimizedResult(True, search_cnt, contrast(best_variables), best_variables,safety_penalty_list(best_variables))


def optimize_sa3(k: int, init_variables=None, maxitr=10, markov=2000, esp_p=0.5, esp_r=0.5, initial_temp=2000.0,
                 terminated_temp=0.001, alpha=0.95, ws=50, wc=1):
    """
    模拟退火3，两层循环，每次迭代更新`esp_p`,`esp_r`和`markov`
    :param k: 阈值
    :param init_variables: 初始变量
    :param maxitr: 最大迭代次数
    :param markov: 马尔可夫链长度，也就是每次迭代，在同一温度下搜索的次数
    :param esp_p: 更新p0,p1的参数
    :param esp_r: 更新r的参数
    :param initial_temp: 初始温度
    :param terminated_temp: 结束温度
    :param alpha: 降温系数，越大降温越慢
    :param ws: 惩罚函数的安全性系数
    :param wc: 惩罚函数的对比度系数
    :return:
    """

    # randomly initialize
    # [[p0],[p1],[r]]
    prev_variables = np.array([[combination_num(k, i) / (2 ** (k - 1)) if i % 2 == 0 else 0 for i in range(k + 1)],
                               [combination_num(k, i) / (2 ** (k - 1)) if i % 2 != 0 else 0 for i in range(k + 1)],
                               np.random.uniform(0, 1, k + 1)]) if init_variables is None \
        else init_variables
    assert abs(sum(prev_variables[0]) - 1) < EPSILON and abs(sum(prev_variables[1]) - 1) < EPSILON and \
           (0 <= prev_variables[2].all() <= 1)
    temp = initial_temp
    itr_cnt = 0
    best_energy = energy(prev_variables, ws, wc)
    best_variables = prev_variables.copy()
    prev_energy = best_energy

    search_cnt = 0;
    # start sa
    while itr_cnt < maxitr and temp > terminated_temp:
        for _ in range(markov):
            new_variables = prev_variables.copy()
            # randomly alter new_variables (k+1)//2 times
            for _ in range((k + 1) // 2):
                # alter p0,p1
                i1, i2 = np.random.choice(k + 1, 2, False)
                p_color = new_variables[np.random.choice(2)]  # randomly choose p0 or p1
                tmp_esp = np.random.uniform(0, esp_p)  # [0,esp_p)
                chosen_esp = min(tmp_esp, 1 - p_color[i1], p_color[i2])
                p_color[i1] += chosen_esp
                p_color[i2] -= chosen_esp
                # alter r
                j1 = np.random.choice(k + 1)
                tmp_esp = np.random.uniform(-esp_r, esp_r)
                r_j1 = new_variables[2][j1]
                r_j1 += tmp_esp
                if r_j1 > 1:
                    r_j1 = 1
                elif r_j1 < 0:
                    r_j1 = 0
                new_variables[2][j1] = r_j1

            # calc new_energy
            new_energy = energy(new_variables, ws, wc)

            # set up next search
            delta = new_energy - prev_energy
            if delta > 0 or np.random.random() < np.exp(delta / temp):
                # receive new variables as the next start point of searching
                prev_variables = new_variables
                prev_energy = new_energy
                # update best variables
                if prev_energy > best_energy:
                    best_variables = prev_variables.copy()
                    best_energy = prev_energy
                    itr_cnt = 0
            # else use previous variables to search again

            # check if all variables still in bounds
            assert abs(sum(prev_variables[0]) - 1) < EPSILON and abs(sum(prev_variables[1]) - 1) < EPSILON and \
                   (0 <= prev_variables[2].all() <= 1)
            search_cnt += 1
        # update params
        itr_cnt += 1
        temp = temp * alpha
        markov = int(markov * 1.5)
        esp_p = max(esp_p * 0.67, 0.001)
        esp_r = max(esp_r * 0.67, 0.001)
    return OptimizedResult(True, search_cnt, contrast(best_variables), best_variables)


if __name__ == '__main__':
    result = optimize_sa3(3, initial_temp=0.1, terminated_temp=0.1 / 2000, ws=50)
    # result = optimize_sa1(k, init_variables=vs)
    # print(_contrast(vs))
    # print(_penalty(vs, 1, 1))
    # print(_energy(vs))
    print(result)
