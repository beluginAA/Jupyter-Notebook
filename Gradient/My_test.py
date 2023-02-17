import matplotlib.pyplot as plt
import numpy as np
from copy import deepcopy


class LoggingCallback:
    def __init__(self):
        self.x_steps = []
        self.y_steps = []

    def __call__(self, x, y):  # При каждом вызове функции callback у нас в массивы добавляются значения х и y автоматически
        self.x_steps.append(x)
        self.y_steps.append(y)


def plot_convergence_1d(func, x_steps, y_steps, ax, grid=None, title=''):
    ax.set_title(title, fontsize=16, fontweight="bold")
    if grid is None:
        grid = np.linspace(np.min(x_steps), np.max(x_steps), 100)
    fgrid = [func(item) for item in grid]
    ax.plot(grid, fgrid)
    yrange = np.max(fgrid) - np.min(fgrid)
    arrow_kwargs = dict(linestyle="--", color="grey", alpha=0.4)
    for i, _ in enumerate(x_steps):
        if i + 1 < len(x_steps):
            ax.arrow(
                x_steps[i], y_steps[i],
                x_steps[i + 1] - x_steps[i],
                y_steps[i + 1] - y_steps[i],
                **arrow_kwargs
            )
    n = len(x_steps)
    color_list = [(i / n, 0, 0, 1 - i / n) for i in range(n)]
    ax.scatter(x_steps, y_steps, c=color_list)
    ax.scatter(x_steps[-1], y_steps[-1], c="red")
    ax.set_xlabel(r"$x$")
    ax.set_ylabel(r"$y$")


def grad_descent_v1(f, deriv, x0=None, lr=0.1, iters=100, callback=None):
    # Определение шага между х после нахождения производной
    def gradient_descent_step(f_dash, x, alpha=0.001):
        f_dash_x = f_dash(x)  # Определение значения производной в точке
        delta_x = alpha*f_dash_x  # Определение нового значения х
        x_new = x - delta_x
        return x_new
    if x0 is None:  # Определение начальной точки
        np.random.seed(179)
        x0 = np.random.uniform()
    x = x0
    y = f(x)
    xs = [x]
    ys = [y]
    step_count = 0
    callback(x, f(x))
    while True:
        step_count += 1
        x = gradient_descent_step(deriv, x, lr)  # Возвращение х нового
        y = f(x)
        delta_y = np.abs(y-ys[-1])
        xs.append(x)
        ys.append(y)
        if step_count >= iters and delta_y < lr:
            break
    return x


def test_convergence_1d(grad_descent, test_cases, tol=1e-2, axes=None, grid=None):
    right_flag = True
    debug_log = []
    for i, key in enumerate(test_cases.keys()):
        answer = test_cases[key]["answer"]
        test_input = deepcopy(test_cases[key])
        del test_input['answer']
        callback = LoggingCallback()
        res_point = grad_descent(*test_input.values(),  callback=callback)
        if axes is not None:
            # Данная команда позволяет выбрать по порядковому номеру элемент в матрице
            ax = axes[np.unravel_index(i, shape=axes.shape)]
            x_steps = np.array(callback.x_steps)
            y_steps = np.array(callback.y_steps)
            plot_convergence_1d(
                test_input["func"], x_steps, y_steps,
                ax, grid, key
            )
        ax.axvline(answer, 0, linestyle="--", c="red",
                   label=f"true answer = {answer}")
        ax.axvline(res_point, 0, linestyle="--", c="xkcd:tangerine",
                   label=f"estimate = {np.round(res_point, 3)}")
        ax.legend(fontsize=16)
    if abs(answer - res_point) > tol or np.isnan(res_point):
        debug_log.append(
            f"Тест '{key}':\n"
            f"\t- ответ: {answer}\n"
            f"\t- вывод алгоритма: {res_point}"
        )
        right_flag = False
    return right_flag, debug_log


test_cases = {
    "square": {
        "func": lambda x: x**2,
        "deriv": lambda x: 2*x,
        "start": 2,
        "answer": 0.0},
    "other concentric circles": {
        "func": lambda x: (
            -1 / ((x[0])**2 + (x[1] - 3)**2 + 1)
            * np.cos(2 * (x[0])**2 + 2 * (x[1] - 3)**2)),
        "start": np.array([1.1, 3.3]),
        "answer": np.array([0, 3])},
    "straightened ellipses": {
        "func": lambda x: (
            -1 / ((x[0])**4 + (x[1] - 3)**6 + 1)
            * np.cos(2 * (x[0])**4 + 2 * (x[1] - 3)**6)),
        # точка так близко к ответу тк в окрестности ответа градиент маленкьий и функция очень плохо сходится
        "start": np.array([.8, 3.001]),
        "answer": np.array([0, 3])}
}  # Функция квадратичная
tol = 1e-2  # Точность
fig, axes = plt.subplots(2, 2, figsize=(14, 12))
fig.suptitle("Градиентный спуск", fontweight="bold", fontsize=20)
grid = np.linspace(-2, 2, 100)
is_correct, debug_log = test_convergence_1d(
    grad_descent_v1, test_cases, tol,
    axes, grid
)
plt.show()
