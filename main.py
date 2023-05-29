import numpy as np
from scipy.optimize import linprog
import warnings

warnings.filterwarnings("ignore")


def transport_task(a, b, c):
    def prepare(a, b):
        m = len(a)
        n = len(b)
        h = np.diag(np.ones(n))
        v = np.zeros((m, n))
        v[0] = 1
        for i in range(1, m):
            h = np.hstack((h, np.diag(np.ones(n))))
            k = np.zeros((m, n))
            k[i] = 1
            v = np.hstack((v, k))
        return np.vstack((h, v)).astype(int), np.hstack((b, a))

    a = np.copy(a)
    b = np.copy(b)
    c = np.copy(c)
    if a.sum() > b.sum():
        b = np.hstack((b, [a.sum() - b.sum()]))
        c = np.hstack((c, np.zeros(len(a)).reshape(-1, 1)))
    elif a.sum() < b.sum():
        a = np.hstack((a, [b.sum() - a.sum()]))
        c = np.vstack((c, np.zeros(len(b))))
    m = len(a)
    n = len(b)
    aa, bb = prepare(a, b)
    res = linprog(
        c.reshape(1, -1), A_eq=aa, b_eq=bb, bounds=(0, None), method="simplex"
    )

    return res.x.astype(int).reshape(m, n)


# a = np.array([90, 60, 90])
# b = np.array([40, 60, 40, 50, 50])
# c = np.array([[6, 4, 3, 4, 2], [3, 6, 4, 9, 2], [3, 1, 2, 2, 6]])

a = np.array([90, 60, 90])
b = np.array([40, 60, 40, 100])
c = np.array([[6, 4, 3, 4], [3, 6, 4, 9], [3, 1, 2, 2]])

summa = 0
final_array = transport_task(a,b,c)
for i in range(c.shape[0]):
    for j in range(c.shape[1]):
        summa += final_array[i][j]*c[i][j]
print(f"Стоимость проекта: {summa}")
print(f"Матрица перевозок:\n {final_array}")


