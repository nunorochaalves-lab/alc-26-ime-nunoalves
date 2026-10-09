"""Questao 05 - Algebra Linear Computacional. Aluno: Nuno Alves."""
import numpy as np


def resolve_lu(A, b):
    """Constroi A = LU sem pivotamento e retorna L, U, x, com Ax = b."""
    # Copias em ponto flutuante: os dados originais nao sao alterados.
    U = np.array(A, dtype=float)
    b = np.array(b, dtype=float)
    if U.ndim != 2 or U.shape[0] != U.shape[1] or U.shape[0] == 0:
        raise ValueError("A deve ser uma matriz quadrada nao vazia.")
    n = U.shape[0]
    if b.ndim != 1 or len(b) != n:
        raise ValueError("b deve ser um vetor com n elementos.")
    L = np.eye(n)

    # 1. Eliminacao: U guarda as linhas resultantes; L, os multiplicadores.
    for k in range(n):
        if U[k, k] == 0:
            raise Exception(
                "Pivo nulo: utilize uma funcao alternativa com pivotamento."
            )
        for i in range(k + 1, n):
            multiplicador = U[i, k] / U[k, k]
            L[i, k] = multiplicador
            for j in range(k + 1, n):
                U[i, j] = U[i, j] - multiplicador * U[k, j]
            U[i, k] = 0.0

    # 2. Substituicao progressiva: resolve Ly = b, de cima para baixo.
    y = np.zeros(n)
    for i in range(n):
        soma = 0.0
        for j in range(i):
            soma += L[i, j] * y[j]
        y[i] = b[i] - soma  # A diagonal de L e igual a 1.

    # 3. Substituicao regressiva: resolve Ux = y, de baixo para cima.
    x = np.zeros(n)
    for i in range(n - 1, -1, -1):
        soma = 0.0
        for j in range(i + 1, n):
            soma += U[i, j] * x[j]
        x[i] = (y[i] - soma) / U[i, i]

    return L, U, x


if __name__ == "__main__":
    A = np.array([[2, 1, 1], [4, 3, 3], [8, 7, 9]], dtype=float)
    b = np.array([7, 18, 44], dtype=float)
    L, U, x = resolve_lu(A, b)
    print("L =\n", L)
    print("U =\n", U)
    print("x =", x)
