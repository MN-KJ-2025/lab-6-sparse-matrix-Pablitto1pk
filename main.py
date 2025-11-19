# =================================  TESTY  ===================================
# Testy do tego pliku zostały podzielone na dwie kategorie:
#
#  1. `..._invalid_input`:
#     - Sprawdzające poprawną obsługę nieprawidłowych danych wejściowych.
#
#  2. `..._correct_solution`:
#     - Weryfikujące poprawność wyników dla prawidłowych danych wejściowych.
# =============================================================================
import numpy as np
import scipy as sp


def is_diagonally_dominant(A: np.ndarray | sp.sparse.csc_array) -> bool | None:
    """Funkcja sprawdzająca czy podana macierz jest diagonalnie zdominowana.

    Args:
        A (np.ndarray | sp.sparse.csc_array): Macierz A (m,m) podlegająca 
            weryfikacji.
    
    Returns:
        (bool): `True`, jeśli macierz jest diagonalnie zdominowana, 
            w przeciwnym wypadku `False`.
        Jeżeli dane wejściowe są niepoprawne funkcja zwraca `None`.
    """

    if A is None:
        return None
    if not isinstance(A, (np.ndarray, sp.sparse.csc_array)):
        return None

    if A.ndim != 2:
        return None

    m, n = A.shape

    if m != n:
        return None

    if isinstance(A, np.ndarray):
        abs_A = np.abs(A)
        row_sum_abs = np.sum(abs_A, axis=1)
        diag_abs = np.abs(np.diag(A))

    else:
        abs_A = A.copy()
        abs_A.data = np.abs(abs_A.data)
        row_sum_abs = np.array(abs_A.sum(axis=1)).ravel()
        diag_abs = np.abs(A.diagonal())

    off_diag_sum = row_sum_abs - diag_abs

    return bool(np.all(diag_abs > off_diag_sum))

def residual_norm(A: np.ndarray, x: np.ndarray, b: np.ndarray) -> float | None:
    """Funkcja obliczająca normę residuum dla równania postaci: 
    Ax = b.

    Args:
        A (np.ndarray): Macierz A (m,n) zawierająca współczynniki równania.
        x (np.ndarray): Wektor x (n,) zawierający rozwiązania równania.
        b (np.ndarray): Wektor b (m,) zawierający współczynniki po prawej 
            stronie równania.
    
    Returns:
        (float): Wartość normy residuum dla podanych parametrów.
        Jeżeli dane wejściowe są niepoprawne funkcja zwraca `None`.
    """
    if not isinstance(x, np.ndarray) or not isinstance(b, np.ndarray):
        return None

    if not isinstance(A, (np.ndarray, sp.sparse.spmatrix)):
        return None
    m, n = A.shape

    if b.shape != (m,) or x.shape != (n,):
        return None

    r = b - A @ x

    return float(np.linalg.norm(r))
