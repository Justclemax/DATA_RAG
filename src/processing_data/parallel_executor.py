"""
parallel_executor.py

Helper simple pour paralléliser des tâches CPU-bound en Python
Utilise concurrent.futures.ProcessPoolExecutor.

Usage:
  from parallel_executor import parallel_map
  results = parallel_map(func, iterable, max_workers=4, chunksize=1, show_progress=True)

Le module essaie d'utiliser tqdm si disponible pour afficher la progression.
"""
from concurrent.futures import ProcessPoolExecutor, as_completed
import os
from typing import Callable, Iterable, List, Any, Optional


def _with_tqdm(iterator, total=None, desc=None):
    try:
        from tqdm.auto import tqdm
        return tqdm(iterator, total=total, desc=desc)
    except Exception:
        return iterator


def parallel_map(func: Callable[[Any], Any],
                 iterable: Iterable[Any],
                 max_workers: Optional[int] = None,
                 chunksize: int = 1,
                 show_progress: bool = False) -> List[Any]:
    """
    Applique `func` à chaque élément de `iterable` en parallèle (processes).

    Args:
        func: fonction picklable à exécuter dans chaque process.
        iterable: collection d'éléments à traiter.
        max_workers: nombre de workers; défaut = os.cpu_count() ou 1.
        chunksize: taille des lots envoyés aux workers (améliore la perf pour petites tâches).
        show_progress: si True et si tqdm installé, affiche une barre de progression.

    Returns:
        Liste des résultats dans le même ordre que l'itérable fourni.
    """
    data = list(iterable)
    n = len(data)
    if max_workers is None:
        max_workers = max(1, (os.cpu_count() or 1) - 0)

    results = [None] * n

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(func, item): idx for idx, item in enumerate(data)}

        iterator = _with_tqdm(as_completed(futures), total=n, desc="Processing") if show_progress else as_completed(futures)

        for future in iterator:
            idx = futures[future]
            try:
                results[idx] = future.result()
            except Exception as exc:
                results[idx] = exc

    return results
