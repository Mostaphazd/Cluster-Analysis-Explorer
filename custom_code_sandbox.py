"""
Sandboxed execution for Cluster Analysis Explorer's custom-code feature.

Replaces the previous bare `exec(self.custom_code, exec_namespace)` call in
cluster_explorer.py, which gave custom code full, unrestricted builtins
access (confirmed: unrestricted `import os; os.system(...)`, arbitrary file
read/write, and no timeout on infinite loops). See the manuscript's Security
Considerations section for the full writeup and verification table.

This module is intentionally kept separate from cluster_explorer.py and
imports only numpy at module scope. multiprocessing's Windows 'spawn' start
method re-imports whatever module defines the worker target function in the
child process; keeping that module free of tkinter/matplotlib/umap/genai
avoids paying their import cost on every custom-code execution.
"""

import ast
import builtins
import multiprocessing as mp

import numpy as np
import pandas as pd


class UnsafeCustomCodeError(Exception):
    pass


ALLOWED_MODULES = {"numpy", "sklearn", "pandas"}

FORBIDDEN_NAMES = {
    "__import__",
    "eval",
    "exec",
    "compile",
    "open",
    "input",
    "breakpoint",
    "globals",
    "locals",
    "vars",
    "getattr",
    "setattr",
    "delattr",
    "__builtins__",
}

SAFE_BUILTINS = {
    name: getattr(builtins, name)
    for name in (
        "abs",
        "all",
        "any",
        "bool",
        "dict",
        "enumerate",
        "float",
        "int",
        "len",
        "list",
        "max",
        "min",
        "print",
        "range",
        "round",
        "set",
        "sorted",
        "str",
        "sum",
        "tuple",
        "zip",
        "True",
        "False",
        "None",
        "ValueError",
        "TypeError",
        "KeyError",
        "IndexError",
        "Exception",
    )
}


def _validate_ast(code):
    tree = ast.parse(code, mode="exec")
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                root = alias.name.split(".")[0]
                if root not in ALLOWED_MODULES:
                    raise UnsafeCustomCodeError(
                        f"Import of '{alias.name}' is not on the allow-list "
                        f"({sorted(ALLOWED_MODULES)}) at line {node.lineno}."
                    )
        elif isinstance(node, ast.ImportFrom):
            root = (node.module or "").split(".")[0]
            if root not in ALLOWED_MODULES:
                raise UnsafeCustomCodeError(
                    f"Import from '{node.module}' is not on the allow-list "
                    f"({sorted(ALLOWED_MODULES)}) at line {node.lineno}."
                )
        if isinstance(node, ast.Name) and node.id in FORBIDDEN_NAMES:
            raise UnsafeCustomCodeError(
                f"Disallowed identifier '{node.id}' at line {node.lineno}."
            )
        if (
            isinstance(node, ast.Attribute)
            and node.attr.startswith("__")
            and node.attr.endswith("__")
        ):
            raise UnsafeCustomCodeError(
                f"Disallowed dunder attribute access '.{node.attr}' at line {node.lineno}."
            )
    return tree


def _allow_listed_import(name, globals=None, locals=None, fromlist=(), level=0):
    # The AST pre-check already rejects disallowed imports before this ever
    # runs; this is a second, independent enforcement point at the actual
    # import call, since Python's `import` statement requires *some*
    # `__import__` to be present in builtins to execute at all -- it cannot
    # simply be omitted.
    root = name.split(".")[0]
    if root not in ALLOWED_MODULES:
        raise UnsafeCustomCodeError(
            f"Import of '{name}' is not on the allow-list ({sorted(ALLOWED_MODULES)})."
        )
    return builtins.__import__(name, globals, locals, fromlist, level)


def _worker(code, X, n_samples, conn):
    try:
        tree = _validate_ast(code)
        restricted_builtins = dict(SAFE_BUILTINS)
        restricted_builtins["__import__"] = _allow_listed_import
        namespace = {
            "X": X,
            "n_samples": n_samples,
            "np": np,
            "pd": pd,
            "__builtins__": restricted_builtins,
        }
        exec(compile(tree, "<custom_code>", "exec"), namespace)
        if "labels" not in namespace:
            raise ValueError("Custom code must set 'labels' variable.")
        conn.send(("ok", np.asarray(namespace["labels"]).tolist()))
    except Exception as e:  # noqa: BLE001
        conn.send(("error", f"{type(e).__name__}: {e}"))
    finally:
        conn.close()


def run_custom_code_sandboxed(code, X, n_samples, timeout_sec=10.0):
    """Execute user-supplied custom clustering code in an AST-validated,
    builtins-restricted, time-limited child process. Raises
    UnsafeCustomCodeError on a rejected payload or TimeoutError on a hang;
    returns a numpy array of labels on success."""
    parent_conn, child_conn = mp.Pipe()
    proc = mp.Process(target=_worker, args=(code, X, n_samples, child_conn))
    proc.start()
    proc.join(timeout_sec)
    if proc.is_alive():
        proc.terminate()
        proc.join()
        raise TimeoutError(
            f"Custom code exceeded {timeout_sec}s execution limit and was terminated."
        )
    if parent_conn.poll():
        status, payload = parent_conn.recv()
        if status == "error":
            raise UnsafeCustomCodeError(payload)
        return np.array(payload)
    raise RuntimeError("Sandboxed custom code execution produced no result.")
