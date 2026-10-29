"""provenance stage. Explicit run-scoped state; import has no GPU or I/O side effects."""
from types import MethodType

def run(runtime):
    runtime.SOURCE_HASH = '1bb31a5bf5c1fd5990e6a317ae33e1b981a3b5e17217bf8d9c377bd22f6d44df'
