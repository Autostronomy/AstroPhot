from autoprof.pipeline import build_pipeline
from autoprof.utils.state import State

P = build_pipeline()

res = P(State(ap_dummy = 'test'))

print(res.options.options)


P2 = build_pipeline()

res2 = P2(State(ap_dummy = ['test1', 'test2', 'test3']))

print(list(res.options.options for res in res2))
