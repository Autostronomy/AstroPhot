from pipeline import Pipeline

P = Pipeline()

res = P(ap_dummy = 'test')

print(res)


P2 = Pipeline()

res2 = P2(ap_dummy = ['test1', 'test2', 'test3'])

print(res2)
