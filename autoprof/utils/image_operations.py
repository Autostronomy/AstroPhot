import numpy as np


def blockwise_sum(img, block):

    block = np.asanyarray(block)
    sh = np.column_stack([img.shape//block, block]).ravel()
    return img.reshape(sh).sum(tuple(range(1, 2*img.ndim, 2)))

if __name__ == "__main__":
    print("testing")
    a = np.random.randint(0, 5, (16,20))
    
    print(a)

    print(blockwise_sum(a, (2,2)))
    
    print(blockwise_sum(a, (2,4)))
