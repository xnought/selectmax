def symax(x, axis=0, eta=1):
    sizes = x.abs()
    return sizes / (eta + sizes.sum(axis=axis))
