def gen_arg(arg, n_workers):
    for _ in range(n_workers):
        yield arg
    while True:
        yield None
