import sampler as S

smp = S.RandomSampler(n = 100, r_test = 0.2, r_train = 0.2, num_tries_inner = 2, test_crossval = True)

smp.test_split()
smp.sampling_split()
