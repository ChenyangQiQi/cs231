def primes(nb_primes):
    primes = []
    if nb_primes > 1000:
        nb_primes = 1000
    n = 2
    len_p = 0
    i = 0
    while len_p < nb_primes:
        for i in primes:
            if n%i == 0:
                break
        else:
            primes.append(n)
            len_p+=1
        n+=1
    result_list = [prime for prime in primes[:len_p]]
    return result_list