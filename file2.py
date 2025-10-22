settings = {"a":1,"b":2}

def f(x,a,**settings):
    print(x,a,settings['b'])


f(9,**settings)