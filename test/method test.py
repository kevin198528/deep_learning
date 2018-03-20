class Dog(object):
    n = 0

    def __init__(self, name):
        self.name = name

    @classmethod
    def eat(cls, food):
        print("%s eating %s" % (cls.n, food))


if __name__ == '__main__':
    Dog.eat(food='bone')

