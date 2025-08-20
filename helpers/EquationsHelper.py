import tensorflow as tf

class ArticleProblemsHelper:
    def __init__(self,f0,f1,g0,g1):
        self.f0 = f0
        self.f1 = f1
        self.g0 = g0
        self.g1 = g1

    def calculate(self,x,y):
        zero = tf.cast(0, tf.float64)
        one = tf.cast(1, tf.float64)
        a = (1 - x) * self.f0(y) + x*self.f1(y) + self.g0(x)
        b = (1 - x)*self.g0(zero) + x * self.g0(one)
        c = self.g1(x) - ((1-x) * self.g1(zero) + x * self.g1(one) )
        return a - b + y*c