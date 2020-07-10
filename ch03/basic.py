import tensorflow as tf

@tf.function
def square_pos(x):
    if x > 0:
        x = x * x
    else:
        x = x * -1
    return x

print(square_pos(tf.constant(2)))
print(square_pos)

def square_pos(x):
    if x > 0:
        x = x * x
    else:
        x = x * -1
    return x

print(square_pos(tf.constant(2)))
print(square_pos)
