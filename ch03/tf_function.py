import tensorflow as tf
@tf.function
def square_pos(x):
    if x > 0:
        x = x * x
    else:
        x = x * -1
    return x

print(square_pos(tf.constant(2)))
print(square_pos(tf.constant(-3)))

# tf.function을 사용하지 않은 함수
def square_pos_python(x):
    if x > 0:
        x = x * x
    else:
        x = x * -1
    return x
print(square_pos_python(2))
print(square_pos_python(-3))