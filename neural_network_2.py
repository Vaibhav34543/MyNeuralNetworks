import random

w = random.random()
b = random.random()

x = [1, 2, 3, 4, 5]
y = [3, 1, -1, -3, -5]

LENGTH = len(x)
LR = 0.05

for i in range(500):

    buffer_weight = 0
    buffer_bias = 0
    loss = 0

    for j in range(LENGTH):

        output = w * x[j] + b

        loss += (output - y[j]) ** 2

        buffer_weight += 2 * (output - y[j]) * x[j]
        buffer_bias += 2 * (output - y[j])

    buffer_weight /= LENGTH
    buffer_bias /= LENGTH
    loss /= LENGTH

    w -= LR * buffer_weight
    b -= LR * buffer_bias

    if i % 50 == 0:
        print(f"Iteration {i}")
        print(f"Weight: {w}")
        print(f"Bias: {b}")
        print(f"Loss: {loss}")
        print()

print(f"FINAL: {w}x + {b}")
print()
