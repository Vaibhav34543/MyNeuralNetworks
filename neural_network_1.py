import random
    
weight = random.random()
bias = random.random()

x = 1
y = 5

c = 0
for i in range(20):
    c+=1
    output = weight*x + bias
    print(f"Prediction {c}:\t\t{output}")

    loss = pow((y - output), 2)

    lr = 0.3
    weight = weight - lr*(2*weight*pow(x, 2) - 2*y*x)
    bias = bias - lr*(2*(output - y)*x)

print(f"\nOutput = {weight}x + {bias}")

