import numpy as np
import matplotlib.pyplot as plt
x=np.array([[0,0],[0,1],[1,0],[1,1]])
y=np.array([[0],[1],[1],[0]])
w1=np.random.rand(2,3)
b1=np.zeros((1,3))
w2=np.random.rand(3,1)
b2=np.zeros((1,1))
def sigmoid(x):
    return 1/(1+np.exp(-x))
def sigmoid_derivative(x):
    return x*(1-x)
for epoch in range (10000):
    z1=np.dot(x,w1)+b1
    a1=sigmoid(z1)
    z2=np.dot(a1,w2)+b2
    a2=sigmoid(z2)

    loss=np.mean((y-a2)**2)
    da2=2*(a2-y)
    dz2=da2*sigmoid_derivative(a2)
    dw2=np.dot(a1.T,dz2)
    db2=np.sum(dz2,axis=0,keepdims=True)

    da1=np.dot(dz2,w2.T)
    dz1=da1*sigmoid_derivative(a1)
    dw1=np.dot(x.T,dz1)
    db1=np.sum(dz1,axis=0)
    
    w2-=0.1*dw2
    b2-=0.1*db2
    w1-=0.1*dw1
    b1-=0.1*db1
    if epoch %1000==0:
        print(f"Epoch {epoch},Loss: {loss}")
xx,yy=np.meshgrid(np.linspace(0,1,100),np.linspace(0,1,100))
grid=np.c_[xx.ravel(),yy.ravel()]

z1=np.dot(grid,w1)+b1
a1=sigmoid(z1)
z2=np.dot(a1,w2)+b2
a2=sigmoid(z2)

z=a2.reshape(xx.shape)

plt.figure(figsize=(6,5))
plt.contourf(xx,yy,z,levels=50,cmap="coolwarm",alpha=0.6)
plt.scatter(x[:,0],x[:,1],c=y[:,0],cmap="coolwarm",edgecolor="k",s=100)
plt.title("XOR Decision Boundary after Training")
plt.xlabel("x1")
plt.ylabel("x2")
plt.colorbar(label="Output Value")
plt.show()
