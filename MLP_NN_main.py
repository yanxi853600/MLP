# 針對XOR沒有隱藏層加入 hidden layer，解決非線性問題

from MLP_NN_module import NeuralNetwork
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
# 生成的数据
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y_or = np.array([[0], [1], [1], [1]])
y_and = np.array([[0], [0], [0], [1]])
y_xor = np.array([[0], [1], [1], [0]])

#NN_2-2-1:Input(2)  hidden(2) Output(1)
nn = NeuralNetwork([2, 2, 1], alpha=0.5)

#開始訓練，更新得到不斷迭代更新的weight矩陣
losses = nn.fit(X, y_xor, epochs=2000000)

for (x, target) in zip(X, y_xor):
	pred = nn.predict(x)[0][0]
	step = 1 if pred > 0.5 else 0
	print("data-{}, truth={}, pred={:.4f}, step={}"
		.format(x, target[0], pred, step))
 
plt.style.use("ggplot")
plt.figure()
plt.title("Data")
cm_dark = mpl.colors.ListedColormap(['g', 'b'])
plt.scatter(X[:, 0], X[:, 1], marker="o", c=y_xor.ravel(), cmap=cm_dark, s=80)
# print(testY)
 
plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0, len(losses)), losses)
plt.title("Training Loss")
plt.xlabel("Epoch #")
plt.ylabel("Loss")
plt.show()
 
print("W\n", nn.W)

print("\n結論: \n加入一層hidden layer後，可以很好解決非線性問題")