import numpy as np
import matplotlib.pyplot as plt
y = np.asarray([0.4, 3.9, 1.3])
puv = np.asarray([1.8, 2.7, 3.1])
ci = np.asarray([0.3, 1.7, 4.3])
piuv = puv - ci

d1 = (np.linalg.norm(puv-ci)) * (np.linalg.norm(puv-ci)) * (np.linalg.norm(puv-y)) * (np.linalg.norm(puv-y))
d2 = np.dot((puv-ci), (puv-y)) * np.dot((puv-ci), (puv-y))

d1v = (np.linalg.norm(puv-ci)) * (np.linalg.norm(puv-ci)) * (np.dot(puv, puv)-2*np.dot(puv, y)+np.dot(y, y))

P = np.matmul(np.asmatrix(piuv).transpose(), np.asmatrix(piuv))
d2v = np.matmul(np.asmatrix(puv), np.matmul(P, np.asmatrix(puv).transpose())) - 2*np.matmul(np.asmatrix(puv), np.matmul(P, np.asmatrix(y).transpose())) + np.matmul(np.asmatrix(y), np.matmul(P, np.asmatrix(y).transpose()))
l = d1*d1 - d1v
print('d')

sigma = 0.06
curve = []
for i in range(-1000, 1000):
    ii = i/100.0
    curve.append(np.exp(-ii*ii/(2*sigma**2)) * (ii/(sigma)))
    #curve.append(np.exp(-ii * ii / (2 * sigma ** 2)) * np.sin(ii/sigma))
    #curve.append(np.exp(-ii*ii/(sigma**2)) - 1*np.abs(ii*ii/(sigma**2))*np.exp(-ii*ii/(sigma**2)))
plt.plot(curve)
plt.show()