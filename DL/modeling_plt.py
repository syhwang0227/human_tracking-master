from matplotlib import pyplot as plt  # 데이터 시각화 패키지
# import matplotlib.pyplot as plt

# plt.plot([1, 2, 3], [110, 130, 120])

plt.plot(["Seoul", "Paris", "Seattle"], [30, 25, 55])
plt.xlabel("City")
plt.ylabel("Response")
plt.title("Result")
plt.legend("value")
plt.show()
