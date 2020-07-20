import matplotlib.pyplot as plt
def plt_draw(data, data_name, x_label, y_label):
    plt.plot(data, label=data_name)
    plt.xticks(range(len(data)))
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.legend()
    plt.show()

data1=[32, 64, 96, 128]
data2=[8.83, 17.12, 22.56, 55.16]
plt.plot(data2)
plt.xticks([0,1,2,3],data1)
plt.xlabel('batch size')
plt.ylabel('run time')
plt.show()