import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.font_manager import FontProperties

ip1 = open('1gp_plot.txt','r')
X_input=[]
Y_input_1=[]

for line in ip1.readlines():
	line = line.strip()
	X,Y = line.split(" ")
	X_input.append(X)
	Y_input_1.append(Y)

Y_input_2 = []
ip2 = open('2gp_plot.txt','r')
for line in ip2.readlines():
	line = line.strip()
	X,Y = line.split(" ")
	Y_input_2.append(Y)


Y_input_3 = []
ip3 = open('3gp_plot.txt','r')
for line in ip3.readlines():
	line = line.strip()
	X,Y = line.split(" ")
	Y_input_3.append(Y)

Y_input_4 = []
ip4 = open('4gp_plot.txt','r')
for line in ip4.readlines():
	line = line.strip()
	X,Y = line.split(" ")
	Y_input_4.append(Y)


plt.title('f1 score vs training data set size')
plt.grid(True)


plt.scatter(X_input,Y_input_1,color='r')
plt.scatter(X_input,Y_input_2,color='g')
plt.scatter(X_input,Y_input_3,color='b')
plt.scatter(X_input,Y_input_4,color='y')

fontP = FontProperties()
fontP.set_size('small')

red_patch = mpatches.Patch(color='red', label='Naive Bayes')
green_patch = mpatches.Patch(color='green', label='Logistic Regression')
blue_patch = mpatches.Patch(color='blue', label='SVM')
yellow_patch = mpatches.Patch(color='yellow', label='Random Forest')

plt.legend(handles=[red_patch,green_patch,blue_patch,yellow_patch],prop=fontP,loc=4)

#plt.axis([800, 1200, 0, 0.07])
plt.ylabel('Macro averaged F1 score')
plt.xlabel('Training data size')

plt.show()

