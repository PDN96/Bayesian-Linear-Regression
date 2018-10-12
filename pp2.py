import math
import random
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import csv
from numpy import genfromtxt

#return data as a numpy array
def get_data(filename):
    my_data = genfromtxt(filename, delimiter=',')
    return my_data

#compute the MSE
def mean_square(x, y, w):
    sum = 0
    for i in range(len(y)):
        sum += ((np.dot(x[i].T,w) - y[i])**2)
    return sum/len(y)

#l is the lambda varying from 1 to 150
def regression(x, y, l):
    r = list()
    for i in range(l):
        t = np.dot(x.T, x)
        a = t + i * np.identity(len(t[0]))
        w = np.dot(np.dot(np.linalg.inv(a), x.T), y)
        r.append(w)
    return r

#function to calculate MSE for the file passed to it
def calc_mse(files):
    temp_train = get_data(files[0])
    temp_trainr = get_data(files[1])
    temp_test = get_data(files[2])
    temp_testr = get_data(files[3])
    temp_w = regression(temp_train, temp_trainr, 151)
    temp_result = []
    temp_resultr = []
    for w in temp_w:
        b1 = mean_square(temp_train, temp_trainr, w)
        b2 = mean_square(temp_test, temp_testr, w)
        temp_result.append(b1)
        temp_resultr.append(b2)
    return temp_result, temp_resultr, temp_train, temp_trainr, temp_test, temp_testr

#variable to assign names to graphs
global name
name = 1
lam = []
for i in range(151):
    lam.append(i)

#function to plot the graphs
def plotting(train_mse, test_mse, file_name):
    plt.figure()
    train_file = file_name[0].split('.')[0]
    test_file = file_name[2].split('.')[0]
    plt.plot(lam, train_mse, 'b-', label="" + train_file)
    plt.plot(lam, test_mse, 'r-', label="" + test_file)
    plt.legend()
    plt.title("MSE as a function of Lambda")
    plt.ylabel("MSE")
    plt.xlabel("Lambda value")
    global name
    plt.savefig('' + str(name) + '.png')
    name += 1

#list of lists of file names
all_files = [['train-100-10.csv', 'trainR-100-10.csv', 'test-100-10.csv', 'testR-100-10.csv'],
               ['train-100-100.csv', 'trainR-100-100.csv', 'test-100-100.csv', 'testR-100-100.csv'],
               ['train-1000-100.csv', 'trainR-1000-100.csv', 'test-1000-100.csv', 'testR-1000-100.csv'],
               ['train-crime.csv', 'trainR-crime.csv', 'test-crime.csv', 'testR-crime.csv'],
               ['train-wine.csv', 'trainR-wine.csv', 'test-wine.csv', 'testR-wine.csv']]

#getting MSEs of all files
result_100_10, resultr_100_10, train_100_10, trainr_100_10, test_100_10, testr_100_10 = calc_mse(all_files[0])
result_100_100, resultr_100_100, train_100_100, trainr_100_100, test_100_100, testr_100_100 = calc_mse(all_files[1])
result_1000_100, resultr_1000_100, train_1000_100, trainr_1000_100, test_1000_100, testr_1000_100 = calc_mse(all_files[2])
result_crime, resultr_crime, train_crime, trainr_crime, test_crime, testr_crime = calc_mse(all_files[3])
result_wine, resultr_wine, train_wine, trainr_wine, test_wine, testr_wine = calc_mse(all_files[4])

#plotting the results of all files
plotting(result_100_10, resultr_100_10, all_files[0])
plotting(result_100_100, resultr_100_100, all_files[1])
plotting(result_1000_100, resultr_1000_100, all_files[2])
plotting(result_crime, resultr_crime, all_files[3])
plotting(result_wine, resultr_wine, all_files[4])

print("PART 1")

l_min = np.argmin(resultr_100_10)
print("\nDataset -100-10:")
print("Minimum MSE found at lamda " ,l_min, " : " , resultr_100_10[l_min] )

l_min = np.argmin(resultr_100_100)
print("\nDataset -100-100:")
print("Minimum MSE found at lamda " ,l_min, " : " , resultr_100_100[l_min] )

l_min = np.argmin(resultr_1000_100)
print("\nDataset -1000-100:")
print("Minimum MSE found at lamda " ,l_min, " : " , resultr_1000_100[l_min] )

l_min = np.argmin(resultr_crime)
print("\nDataset crime:")
print("Minimum MSE found at lamda " ,l_min, " : " , resultr_crime[l_min] )

l_min = np.argmin(resultr_wine)
print("\nDataset wine:")
print("Minimum MSE found at lamda " ,l_min, " : " , resultr_wine[l_min] )


########################################################################
#                       PART 2                                         #
########################################################################

train_size = []
for i in range(10, 801, 10):
    train_size.append(i)
c = range(0, 1000)

#random sampling
def make_matrix(k):
    train_list = []
    tlabel_list = []
    numbers = random.sample(c, k)
    for num in numbers:
        train_list.append(train_1000_100[num])
        tlabel_list.append(trainr_1000_100[num])
    return (np.array(train_list), np.array(tlabel_list))

#z is the lambda value we are interested in
def single_reg(x, y, z):
    t = np.dot(x.T, x)
    a = np.add(z * np.identity(len(t[0])), t)
    w = np.dot(np.dot(np.linalg.inv(a), x.T), y)
    return w

def repeat(z, k):
    sum = 0
    for i in range(10):
        data, label = make_matrix(k)
        w = single_reg(data, label, z)
        mse = mean_square(test_1000_100, testr_1000_100, w)
        sum += mse
    return sum / 10

def get_all_mse(z):
    all_mse = list()
    for size in train_size:
        all_mse.append(repeat(z, size))
    return all_mse

l_small = 5
l_perfect = np.argmin(resultr_1000_100)
l_big = 100

mse_1000_100_small = get_all_mse(l_small)
mse_1000_100_right = get_all_mse(l_perfect)
mse_1000_100_big = get_all_mse(l_big)

#plotting
plt.figure()
plt.plot(train_size, mse_1000_100_small, 'r-', label="lambda = 5")
plt.plot(train_size, mse_1000_100_right, 'b-', label="lambda = 27")
plt.plot(train_size, mse_1000_100_big, 'k-', label="lambda = 100")

plt.xlim(0, 800)
plt.ylim(0, 70)
plt.legend()
plt.title("Part 2: Learning Curves")
plt.ylabel("MSE")
plt.xlabel("Training Size")
plt.savefig('6.png')

########################################################################
#                       PART 3                                         #
########################################################################

#iterative approach to find the best alpha and beta
def get_a_b(x ,y):
    new_a = random.randrange(1 ,10)
    new_b = random.randrange(1 ,10)
    a = 11
    b = 11
    n = len(y)
    while abs(new_a - a) > 0.000001 and abs(new_b - b) > 0.000001:
        a = new_a
        b = new_b
        temp = np.dot(x.T ,x)
        s_n_i = np.add(a * np.identity(len(temp[0])), b * temp)
        s_n = np.linalg.inv(s_n_i)
        m_n = b * np.dot(np.dot(s_n, x.T), y)
        lamb = np.linalg.eigvals(s_n_i) - a
        r = 0
        for l in lamb:
            r += l / (a + l)
        new_a = r / np.dot(m_n.T, m_n)
        sum = 0
        for i in range(n):
            sum += (y[i] - np.dot(m_n.T, x[i]))**2
        new_b = (n - r) / sum

    return new_a, new_b


def w_map(x, y):
    a, b = get_a_b(x, y)
    lambd = a / b
    t = np.dot(x.T, x)
    s_n = np.linalg.inv(np.add(a * np.identity(len(t[0])), b * t))
    w_max = b * np.dot(np.dot(s_n, x.T), y)

    return w_max, lambd, a, b

print("PART 2 is plotted and saved")
print("\nPART 3")

w, lambd_100_10, a_100_10, b_100_10 = w_map(train_100_10, trainr_100_10)
w_100_10 = mean_square(test_100_10, testr_100_10, w)
print("\nDataset test-100-10: " ,w_100_10)
print("Found at (lambda, alpha, beta) : " ,lambd_100_10, a_100_10, b_100_10)

w, lambd_100_100, a_100_100, b_100_100 = w_map(train_100_100, trainr_100_100)
w_100_100 = mean_square(test_100_100, testr_100_100, w)
print("\nDataset test-100-100: " ,w_100_100)
print("Found at (lambda, alpha, beta) : " ,lambd_100_100, a_100_100, b_100_100)

w, lambd_1000_100, a_1000_100, b_1000_100 = w_map(train_1000_100, trainr_1000_100)
w_1000_100 = mean_square(test_1000_100, testr_1000_100, w)
print("\nDataset test-1000-100: " ,w_1000_100)
print("Found at (lambda, alpha, beta) : " ,lambd_1000_100, a_1000_100, b_1000_100)

w, lambd_crime, a_crime, b_crime = w_map(train_crime, trainr_crime)
w_crime = mean_square(test_crime, testr_crime, w)
print("\nDataset crime: " ,w_crime)
print("Found at (lambda, alpha, beta) : " ,lambd_crime, a_crime, b_crime)

w, lambd_wine, a_wine, b_wine = w_map(train_wine, trainr_wine)
w_wine = mean_square(test_wine, testr_wine, w)
print("\nDataset wine: " ,w_wine)
print("Found at (lambda, alpha, beta) : " ,lambd_wine, a_wine, b_wine)


########################################################################
#                       PART 4                                         #
########################################################################

#reading data in a list for files f3 and f5 only
def open_matrix2(filename):
    list_data = []
    with open(filename) as csvfile:
        data = csv.reader(csvfile, delimiter = ",")
        for row in data:
            list_data.append(row)
        return list_data

#converting data to vectors for easy manipulation
def train_generate(list_data ,d):
    result = []
    for data in list_data:
        vector = []
        for i in range(d + 1):
            vector.append(math.pow(float(data[0]) ,i))
        result.append(vector)

    return np.array(result ,dtype = float)

#iterative approach to find alpha beta for task 4
def a_b_4(x ,y):
    new_a = random.randrange(1 ,10)
    new_b = random.randrange(1 ,10)
    a = 11
    b = 11
    n = len(y)

    while abs(new_a - a) > 0.0000001 and abs(new_b - b) > 0.0000001:
        a = new_a
        b = new_b
        t = np.dot(x.T ,x)

        s_n_1 = np.add(a * np.identity(len(t[0])) ,b * t)
        s_n = np.linalg.inv(s_n_1)
        m_n = b * np.dot(np.dot(s_n, x.T), y)

        evige_value = np.linalg.eigvals(s_n_1)
        lamb_da = evige_value - a

        r = 0
        for lamb in lamb_da:
            r += lamb / (a + lamb)

        new_a = r / np.dot(m_n.T, m_n)[0][0]
        sum = 0
        for i in range(n):
            sum += math.pow((y[i] - np.dot(m_n.T, x[i])), 2)
        new_b = (n - r) / sum

    return new_a, new_b

#calculating evidnenec for task 4
def evidence(x, y):
    sum = 0
    a, b = a_b_4(x, y)
    n, m = x.shape
    sum += (m / 2) * np.log(a)
    sum += (n / 2) * np.log(b)
    t = np.dot(x.T, x)
    s_n = np.linalg.inv(np.add(a * np.identity(len(t[0])), b * t))
    m_n = b * np.dot(np.dot(s_n, x.T), y)
    c = y - np.dot(x, m_n)
    e_m = (b / 2.0) * (np.linalg.norm(c) ** 2) + (a / 2.0) * np.dot(m_n.T, m_n)
    sum -= e_m
    A = a * np.identity(m) + b * np.dot(x.T, x)
    A_det = np.linalg.det(A)
    sum -= (1 / 2) * np.log(A_det)
    sum -= (n / 2) * np.log(2 * math.pi)

    return sum

def w_map(x, y):
    a, b = a_b_4(x, y)
    t = np.dot(x.T, x)
    s_n = np.linalg.inv(a * np.identity(len(t[0])) + b * t)
    w_max = b * np.dot(s_n, np.dot(x.T, y))
    return w_max

f_files = [["train-f3.csv", "trainr-f3.csv", "test-f3.csv", "testR-f3.csv"],
           ["train-f5.csv", "trainr-f5.csv", "test-f5.csv", "testR-f5.csv"]]

train_f3 = open_matrix2(f_files[0][0])
trainr_f3 = open_matrix2(f_files[0][1])
new_trainr_f3 = np.array(trainr_f3, dtype=float)

test_f3 = open_matrix2(f_files[0][2])
testr_f3 = open_matrix2(f_files[0][3])
new_testr_f3 = np.array(testr_f3, dtype=float)

evi_list_3 = []
w_mean_list_f3 = []
w_noregular_list_f3 = []

for i in range(1, 11):
    new_train_f3 = train_generate(train_f3, i)
    new_test_f3 = train_generate(test_f3, i)
    w_max_f3 = w_map(new_train_f3, new_trainr_f3)
    w_mean_squre = mean_square(new_test_f3, new_testr_f3, w_max_f3)
    w_mean_list_f3.append(w_mean_squre)
    w_noregular = np.dot(np.dot(np.linalg.inv(np.dot(new_train_f3.T, new_train_f3)), new_train_f3.T), new_trainr_f3)
    w_mean_square_no = mean_square(new_test_f3, new_testr_f3, w_noregular)
    w_noregular_list_f3.append(w_mean_square_no)
    evi_value = evidence(new_train_f3, new_trainr_f3)
    evi_list_3.append(evi_value[0])

train_f5 = open_matrix2(f_files[1][0])
trainr_f5 = open_matrix2(f_files[1][1])
new_trainr_f5 = np.array(trainr_f5, dtype=float)

test_f5 = open_matrix2(f_files[1][2])
testr_f5 = open_matrix2(f_files[1][3])
new_testr_f5 = np.array(testr_f5, dtype=float)

evi_list_5 = []
w_mean_list_f5 = []
w_noregular_list_f5 = []

for i in range(1, 11):
    new_train_f5 = train_generate(train_f5, i)
    new_test_f5 = train_generate(test_f5, i)
    w_max_f5 = w_map(new_train_f5, new_trainr_f5)
    w_mean_squre = mean_square(new_test_f5, new_testr_f5, w_max_f5)
    w_mean_list_f5.append(w_mean_squre)
    w_noregular = np.dot(np.dot(np.linalg.inv(np.dot(new_train_f5.T, new_train_f5)), new_train_f5.T), new_trainr_f5)
    w_mean_squre_no = mean_square(new_test_f5, new_testr_f5, w_noregular)
    w_noregular_list_f5.append(w_mean_squre_no)
    evi_value = evidence(new_train_f5, new_trainr_f5)
    evi_list_5.append(evi_value[0])

print("PART 4")

print("\nDataset f3:")
for i in range(len(w_noregular_list_f3)):
    print("\nMSE using unregularized for dimension " , i+1, " : ", w_noregular_list_f3[i])
    print("MSE using bayesian for dimension " , i+1, " : ", w_mean_list_f3[i])
    print("Log evidence :" , evi_list_3[i])

print("\nDataset f5:")
for i in range(len(w_noregular_list_f5)):
    print("\nMSE using unregularized for dimension " , i+1, " : ", w_noregular_list_f5[i])
    print("MSE using bayesian for dimension " , i+1, " : ", w_mean_list_f5[i])
    print("Log evidence :" , evi_list_5[i])

d = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

fig, a1 = plt.subplots()
a2 = a1.twinx()
a1.plot(d, w_noregular_list_f3, 'b-', label="MSE of non regularized model of f3")
a1.plot(d, w_mean_list_f3, 'k-', label="MSE of bayesian model of f3")
a1.set_ylabel("MSE")
a1.set_xlabel("Degree size")
a1.legend(["Bayesian Model", "Unregularized Regression"])
a2.plot(d, evi_list_3, 'r-', label="Log evidence of f3")
a2.set_ylabel("Log evidence")
a2.legend(["Log Evidence"])
plt.savefig('7.png')

plt.figure()
plt.plot(d, evi_list_3, 'r-', label="Log evidence of f3")
plt.legend()
plt.ylabel("Evidence")
plt.xlabel("Degree size ")
plt.savefig('8.png')

fig, a1 = plt.subplots()
a2 = a1.twinx()
a1.plot(d, w_noregular_list_f5, 'b-', label="MSE of non regularized model of f5")
a1.plot(d, w_mean_list_f5, 'k-', label="MSE of bayesian model of f5")
a1.set_ylabel("MSE")
a1.set_xlabel("Degree size")
a1.legend(["Bayesian Model", "Unregularized Regression"])
a2.plot(d, evi_list_5, 'r-', label="Log evidence of f5")
a2.set_ylabel("Log evidence")
a2.legend(["Log Evidence"])
plt.savefig('9.png')

plt.figure()
plt.plot(d, evi_list_5, 'r-', label="Log evidence of f5")
plt.legend()
plt.title("evidence as a function of degree size")
plt.ylabel("evidence value")
plt.xlabel("degree size ")
plt.savefig('10.png')
