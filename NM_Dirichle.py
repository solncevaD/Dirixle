import sys
import math
import copy
from interface import *
# Импортируем наш интерфейс из файла
from PyQt5.QtWidgets import QApplication, QMainWindow
import numpy as np
from numpy import float64
from PyQt5 import QtWidgets, QtGui, QtCore
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
class MyWin(QMainWindow, Ui_MainWindow):

    def __init__(self, parent=None, *args, **kwargs):
        QMainWindow.__init__(self, parent)
        self.setupUi(self)
        self.pushButton.clicked.connect(self.MyFunction_test)
        self.pushButton_2.clicked.connect(self.MyFunction_main)
        self.pushButton_3.clicked.connect(self.MyFunction_test_cut)
    def tau(self, f, n, m, start_sol):
        a = 0
        b = 2
        c = 0
        d = 1
        v = start_sol
        h = (b - a) / n
        k = (d - c) / m
        A = -2 * (1 / (h**2) + 1 / (k**2))
        Rs = []
        for i in range(n + 1):
            Rs.append([0] * (m + 1))
        ARs = []
        for i in range(n + 1):
            ARs.append([0] * (m + 1))     
        for i in range(1, m):
            for j in range(1, n):
                Rs[i][j] = f[i - 1][j - 1] + v[i][j] * A + (1 / h**2) * (v[i - 1][j] + v[i + 1][j]) + (1 / k**2) * (v[i][j + 1] + v[i][j - 1])
        Rs_zeros = []
        for i in range(n + 1):
            Rs_zeros.append([0] * (m + 1))
        for i in range(len(Rs)):
            for j in range(len(Rs[i])):
                Rs_zeros[i][j] = Rs[i][j]
        for i in range(1, m):
            Rs_zeros[0][i] = 0
            Rs_zeros[n][i] = 0
        for i in range(1, n):
            Rs_zeros[i][0] = 0
            Rs_zeros[i][m] = 0  
        Rs_zeros[0][0] = 0 
        Rs_zeros[0][m] = 0 
        Rs_zeros[n][0] = 0 
        Rs_zeros[n][m] = 0 
        for i in range(1, m):
            for j in range(1, n):
                ARs[i][j] = Rs_zeros[i][j] * A + (1 / h**2) * (Rs_zeros[i - 1][j] + Rs_zeros[i + 1][j]) + (1 / k**2) * (Rs_zeros[i][j + 1] + Rs_zeros[i][j - 1])
        denominator, numenator = 0, 0
        for i in range(1, m):
            for j in range(1, n):
                denominator += ARs[i][j]**2
                numenator += ARs[i][j] * Rs[i][j]
        res = -numenator / denominator
        return res, Rs
    def min_discrepancy(self, N, eps, n, m, start_sol, f):
        a = 0
        b = 2
        c = 0
        d = 1
        v = start_sol
        h = (b - a) / n
        k = (d - c) / m
        A = -2 * (1 / (h**2) + 1 / (k**2))
        t = 0
        r = 0
        count = 0
        while(True):
            eps_max = 0
            v_copy = []
            for i in range(n + 1):
                v_copy.append([0] * (m + 1))
            for i in range(len(v)):
                for j in range(len(v[i])):
                    v_copy[i][j] = v[i][j]
            t, r = self.tau(f, n, m, v_copy)
            for i in range(1, m):
                for j in range(1, n):
                    v_old = v[i][j]
                    v_new = v_copy[i][j] - t * (-f[i - 1][j - 1] - v_copy[i][j] * A - (1 / h**2) * (v_copy[i - 1][j] + v_copy[i + 1][j]) - (1 / k**2) * (v_copy[i][j + 1] + v_copy[i][j - 1]))
                    eps_cur = abs(v_old - v_new)
                    if eps_cur > eps_max:
                        eps_max = eps_cur
                    v[i][j] = v_new
            count += 1
            if eps_max < eps or count >= N:
                break
        resid = 0
        for i in range(len(r)):
            for j in range(len(r[i])):
                if abs(r[i][j]) > resid:
                    resid = abs(r[i][j])
        self.textBrowser_2.append("При решении Р.С. с помощью метода Минимальных Невязок с параметрами NMax = " + 
        str(N) + "\nи eps = " + str(eps) + " за S = " + str(count) + " итераций было получено решение \nс точностью eps max = " + 
            str(eps_max) + " и максимальной невязкой ||r|| =  " + str(resid))
        return v

    def tau_cut(self, f, n, m, start_sol):
        a = 0
        b = 2
        c = 0
        d = 1
        v = start_sol
        h = (b - a) / n
        k = (d - c) / m
        A = -2 * (1 / (h**2) + 1 / (k**2))
        Rs = []
        for i in range(int(n / 2) + 1):
            Rs.append([0] * (m + 1))
        for i in range(int(n / 2)):
            Rs.append([0] * (int(m / 2) + 1))
        ARs = []
        for i in range(int(n / 2) + 1):
            ARs.append([0] * (m + 1))
        for i in range(int(n / 2)):
            ARs.append([0] * (int(m / 2) + 1))   
        for i in range(1, int(m / 2)):
            for j in range(1, n):
               Rs[i][j] = f[i - 1][j - 1] + v[i][j] * A + (1 / h**2) * (v[i - 1][j] + v[i + 1][j]) + (1 / k**2) * (v[i][j + 1] + v[i][j - 1])
        for i in range(int(m / 2), m):
            for j in range(1, int(n / 2)):       
                Rs[i][j] = f[i - 1][j - 1] + v[i][j] * A + (1 / h**2) * (v[i - 1][j] + v[i + 1][j]) + (1 / k**2) * (v[i][j + 1] + v[i][j - 1])
        Rs_zeros = []
        for i in range(int(n / 2) + 1):
            Rs_zeros.append([0] * (m + 1))
        for i in range(int(n / 2)):
            Rs_zeros.append([0] * (int(m / 2) + 1))
        for i in range(len(start_sol)):
            for j in range(len(start_sol[i])):
                Rs_zeros[i][j] = Rs[i][j]
        for i in range(1, m):
            Rs_zeros[0][i] = 0

        for i in range (1, int(m / 2)):
            Rs_zeros[n][i] = 0

        for i in range(1, n):
            Rs_zeros[i][0] = 0
        for i in range(1, int(n / 2)):
            Rs_zeros[i][m] = 0
        for i in range(int(n / 2), n):
            Rs_zeros[i][int(m / 2)] = 0
        for i in range(int(m / 2), m):
            Rs_zeros[int(n / 2)][i] = 0              
        Rs_zeros[0][0] = 0 
        Rs_zeros[0][m] = 0 
        Rs_zeros[n][0] = 0 
        Rs_zeros[int(n / 2)][m] = 0
        Rs_zeros[n][int(m / 2)] = 0
        for i in range(1, int(m / 2)):
            for j in range(1, n):
                ARs[i][j] = Rs_zeros[i][j] * A + (1 / h**2) * (Rs_zeros[i - 1][j] + Rs_zeros[i + 1][j]) + (1 / k**2) * (Rs_zeros[i][j + 1] + Rs_zeros[i][j - 1])
        for i in range(int(m / 2), m):
            for j in range(1, int(n / 2)):   
                ARs[i][j] = Rs_zeros[i][j] * A + (1 / h**2) * (Rs_zeros[i - 1][j] + Rs_zeros[i + 1][j]) + (1 / k**2) * (Rs_zeros[i][j + 1] + Rs_zeros[i][j - 1])
        denominator, numenator = 0, 0
        for i in range(1, int(m / 2)):
            for j in range(1, n):
                denominator += ARs[i][j]**2
                numenator += ARs[i][j] * Rs[i][j]
        for i in range(int(m / 2), m):
            for j in range(1, int(n / 2)):  
                denominator += ARs[i][j]**2
                numenator += ARs[i][j] * Rs[i][j]
        res = -numenator / denominator
        return res, Rs

    def min_discrepancy_cut(self, N, eps, n, m, start_sol, f):
        a = 0
        b = 2
        c = 0
        d = 1
        v = []
        for i in range(int(n / 2) + 1):
            v.append([0] * (m + 1))
        for i in range(int(n / 2)):
            v.append([0] * (int(m / 2) + 1))
        for i in range(len(start_sol)):
            for j in range(len(start_sol[i])):
                v[i][j] = start_sol[i][j]    
        h = (b - a) / n
        k = (d - c) / m
        A = -2 * (1 / (h**2) + 1 / (k**2))
        t = 0
        r = 0
        count = 0
        while(True):
            eps_max = 0
            v_copy = []
            for i in range(int(n / 2) + 1):
                v_copy.append([0] * (m + 1))
            for i in range(int(n / 2)):
                v_copy.append([0] * (int(m / 2) + 1))
            for i in range(len(v)):
                for j in range(len(v[i])):
                    v_copy[i][j] = v[i][j]
            t, r = self.tau_cut(f, n, m, v_copy)
            for i in range(1, int(m / 2)):
                for j in range(1, n):
                    v_old = v[i][j]
                    v_new = v_copy[i][j] - t * (-f[i - 1][j - 1] - v_copy[i][j] * A - (1 / h**2) * (v_copy[i - 1][j] + v_copy[i + 1][j]) - (1 / k**2) * (v_copy[i][j + 1] + v_copy[i][j - 1]))
                    eps_cur = abs(v_old - v_new)
                    if eps_cur > eps_max:
                        eps_max = eps_cur
                    v[i][j] = v_new
            for i in range(int(m / 2), m):
                for j in range(1, int(n / 2)): 
                    v_old = v[i][j]
                    v_new = v_copy[i][j] - t * (-f[i - 1][j - 1] - v_copy[i][j] * A - (1 / h**2) * (v_copy[i - 1][j] + v_copy[i + 1][j]) - (1 / k**2) * (v_copy[i][j + 1] + v_copy[i][j - 1]))
                    eps_cur = abs(v_old - v_new)
                    if eps_cur > eps_max:
                        eps_max = eps_cur
                    v[i][j] = v_new 
            count += 1
            if eps_max < eps or count >= N:
                break
        resid = 0
        for i in range(len(r)):
            for j in range(len(r[i])):
                if abs(r[i][j]) > resid:
                    resid = abs(r[i][j])
        self.textBrowser_2.append("При решении Р.С. с помощью метода Минимальных невязок для области с вырезом с параметрами NMax = " + 
        str(N) + "\nи eps = " + str(eps) + " за S = " + str(count) + " итераций было получено решение \nс точностью eps max = " + 
            str(eps_max) + " и максимальной невязкой ||r|| =  " + str(resid))
        return v

    def zeidel_method(self, N, eps, n, m, start_sol, f):
        v_old, v_new = 0, 0
        eps_cur, eps_max, S = 0, 0, 0
        a = 0
        b = 2
        c = 0
        d = 1
        h2 = (n / (b - a))**2
        k2 = (m / (d - c))**2
        a2 = 2 * (h2 + k2)
        v = start_sol
        while(True):
            eps_max = 0
            for i in range(1, m):
                for j in range(1, n):
                    v_old = v[i][j]
                    v_new = (h2 * (v[i + 1][j] + v[i - 1][j]) + k2 * (v[i][j + 1] + v[i][j - 1]))
                    v_new += f[i - 1][j - 1]
                    v_new /= a2
                    eps_cur = abs(v_old - v_new)
                    if eps_cur > eps_max:
                        eps_max = eps_cur
                    v[i][j] = v_new
            S += 1
            if eps_max < eps or S >= N:
                break

        r = 0
        for i in range(1, m):
            for j in range(1, n):
                r_curr = a2 * v[i][j] - (v[i + 1][j] + v[i - 1][j]) * h2  - (v[i][j + 1] + v[i][j - 1]) * k2 - f[i - 1][j - 1]
                if abs(r_curr) > r:
                    r = abs(r_curr)
        self.textBrowser_2.append("При решении Р.С. с помощью метода Зейделя с параметрами NMax = " + 
        str(N) + "\nи eps = " + str(eps) + " за S = " + str(S) + " итераций было получено решение \nс точностью eps max = " + 
            str(eps_max) + " и максимальной невязкой ||r|| =  " + str(r))
        return v
    def acc_solution(self, x, y):
        return np.exp((np.sin(np.pi * x * y))**2)
    def f_test(self, x, y):
        return (2 * np.pi**2 * np.exp((np.sin(np.pi * x * y))**2) * ((2 * (np.sin(np.pi * y * x))**2 + 1) * (np.cos(np.pi * x * y))**2 - (np.sin(np.pi * y * x))**2)) * (x**2 + y**2)
    def myu1_test(self, y):
        return 1
    def myu2_test(self, y):
        return np.exp((np.sin(2 * np.pi * y))**2)
    def myu3_test(self, x):
        return 1
    def myu4_test(self, x):
        return np.exp((np.sin(np.pi * x))**2)
    def myu5_test(self, x):
        return np.exp((np.sin(0.5 * np.pi * x))**2)
    def myu6_test(self, y):
        return np.exp((np.sin(np.pi * y))**2)    
    def myu1(self, y):
        return (np.sin(np.pi * y))**2
    def myu2(self, y):
        return (np.sin(2 * np.pi * y))**2
    def myu3(self, x):
        return (np.sin(np.pi * x))**2
    def myu4(self, x):
        return (np.sin(2 * np.pi * x))**2
    def f_main(self, x, y):
        return abs(x**2 - 2 * y)


    def MyFunction_test(self):
        self.textBrowser_2.setText("")
        n = int(self.lineEdit.text())
        m = int(self.lineEdit_2.text())
        lim_step = int(self.lineEdit_4.text())
        eps = float(self.lineEdit_3.text())
        h = 2 / n
        k = 1 / m
        start_sol = []
        f = []
        for i in range(n + 1):
            start_sol.append([0] * (m + 1))
        for i in range(n - 1):
            f.append([0] * (m - 1))
        for i in range(n - 1):
            for j in range(m - 1):
                f[i][j] = -self.f_test((i + 1) * h, (j + 1) * k)
        for i in range(m + 1):
            start_sol[0][i] = self.myu1_test(i * k)
        for i in range(m + 1):
            start_sol[n][i] = self.myu2_test(i * k)
        for i in range(n + 1):
            start_sol[i][0] = self.myu3_test(i * h)
        for i in range(n + 1):
            start_sol[i][m] = self.myu4_test(i * h)
        solution = []
        for i in range(n + 1):
            solution.append([0] * (m + 1))
        if(self.comboBox.currentText() == "Метод Зейделя"):
            solution = self.zeidel_method(lim_step, eps, n, m, start_sol, f)
        if(self.comboBox.currentText() == "ММН"):
            solution = self.min_discrepancy(lim_step, eps, n, m, start_sol, f)

        acc_sol = []
        for i in range(n + 1):
            acc_sol.append([0] * (m + 1))
        for i in range(n + 1):
            for j in range(m + 1):
                acc_sol[i][j] = self.acc_solution(i * h, j * k)
        self.tableWidget.setRowCount(m + 1)
        self.tableWidget.setColumnCount(n + 1)
        for i in range(n + 1):
            for j in range(m + 1):
                self.tableWidget.setItem(i, j, QtWidgets.QTableWidgetItem(str(solution[i][j])))
        max_diff = 0
        for i in range(n + 1):
            for j in range(m + 1):   
                curr_diff = abs(solution[i][j] - acc_sol[i][j])
                if curr_diff > max_diff:
                    max_diff = curr_diff
        self.textBrowser.setText("Погрешность решения тестовой задачи " + str(max_diff))    
    def MyFunction_main(self):
        # full step
        self.textBrowser_2.setText("")
        n = int(self.lineEdit.text())
        m = int(self.lineEdit_2.text())
        lim_step = int(self.lineEdit_4.text())
        eps = float(self.lineEdit_3.text())
        h = 2 / n
        k = 1 / m
        start_sol = []
        f = []
        for i in range(n + 1):
            start_sol.append([0] * (m + 1))
        for i in range(n - 1):
            f.append([0] * (m - 1))
        for i in range(n - 1):
            for j in range(m - 1):
                f[i][j] = -self.f_main((i + 1) * h, (j + 1) * k)
        for i in range(m + 1):
            start_sol[0][i] = self.myu1(i * k)
        for i in range(m + 1):
            start_sol[n][i] = self.myu2(i * k)
        for i in range(n + 1):
            start_sol[i][0] = self.myu3(i * h)
        for i in range(n + 1):
            start_sol[i][m] = self.myu4(i * h)
        solution = []
        for i in range(n + 1):
            solution.append([0] * (m + 1))
        if(self.comboBox.currentText() == "Метод Зейделя"):
            solution = self.zeidel_method(lim_step, eps, n, m, start_sol, f)
        if(self.comboBox.currentText() == "ММН"):
            solution = self.min_discrepancy(lim_step, eps, n, m, start_sol, f)       
        self.tableWidget.setRowCount(m + 1)
        self.tableWidget.setColumnCount(n + 1)
        for i in range(n + 1):
            for j in range(m + 1):
                self.tableWidget.setItem(i, j, QtWidgets.QTableWidgetItem(str(solution[i][j])))
        # half step
        n_half = n * 2
        m_half = m * 2
        lim_step_half = lim_step * 2
        h_half = 2 / n_half
        k_half = 1 / m_half
        start_sol_half = []
        f_half = []
        for i in range(n_half + 1):
            start_sol_half.append([0] * (m_half + 1))
        for i in range(n_half - 1):
            f_half.append([0] * (m_half - 1))
        for i in range(n_half - 1):
            for j in range(m_half - 1):
                f_half[i][j] = -self.f_main((i + 1) * h_half, (j + 1) * k_half)
        for i in range(m_half + 1):
            start_sol_half[0][i] = self.myu1(i * k_half)
        for i in range(m_half + 1):
            start_sol_half[n_half][i] = self.myu2(i * k_half)
        for i in range(n_half + 1):
            start_sol_half[i][0] = self.myu3(i * h_half)
        for i in range(n_half + 1):
            start_sol_half[i][m_half] = self.myu4(i * h_half)
        solution_half = []
        for i in range(n_half + 1):
            solution_half.append([0] * (m_half + 1))
        if(self.comboBox.currentText() == "Метод Зейделя"):
            solution_half = self.zeidel_method(lim_step_half, eps, n_half, m_half, start_sol_half, f_half)
        if(self.comboBox.currentText() == "ММН"):
            solution_half = self.min_discrepancy(lim_step_half, eps, n_half, m_half, start_sol_half, f_half)
        max_diff = 0
        max_diff_x, max_diff_y = 0, 0
        for i in range(n + 1):
            for j in range(m + 1):   
                curr_diff = abs(solution[i][j] - solution_half[2 * i][2 * j])
                if curr_diff > max_diff:
                    max_diff = curr_diff
                    max_diff_x = i
                    max_diff_y = j
        self.textBrowser.setText("Погрешность решения тестовой задачи " + str(max_diff) + 
        "\nМаксимальное отклонение численного от точного в точке: x = " + str(max_diff_x * 2 * h) + 
            ", y = " + str(max_diff_y * 2 * k))    
    def MyFunction_test_cut(self):
        self.tableWidget.clear()
        self.textBrowser_2.setText("")
        n = int(self.lineEdit.text())
        m = int(self.lineEdit_2.text())
        lim_step = int(self.lineEdit_4.text())
        eps = float(self.lineEdit_3.text())
        h = 2 / n
        k = 1 / m
        start_sol = []
        acc_sol = []
        solution = []
        f = []
        for i in range(int(n / 2) + 1):
            start_sol.append([0] * (m + 1))
        for i in range(int(n / 2)):
            start_sol.append([0] * (int(m / 2) + 1))

        for i in range(int(n / 2) + 1):
            solution.append([0] * (m + 1))
        for i in range(int(n / 2)):
            solution.append([0] * (int(m / 2) + 1))

        for i in range(int(n / 2) + 1):
            acc_sol.append([0] * (m + 1))
        for i in range(int(n / 2)):
            acc_sol.append([0] * (int(m / 2) + 1))

        for i in range(int(n / 2)):
            f.append([0] * (m - 1))
        for i in range(0, int(n / 2) - 1):
            f.append([0] * (int(m / 2) - 1))

        for i in range(n - 1):
            for j in range(int(m / 2) - 1):
                f[i][j] = -self.f_test((i + 1) * h, (j + 1) * k)

        for i in range(int(n / 2)):
            for j in range(int(m / 2) - 1, m - 1):
                f[i][j] = -self.f_test((i + 1) * h, (j + 1) * k)
        
        for i in range(m + 1):
            start_sol[0][i] = self.myu1_test(i * k)
        for i in range(int(m / 2) + 1):
            start_sol[n][i] = self.myu2_test(i * k)
        for i in range(n + 1):
            start_sol[i][0] = self.myu3_test(i * h)
        for i in range(int(n / 2) + 1):
            start_sol[i][m] = self.myu4_test(i * h)
        for i in range(int(n / 2), n + 1):
            start_sol[i][int(m / 2)] = self.myu5_test(i * h)
        for i in range(int(m / 2), m + 1):
            start_sol[int(n / 2)][i] = self.myu6_test(i * k)

        solution = self.min_discrepancy_cut(lim_step, eps, n, m, start_sol, f)

        for i in range(n + 1):
            for j in range(int(m / 2) + 1):
                acc_sol[i][j] = self.acc_solution(i * h, j * k)

        for i in range(int(n / 2) + 1):
            for j in range(int(m / 2) + 1, m + 1):
                acc_sol[i][j] = self.acc_solution(i * h, j * k)

        self.tableWidget.setRowCount(m + 1)
        self.tableWidget.setColumnCount(n + 1)
        for i in range(n + 1):
            for j in range(int(m / 2) + 1):
                self.tableWidget.setItem(i, j, QtWidgets.QTableWidgetItem(str(solution[i][j])))
        for i in range(int(n / 2) + 1):
            for j in range(int(m / 2) + 1, m + 1):
                self.tableWidget.setItem(i, j, QtWidgets.QTableWidgetItem(str(solution[i][j])))        
        max_diff = 0
        for i in range(n + 1):
            for j in range(int(m / 2) + 1):
                curr_diff = abs(solution[i][j] - acc_sol[i][j])
                if curr_diff > max_diff:
                    max_diff = curr_diff
        for i in range(int(n / 2) + 1):
            for j in range(int(m / 2) + 1, m + 1):
                curr_diff = abs(solution[i][j] - acc_sol[i][j])
                if curr_diff > max_diff:
                    max_diff = curr_diff      
        self.textBrowser.setText("Погрешность решения тестовой задачи " + str(max_diff))    





if __name__=="__main__":
    app = QtWidgets.QApplication(sys.argv)
    myapp = MyWin()
    myapp.show()
    try:
        sys.exit(app.exec_())
    except SystemExit:
        pass