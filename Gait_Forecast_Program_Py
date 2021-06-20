#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import tkinter as tk
from tkinter import filedialog as fd
from tkinter import messagebox
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import (
    FigureCanvasTkAgg,
    NavigationToolbar2Tk,
)
from matplotlib.figure import Figure


# In[2]:


class GaitDataMLP:
    def __init__(
        self,
        layer_dims,
        learning_rate,
        gait_raw_data="Sheet2.dat",
        num_iter_cap=100,
        train_test_split=0.8,
        polarity="bi",
        cost_limit=10 ^ (-6),
    ):
        self.gait_raw_data = np.loadtxt(gait_raw_data)
        self.learning_rate = learning_rate
        self.layer_dims = layer_dims
        self.num_iter_cap = num_iter_cap
        self.cost_limit = cost_limit
        self.train_test_split = train_test_split
        if polarity == "uni" or polarity == "bi":
            self.polarity = polarity
        else:
            raise Exception(
                "Polarity can only be unipolar -> 'uni' or bipolar ->'bi'"
            )

    def data_preprocessing(self, a):

        # Normalization
        self.aMin = np.min(a)
        self.aMax = np.max(a)

        if self.polarity == "uni":
            aNorm = (a - self.aMin) / (self.aMax - self.aMin)
        elif self.polarity == "bi":
            aNorm = (2 * (a - self.aMin) / (self.aMax - self.aMin)) - 1

        self.aNorm = aNorm

        # Padding to prevent data loss from windowing, the input data is padded with the repetition of the edge data
        aNorm = np.pad(self.aNorm, (self.layer_dims[0], 0), "edge")
        self.test = aNorm
        # Rolling Window
        window = self.layer_dims[0] + 1
        shape = aNorm.shape[:-1] + (aNorm.shape[-1] - window + 1, window)
        strides = aNorm.strides + (aNorm.strides[-1],)

        windowed_data = np.lib.stride_tricks.as_strided(
            aNorm, shape=shape, strides=strides
        )

        # Train Test Split

        # for training splitting in time series can't be randomized
        # it needs to follow the change of data through time

        split_index = int(windowed_data.shape[0] * self.train_test_split)

        self.x_train = windowed_data[:split_index, :-1].T
        self.y_train = windowed_data[:split_index, -1].reshape(1, -1)

        self.x_test = windowed_data[split_index:, :-1].T
        self.y_test = windowed_data[split_index:, -1].reshape(1, -1)

        # Recalling Data
        self.in_gait_data = windowed_data[:, :-1].T
        self.out_gait_data = windowed_data[:, -1].reshape(1, -1)

    def init_param(self):
        """
        Initializing parameter of "W"

        Input:
        layer_dims = list containing the dimensions of each layer in the network

        Output:
        parameters = dictionary containing the parameters "W1", "b1", ..., "WL", "bL":
                        Wl  weight matrix of shape (layer_dims[l], layer_dims[l-1])
        """
        self.parameters = {}
        L = len(self.layer_dims)  # number of layers in the network

        for l in range(1, L):
            self.parameters["W" + str(l)] = (
                np.random.randn(self.layer_dims[l], self.layer_dims[l - 1])
                * 0.001
            )
            self.parameters["b" + str(l)] = np.zeros((self.layer_dims[l], 1))

    def sigmoid(self, Z):

        if self.polarity == "uni":
            A = 1 / (1 + np.exp(-Z))

        elif self.polarity == "bi":
            A = (2 / (1 + np.exp(-Z))) - 1

        return A, Z

    def linear_forward(self, A, W, b):
        """
        Implement the linear part of a layer's forward propagation.

        Input:
        A = activations from previous layer (or input data)
        W = weights matrix
        b = bias vector

        Output:
        Z = the input of the activation function, also called pre-activation parameter
        cache = a python tuple containing "A", "W" and "b" stored for computing the backprop
        """

        Zin = np.dot(np.ascontiguousarray(W), np.ascontiguousarray(A)) - b
        cache = (A, W, b)

        return Zin, cache

    def linear_activation_forward(self, A_prev, W, b):
        """
        Implementing activation function
        """
        Z, linear_cache = self.linear_forward(
            np.ascontiguousarray(A_prev),
            np.ascontiguousarray(W),
            np.ascontiguousarray(b),
        )
        A, activation_cache = self.sigmoid(Z)
        cache = (linear_cache, activation_cache)

        return A, cache

    def L_model_forward(self, X):
        """
        Forward Propagation in one function

        Input:
        X = Input matrix
        parameters = dictionary containing all the 'W' and 'b' necessary for computing
        """

        self.caches = []
        A = X
        L = len(self.parameters) // 2

        for l in range(1, L + 1):
            A_prev = A
            A, cache = self.linear_activation_forward(
                A_prev,
                self.parameters["W" + str(l)],
                self.parameters["b" + str(l)],
            )
            self.caches.append(cache)
        return A

    def compute_cost(self, A, Y):
        cost = np.sum(np.square(Y - A), axis=1) / len(A.T)
        return cost

    def sigmoid_inv(self, v):
        """
        Derivation of sigmoid function to help backward propagation

        Input:
        v = Activation vector cache

        Output:
        g = Gradient
        """
        A, Z = self.sigmoid(v)
        if self.polarity == "uni":
            g = A * (1 - A)
        elif self.polarity == "bi":
            g = (1 - A ** 2) / 2

        return g

    def L_model_backward(self, A, Y):
        """
        In this process, the cache of forward propagation will be used to calculate delta for the parameter update

        Input:
        A= Model Output
        Y= Desired Output
        caches = List of cache (Linear Activation)

        Output:
        d_cache = dictionary of delta for each layer
        """
        L = len(self.caches)
        self.d_cache = {}
        """
        Outer Layer Delta Computation
        """
        d_outer = (Y - A) * self.sigmoid_inv(self.caches[L - 1][1])
        d = d_outer
        self.d_cache["d" + str(L)] = d
        """
        n Layer Delta Computation
        """
        for l in reversed(range(L - 1)):
            d_prev = d
            d = np.dot(self.caches[l + 1][0][1].T, d_prev) * self.sigmoid_inv(
                self.caches[l][1]
            )
            self.d_cache["d" + str(l + 1)] = d

    def update_param(self):
        # Due to the limitation of python for loop performance, numpy array is used instead
        # the updating equation is the same, the difference is there is no iteration for each delta and x
        # and it is replaced by 3d matrix scalar multiplication

        L = len(self.parameters) // 2
        for l in range(1, L + 1):

            d_array = self.d_cache["d" + str(l)][:, None]
            x_array = self.caches[l - 1][0][0][:, None].transpose(1, 0, 2)
            d_x_mul = np.multiply(d_array, x_array)
            d_x_array = np.sum(d_x_mul, axis=2)

            self.parameters["W" + str(l)] += self.learning_rate * d_x_array

            self.parameters["b" + str(l)] -= self.learning_rate * np.sum(
                d_array, axis=2
            )

    def data_postprocessing(self, A):
        # Inversing the process of normalization
        if self.polarity == "uni":
            aInv = A * (self.aMax - self.aMin) + self.aMin
        elif self.polarity == "bi":
            aInv = ((A + 1) * (self.aMax - self.aMin) / 2) + self.aMin

        return aInv

    def train(self):

        self.data_preprocessing(self.gait_raw_data)

        self.init_param()

        self.cost_epoch = []
        self.cost_test_epoch = []
        self.iter_optimize = 0

        while True:
            self.iter_optimize += 1
            A_train = self.L_model_forward(self.x_train)

            self.cost_train = self.compute_cost(A_train, self.y_train)
            self.cost_epoch.append(self.cost_train)

            if (
                self.cost_train < self.cost_limit
                or self.iter_optimize > self.num_iter_cap - 1
            ):
                break

            self.L_model_backward(A_train, self.y_train)

            self.update_param()

            A_test = self.L_model_forward(self.x_test)

            self.cost_test = self.compute_cost(A_test, self.y_test)
            self.cost_test_epoch.append(self.cost_test)

    #             if self.iter_optimize % 100 == 0:
    #                 print(self.iter_optimize)

    def recall(self):
        # in_gait_data is the raw data that had been preprocessed
        A_recall = self.L_model_forward(self.in_gait_data)
        A_post = self.data_postprocessing(A_recall)
        self.cost_recall = self.compute_cost(A_recall, self.out_gait_data)
        return A_post


# In[3]:


class GUI:
    def __init__(self):
        self.inputfile = " "
        self.window = tk.Tk()
        self.window.state("normal")
        self.window.title(
            "Gait Data Prediction System using MLP and EBPA - Aditya Wardianto 07311940000001"
        )

        # Frame
        self.frm_input = tk.Frame()
        frm_input = self.frm_input

        self.cost_texttrain = tk.StringVar()
        self.cost_texttrain.set("Train Error = ")

        self.cost_texttest = tk.StringVar()
        self.cost_texttest.set("Test Error = ")

        self.iter_text = tk.StringVar()
        self.iter_text.set("Iteration = ")

        self.cost_textrecall = tk.StringVar()
        self.cost_textrecall.set("Recall Error = ")

        # Label
        lbl_nin = tk.Label(text="Number of Input Node", master=frm_input)
        lbl_nhidden = tk.Label(
            text="Number of Node in                 \nHidden Layer (L1 L2 L3 ... LN)",
            master=frm_input,
        )
        lbl_nout = tk.Label(text="Number of Output Node", master=frm_input)
        lbl_iter_cap = tk.Label(
            text="Number of Iteration Limit", master=frm_input
        )
        lbl_lrate = tk.Label(text="Learning Rate", master=frm_input)
        lbl_split = tk.Label(
            text="Train Test Split (In decimal)", master=frm_input
        )
        self.lbl_costtrain = tk.Label(
            master=frm_input, textvariable=self.cost_texttrain
        )
        self.lbl_costtest = tk.Label(
            master=frm_input, textvariable=self.cost_texttest
        )
        self.lbl_costrecall = tk.Label(
            master=frm_input, textvariable=self.cost_textrecall
        )
        self.lbl_iter_opt = tk.Label(
            master=frm_input, textvariable=self.iter_text
        )

        one = tk.StringVar()
        one.set("1")
        pred_text = tk.StringVar()
        pred_text.set("Prediction")

        a = tk.StringVar()
        a.set("10")
        b = tk.StringVar()
        b.set("5")
        c = tk.StringVar()
        c.set("50000")
        d = tk.StringVar()
        d.set("0.0002")
        e = tk.StringVar()
        e.set("0.8")

        # Entry
        self.ent_nin = tk.Entry(master=frm_input, textvariable=a)
        self.ent_nhidden = tk.Entry(master=frm_input, textvariable=b)
        ent_nout = tk.Entry(
            master=frm_input, textvariable=one, state="disable"
        )
        self.ent_iter_cap = tk.Entry(master=frm_input, textvariable=c)
        self.ent_lrate = tk.Entry(master=frm_input, textvariable=d)
        self.ent_split = tk.Entry(master=frm_input, textvariable=e)

        # Button
        btn_file = tk.Button(
            text="Input File", master=frm_input, command=self.input_file
        )
        btn_Learn = tk.Button(
            text="Train (Takes Time)", master=frm_input, command=self.train
        )
        btn_Pred = tk.Button(
            text="Recall", master=frm_input, command=self.recall
        )

        # Radio
        self.polar = tk.StringVar()
        self.polar.set("uni")
        R1 = tk.Radiobutton(
            master=frm_input,
            text="Unipolar",
            variable=self.polar,
            value="uni",
        )

        R2 = tk.Radiobutton(
            master=frm_input, text="Bipolar", variable=self.polar, value="bi",
        )

        # Packing
        btn_file.pack(anchor=tk.W)
        lbl_nin.pack(anchor=tk.W)
        self.ent_nin.pack(anchor=tk.W)
        lbl_nhidden.pack(anchor=tk.W)
        self.ent_nhidden.pack(anchor=tk.W)
        lbl_nout.pack(anchor=tk.W)
        ent_nout.pack(anchor=tk.W)

        lbl_iter_cap.pack(anchor=tk.W)
        self.ent_iter_cap.pack(anchor=tk.W)
        lbl_lrate.pack(anchor=tk.W)
        self.ent_lrate.pack(anchor=tk.W)
        lbl_split.pack(anchor=tk.W)
        self.ent_split.pack(anchor=tk.W)

        R1.pack(anchor=tk.W)
        R2.pack(anchor=tk.W)

        btn_Learn.pack(anchor=tk.W, pady=5)
        self.lbl_costtrain.pack(anchor=tk.W, pady=5)
        self.lbl_costtest.pack(anchor=tk.W, pady=5)
        self.lbl_iter_opt.pack(anchor=tk.W, pady=5)

        btn_Pred.pack(anchor=tk.W, pady=5)
        self.lbl_costrecall.pack(anchor=tk.W, pady=5)

        frm_input.grid(row=0, column=1, padx=5, pady=5, sticky="N")

        self.window.mainloop()

    def input_file(self):

        filetypes = [("All Files", "*")]

        self.inputfile = fd.askopenfilename(
            title="Open a file", initialdir="/", filetypes=filetypes
        )

    def input_data(self):
        """
        Getting input for training from GUI

        Output:

        """
        try:
            if (
                int(self.ent_nin.get()) == 0
                or int(self.ent_iter_cap.get()) == 0
                or float(self.ent_lrate.get()) == 0
                or float(self.ent_split.get()) == 0
            ):
                messagebox.showerror("Input is not valid", "Input > 0")
                return 0
            else:

                self.layer_dims = [
                    int(s)
                    for s in self.ent_nhidden.get().split()
                    if s.isdigit()
                ]

                self.layer_dims.insert(0, int(self.ent_nin.get()))

                self.layer_dims.append(1)

                self.learning_rate = float(self.ent_lrate.get())
                self.num_iter_cap = int(self.ent_iter_cap.get())
                self.train_test_split = float(self.ent_split.get())
                self.polarity = self.polar.get()

        except BaseException:
            messagebox.showerror("Input is not valid", "Input Real")

    def train(self):
        if self.inputfile != " ":
            gait_raw_data = self.inputfile
            self.input_data()

            self.model = GaitDataMLP(
                layer_dims=self.layer_dims,
                gait_raw_data=gait_raw_data,
                num_iter_cap=self.num_iter_cap,
                train_test_split=self.train_test_split,
                learning_rate=self.learning_rate,
                polarity=self.polarity,
            )

            self.model.train()

            self.cost_texttrain.set(
                "Train Error = " + str(self.model.cost_train[0])
            )
            self.cost_texttest.set(
                "Test Error = " + str(self.model.cost_test[0])
            )
            self.iter_text.set("Iteration = " + str(self.model.iter_optimize))

            self.cost_textrecall.set("Recall Error = ")

            fig = Figure(figsize=(19.5, 11), dpi=60, constrained_layout=True)
            gs = fig.add_gridspec(ncols=3, nrows=2, hspace=0.5, wspace=0.1)

            y_plot = np.array(
                [
                    [
                        self.model.gait_raw_data,
                        self.model.aNorm,
                        self.model.x_train[0],
                    ],
                    [self.model.cost_epoch, self.model.cost_test_epoch],
                ],
                dtype=object,
            )

            title_list = np.array(
                [
                    ["Raw Data", "Normalized Data", "Train Set"],
                    ["Train and Test Cost Epoch"],
                ],
                dtype=object,
            )

            for row in range(2):
                for col in range(3):
                    if row == 0:
                        ax = fig.add_subplot(gs[row, col])
                        ax.set_title(
                            title_list[row][col], fontweight="bold", size=15
                        )
                        ax.set_xlabel("Time", size=12)
                        ax.set_ylabel("Value", size=12)
                        ax.plot(range(len(y_plot[row][col])), y_plot[row][col])
                    if row == 1 and col == 0:
                        ax = fig.add_subplot(gs[row, col])
                        ax.set_title(
                            title_list[row][col], fontweight="bold", size=15
                        )
                        ax.set_xlabel("Iteration", size=12)
                        ax.set_ylabel("Value", size=12)
                        ax.plot(
                            range(len(y_plot[row][col])),
                            y_plot[row][col],
                            label="Train Error",
                        )
                        ax.plot(
                            range(len(y_plot[row][col + 1])),
                            y_plot[row][col + 1],
                            label="Test Error",
                        )
                        ax.legend()

            canvas = FigureCanvasTkAgg(fig, master=self.window)
            canvas.draw()
            canvas.get_tk_widget().grid(row=0, column=0, padx=5, pady=5)
            toolbarFrame = tk.Frame(master=self.window)
            toolbarFrame.grid(row=1, column=0, padx=5, pady=5)
            toolbar = NavigationToolbar2Tk(canvas, toolbarFrame)

        else:
            messagebox.showerror("No File", "Input File")

    def recall(self):
        try:
            y_recall = self.model.recall()
            self.cost_textrecall.set(
                "Recall Error = " + str(self.model.cost_recall[0])
            )

            fig = Figure(figsize=(19.5, 11), dpi=60, constrained_layout=True)
            gs = fig.add_gridspec(ncols=3, nrows=2, hspace=0.5, wspace=0.1)

            y_plot = np.array(
                [
                    [
                        self.model.gait_raw_data,
                        self.model.aNorm,
                        self.model.x_train[0],
                    ],
                    [
                        self.model.cost_epoch,
                        self.model.cost_test_epoch,
                        self.model.gait_raw_data,
                        y_recall[0],
                    ],
                ],
                dtype=object,
            )

            title_list = np.array(
                [
                    ["Raw Data", "Normalized Data", "Train Set"],
                    ["Train and Test Cost Epoch", "Recall Input and Output"],
                ],
                dtype=object,
            )

            for row in range(2):
                for col in range(3):
                    if row == 0:
                        ax = fig.add_subplot(gs[row, col])
                        ax.set_title(
                            title_list[row][col], fontweight="bold", size=15
                        )
                        ax.set_xlabel("Time", size=12)
                        ax.set_ylabel("Value", size=12)
                        ax.plot(range(len(y_plot[row][col])), y_plot[row][col])
                    if row == 1 and col == 0:
                        ax = fig.add_subplot(gs[row, col])
                        ax.set_title(
                            title_list[row][col], fontweight="bold", size=15
                        )
                        ax.set_xlabel("Iteration", size=12)
                        ax.set_ylabel("Value", size=12)
                        ax.plot(
                            range(len(y_plot[row][col])),
                            y_plot[row][col],
                            label="Train Error",
                        )
                        ax.plot(
                            range(len(y_plot[row][col + 1])),
                            y_plot[row][col + 1],
                            label="Test Error",
                        )
                        ax.legend()
                    if row == 1 and col == 2:
                        ax = fig.add_subplot(gs[row, col - 1])
                        ax.set_title(
                            title_list[row][col - 1],
                            fontweight="bold",
                            size=15,
                        )
                        ax.set_xlabel("Time", size=12)
                        ax.set_ylabel("Value", size=12)
                        ax.plot(
                            range(len(y_plot[row][col])),
                            y_plot[row][col],
                            label="Input Data",
                        )
                        ax.plot(
                            range(len(y_plot[row][col + 1])),
                            y_plot[row][col + 1],
                            label="Predicted Output Data",
                        )
                        ax.legend()

            canvas = FigureCanvasTkAgg(fig, master=self.window)
            canvas.draw()
            canvas.get_tk_widget().grid(row=0, column=0, padx=5, pady=5)
            toolbarFrame = tk.Frame(master=self.window)
            toolbarFrame.grid(row=1, column=0, padx=5, pady=5)
            toolbar = NavigationToolbar2Tk(canvas, toolbarFrame)

        except BaseException:
            messagebox.showerror(
                "No Trained Model", "Train the Model before Recall"
            )


# In[4]:


if __name__ == "__main__":
    app = GUI()


# In[ ]:




