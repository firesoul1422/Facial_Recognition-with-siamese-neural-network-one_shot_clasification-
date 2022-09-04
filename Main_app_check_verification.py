from Verification_App import FaceRecognitionApp
import cv2
import tensorflow as tf
from layers import L1Diist
import os
import numpy as np
import sys
from tkinter import *
from PIL import ImageTk, Image
import tkinter.messagebox as msg
import time



class MyApp(): #Tk):

    def __init__(self):
        super().__init__()
        self.windo = Tk()
        self.windo.title(string="Main App")
        self.windo.geometry("400x400")
        self.verification_label = Label(text=fr.verification).grid(row=260, column=0, padx=50)

        self.windo.resizable(False, False)

    def run(self):
        self.windo.mainloop()


if __name__ == '__main__':
    gpus = tf.config.experimental.list_physical_devices("GPU")
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)

    fr = FaceRecognitionApp()
    if fr.verified:
        s = MyApp()
        s.run()
    else:
        exit()