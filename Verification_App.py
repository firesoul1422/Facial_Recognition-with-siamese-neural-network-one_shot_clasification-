# Import dependencies
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

input_path = os.path.join("application_data", "input_image")

# def show_frame():
#     _, frame = cap.read()
#
#     frame = frame[100:350, 200:450, :]
#     frame = cv2.flip(frame, 1)
#     cv2image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA)
#     img = Image.fromarray(cv2image)
#     imgtk = ImageTk.PhotoImage(image=img)
#     lmain.imgtk = imgtk
#     lmain.configure(image=imgtk)
#     lmain.after(10, show_frame)
#
#
#
# def preprocess(file_path):
#     # Read in image from file path
#     byte_img = tf.io.read_file(file_path)
#     # Load in the image
#     img = tf.io.decode_jpeg(byte_img)
#
#     # Preprocessing steps - resizing the image to be 100x100x3
#     img = tf.image.resize(img, (105, 105))
#     # Scale image to be between 0 and 1
#     img = img / 255.0
#
#     # Return image
#     return img
#
#
# def verify():
#     detection_threshold = 0.99
#     verification_threshold = 0.8
#     print("heeeereeee")
#
#
#     # Capture input image from our webcam
#     SAVE_PATH = os.path.join('application_data', 'input_image', 'input_image.jpg')
#     ret, frame = cap.read()
#     frame = frame[120:120 + 250, 200:200 + 250, :]
#     print(frame)
#     cv2.imwrite(SAVE_PATH, frame)
#
#     # Build results array
#     results = []
#     for image in os.listdir(os.path.join('application_data', 'verification_images')):
#         print(image)
#         input_img = preprocess(os.path.join('application_data', 'input_image', 'input_image.jpg'))
#         validation_img = preprocess(os.path.join('application_data', 'verification_images', image))
#
#         # Make Predictions
#         result = model.predict(list(np.expand_dims([input_img, validation_img], axis=1)))
#         results.append(result)
#
#     # Detection Threshold: Metric above which a prediciton is considered positive
#     detection = np.sum(np.array(results) > detection_threshold)
#
#     # Verification Threshold: Proportion of positive predictions / total positive samples
#     verification = detection / len(os.listdir(os.path.join('application_data', 'verification_images')))
#     verified = verification > verification_threshold
#
#     # Set verification text
#     if verified:
#         verification_label1 = Label(text="Verified",  font=("Courier", 30)).grid(row=260, column=0, padx=50)
#         msg.showinfo("Verified", "you have been Verified please wait 5 second to start the app")
#         time.sleep(5)
#         window.withdraw()
#
#
#     else:
#         verification_label2 = Label(text="UnVerified", font=("Courier", 30)).grid(row=260, column=0, padx=50)
#         msg.showwarning("Unverified", "you have been Unverified please wait 5 second to try again")
#         time.sleep(5)
#         window.withdraw()
#         window
#     pass






# #Set up GUI
# window = Tk()  #Makes main window
# window.wm_title("Digital Microscope")
# window.config(background="#FFFFFF")
# # Load tensorflow/keras model
# model = tf.keras.models.load_model('siamese_model.h5', custom_objects={'L1Diist': L1Diist})
#
# #Graphics window
# imageFrame = Frame(window, width=600, height=500)
# imageFrame.grid(row=0, column=0, padx=10, pady=2)
#
#
#
# #Capture video frames
# lmain = Label(imageFrame)
# lmain.grid(row=0, column=0)
# cap = cv2.VideoCapture(0)
#
#
# #Slider window (slider controls stage position)
# sliderFrame = Frame(window, width=600, height=100)
# sliderFrame.grid(row = 200, column=0, padx=10, pady=2)
# button = Button(text = "verify", command=verify).grid(row=250, column=0, padx=50)
# verification_label = Label(text="Verification").grid(row=260, column=0, padx=50)
#
#



# show_frame()  #Display 2
# window.mainloop()  #Starts GUI

class FaceRecognitionApp():
    def __init__(self):
        # counter to number of tries to enter
        self.counter = 0
        # Set up GUI
        # super().__init__()
        self.window = Tk()  # Makes main window
        self.window.wm_title("Digital Microscope")
        self.window.config(background="#FFFFFF")

        # Load tensorflow/keras model
        self.model = tf.keras.models.load_model('siamese_model.h5', custom_objects={'L1Diist': L1Diist})


        # Graphics window
        self.imageFrame = Frame(self.window, width=600, height=500)
        self.imageFrame.grid(row=0, column=0, padx=10, pady=2)

        # Capture video frames
        self.lmain = Label(self.imageFrame)
        self.lmain.grid(row=0, column=0)
        self.cap = cv2.VideoCapture(0)

        # Slider window (slider controls stage position)
        self.sliderFrame = Frame(self.window, width=600, height=100)
        self.sliderFrame.grid(row=200, column=0, padx=10, pady=2)
        self.button = Button(text="verify", command=self.verify).grid(row=250, column=0, padx=50)
        self.verification_label = Label(text="Verification").grid(row=260, column=0, padx=50)
        self.window.resizable(False, False)

        self.show_frame()
        self.window.mainloop()

    def show_frame(self):
        self._, self.frame = self.cap.read()

        self.frame = self.frame[100:350, 200:450, :]
        self.frame = cv2.flip(self.frame, 1)
        self.cv2image = cv2.cvtColor(self.frame, cv2.COLOR_BGR2RGBA)
        self.img = Image.fromarray(self.cv2image)
        self.imgtk = ImageTk.PhotoImage(image=self.img)
        self.lmain.imgtk = self.imgtk
        self.lmain.configure(image=self.imgtk)
        self.lmain.after(10, self.show_frame)

    def preprocess(self, file_path):
        # Read in image from file path
        self.byte_img = tf.io.read_file(file_path)
        # Load in the image
        self.img = tf.io.decode_jpeg(self.byte_img)

        # Preprocessing steps - resizing the image to be 100x100x3
        self.img = tf.image.resize(self.img, (105, 105))
        # Scale image to be between 0 and 1
        self.img = self.img / 255.0

        # Return image
        return self.img

    def verify(self):
        self.detection_threshold = 0.9
        self.verification_threshold = 0.8

        # Capture input image from our webcam
        self.SAVE_PATH = os.path.join('application_data', 'input_image', 'input_image.jpg')
        self.ret, self.frame = self.cap.read()
        self.frame = self.frame[120:120 + 250, 200:200 + 250, :]
        print(self.frame)
        cv2.imwrite(self.SAVE_PATH, self.frame)

        # Build results array
        self.results = []
        for idx, self.image in enumerate(os.listdir(os.path.join('application_data', 'verification_images'))):
            print(self.image)
            self.input_img = self.preprocess(os.path.join('application_data', 'input_image', 'input_image.jpg'))
            self.validation_img = self.preprocess(os.path.join('application_data', 'verification_images', self.image))

            # Make Predictions
            self.result = self.model.predict(list(np.expand_dims([self.input_img, self.validation_img], axis=1)), verbose = True);
            print(f"{idx+1}:")
            self.results.append(self.result)

        # Detection Threshold: Metric above which a prediciton is considered positive
        self.detection = np.sum(np.array(self.results) > self.detection_threshold)

        # Verification Threshold: Proportion of positive predictions / total positive samples
        self.verification = self.detection / len(os.listdir(os.path.join('application_data', 'verification_images')))
        self.verified = self.verification > self.verification_threshold

        # Set verification text
        if self.verified:
            self.counter = 1 + self.counter
            self.verification_label1 = Label(text="Verified", font=("Courier", 30)).grid(row=260, column=0, padx=50)
            msg.showinfo("Verified", f"you have been Verified please wait 5 second to start the app. you have tried {self.counter} times to  (verification: {self.verification*100}%)")
            time.sleep(5)
            self.window.withdraw()
            self.window.quit()


        else:
            self.verification_label2 = Label(text="Unverified", font=("Courier", 10)).grid(row=260, column=0, padx=50)
            msg.showwarning("Unverified", f"you have been Unverified please try again (verification: {self.verification*100}%)")
            self.counter = 1 + self.counter

            if self.counter == 5:
                msg.showwarning("Unverified", f"you have tried {self.counter} times to enter, the app will shutdown")
                self.window.withdraw()
                self.window.quit()

            # time.sleep(5)


        pass





if __name__ == '__main__':
    gpus = tf.config.experimental.list_physical_devices("GPU")
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)

    FaceRecognitionApp()




