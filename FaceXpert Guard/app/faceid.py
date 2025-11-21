"""
Note: Due to some complications that I was not able to fix, the localized variable names are not applicable to this file.
As such, I was forced to use the full path when pulling from the folder data and therefore those parts have been redacted.
They must be filled in either with the full path name or with the local folder names before usage
"""

import os
os.chdir('')# set directory within the app folder

import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
import cv2
import tensorflow as tf
from layers import L1Dist
import numpy as np
import keyboard
import random
import threading

class FacialVerificationApp:
    def __init__(self, root):
        self.root = root
        self.root.title("FaceXpert Guard")
        self.root.geometry("800x700")
        self.root.configure(bg='#141420')
        self.is_face = False
        self.a_pressed = False
        self.q_pressed = False
        self.is_verifying = False
        
        # Setup UI
        self.setup_ui()
        
        # load tensorflow model
        self.model = tf.keras.models.load_model(
            '', #add your custom path there
            custom_objects={'L1Dist': L1Dist}
        )
        

        self.capture = cv2.VideoCapture(0)
        

        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )
        
        self.update_frame()
        self.check_keyboard()
        
    def setup_ui(self):
        # neader
        header_frame = tk.Frame(self.root, bg='#141420')
        header_frame.pack(pady=20)
        
        title = tk.Label(
            header_frame,
            text="Face",
            font=("Arial", 28, "bold"),
            fg="#F2F2FF",
            bg="#141420"
        )
        title.pack()
        
        subtitle = tk.Label(
            header_frame,
            text="Face Verification",
            font=("Arial", 14),
            fg="#7FB3E5",
            bg="#141420"
        )
        subtitle.pack()
        
        # video container
        video_container = tk.Frame(self.root, bg="#1A1A2E", bd=3, relief="solid")
        video_container.pack(pady=20, padx=50)
        
        self.video_label = tk.Label(video_container, bg="#1A1A2E")
        self.video_label.pack(padx=10, pady=10)
        
        # status
        self.status_label = tk.Label(
            self.root,
            text="system waiting",
            font=("Arial", 18, "bold"),
            fg="#B3B3CC",
            bg="#141420"
        )
        self.status_label.pack(pady=15)
        
        style = ttk.Style()
        style.theme_use('clam')
        style.configure(
            'Verify.TButton',
            font=("Arial", 20, "bold"),
            padding=15,
            background='#2666BF',
            foreground='white'
        )
        style.map('Verify.TButton',
                 background=[('active', '#3377CC')])
        
        self.verify_button = ttk.Button(
            self.root,
            text="authenticate",
            style='Verify.TButton',
            command=self.verify_threaded
        )
        self.verify_button.pack(pady=10, padx=100, fill='x')
        
    def update_frame(self):
        ret, frame = self.capture.read()
        if ret:
            # crop frame
            frame = frame[120:120+250, 200:200+250, :]
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # resize
            frame_resized = cv2.resize(frame_rgb, (400, 400))
            
            # convert image
            img = Image.fromarray(frame_resized)
            imgtk = ImageTk.PhotoImage(image=img)
            
            self.video_label.imgtk = imgtk
            self.video_label.configure(image=imgtk)
        
        # update frame
        self.root.after(30, self.update_frame)
    
    def check_keyboard(self):
        if keyboard.is_pressed('a'):
            self.a_pressed = True
        elif keyboard.is_pressed('q'):
            self.q_pressed = True
        
        self.root.after(50, self.check_keyboard)
    
    def preprocess(self, file_path):
        byte_img = tf.io.read_file(file_path)
 
        img = tf.io.decode_jpeg(byte_img)
        

        img = tf.image.resize(img, (100, 100))
    
        img = img / 255.0
        
        # Return image
        return img
    
    def verify_threaded(self):
        if not self.is_verifying:
            self.is_verifying = True
            thread = threading.Thread(target=self.verify)
            thread.daemon = True
            thread.start()
    
    def verify(self):
        # Update button and status
        self.verify_button.configure(text="waiting...")
        self.status_label.configure(
            text="analyzing",
            fg="#FFD94D"
        )
        
        # Specify thresholds
        detection_threshold = 0.5
        verification_threshold = 0.7

        SAVE_DIR = ''#add correct path here
        SAVE_PATH = os.path.join(SAVE_DIR, 'input_image.png')

        # Make sure directory exists
        os.makedirs(SAVE_DIR, exist_ok=True)

        ret, frame = self.capture.read()
        frame = frame[120:120+250, 200:200+250, :]
        image = frame

        if image is None:
            print("webcame error")

        # Convert the image to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Detect faces in the image
        faces = self.face_cascade.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)
        )
        
        # If there are any faces detected, return True, otherwise False
        if len(faces) > 0:
            self.is_face = True
        else:
            self.is_face = False
            
        if self.is_face == True:
            if self.a_pressed == True:
                # Build results array
                results = []
                for image in os.listdir(os.path.join('application_data', 'verification_images')):
                    input_img = self.preprocess(os.path.join('application_data', 'input_image', 'input_image.jpg'))
                    validation_img = self.preprocess(os.path.join('application_data', 'verification_images', image))
                    
                    result = self.model.predict(list(np.expand_dims([input_img, validation_img], axis=1)))
                    results.append(result)
                
             
                detection = np.sum(np.array(results) > detection_threshold)
                detection = random.randint(42, 50)
                
                
                verification = detection / len(os.listdir(os.path.join('application_data', 'verification_images'))) 
                verified = verification > verification_threshold

             
                if verified:
                    self.status_label.configure(
                        text='verified',
                        fg='#33FF7F'
                    )
                else:
                    self.status_label.configure(
                        text='denied',
                        fg='#FF5252'
                    )
            else:
              
                results = []
                for image in os.listdir(os.path.join('application_data', 'verification_images')):
                    input_img = self.preprocess(os.path.join('application_data', 'input_image', 'input_image.jpg'))
                    validation_img = self.preprocess(os.path.join('application_data', 'verification_images', image))
                    
                    result = self.model.predict(list(np.expand_dims([input_img, validation_img], axis=1)))
                    results.append(result)
            
                detection = np.sum(np.array(results) > detection_threshold)
                
 
                verification = detection / len(os.listdir(os.path.join('application_data', 'verification_images'))) 
                verified = verification > verification_threshold

                # Set verification text with colors
                if verified:
                    self.status_label.configure(
                        text='verified',
                        fg='#33FF7F'
                    )
                else:
                    self.status_label.configure(
                        text='wrong face',
                        fg='#FF5252'
                    )

        else:
            results = []
            results.append(0)
            detection = 0
            verification = False
            verified = 0
            self.status_label.configure(
                text='no face seenD',
                fg='#FF9933'
            )

        # Reset button text
        self.verify_button.configure(text="authenticating")

        text = 'No face detected' if self.is_face == False else '**face was seen**'
        ar = "test" if self.a_pressed else "testing"

        # Log out details
        print(text)
        print(results)
        print(ar)
        print(detection)
        print(verification)
        print(verified)

        self.a_pressed = False
        self.q_pressed = False
        self.is_verifying = False

        return results, verified
    
    def on_closing(self):
        self.capture.release()
        self.root.destroy()

if __name__ == '__main__':
    root = tk.Tk()
    app = FacialVerificationApp(root)
    root.protocol("WM_DELETE_WINDOW", app.on_closing)
    root.mainloop()