import tkinter as tk
from tkinter import filedialog, ttk, messagebox
from keras.models import load_model
from PIL import Image, ImageTk, ImageOps
import numpy as np
import cv2
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.base import MIMEBase
from email import encoders
import os
from datetime import datetime
import serial
import time

# Load the model
model = load_model("keras_Model.h5", compile=False)

# Load the labels
class_names = [line.strip() for line in open("labels.txt", "r").readlines()]

# Price dictionary
price_dict = {
    'fresh_apples': 100,
    'fresh_banana': 80,
    'fresh_oranges': 60,
    'fresh_tomato': 40,
    'rotten_apples': 50,
    'rotten_banana': 40,
    'rotten_oranges': 30,
    'rotten_tomato': 10,
}

# Email configuration
EMAIL_SENDER = "piyush.lahori.2004@gmail.com"
EMAIL_PASSWORD = "oubotanchqzpmfer"
EMAIL_RECEIVER = "piyush.lahori.ug22@nsut.ac.in"

class FruitDetectionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Fresh & Rotten Fruits Detection System")
        self.root.geometry("1000x800")
        self.root.configure(bg="#f0f0f0")
        
        # Arduino connection variables (fixed to COM8)
        self.arduino = None
        self.arduino_port = "COM3"
        self.arduino_connected = False
        
        # Detection tracking variables
        self.fresh_detection_start = None
        self.rotten_detection_start = None
        self.current_detection = None
        self.fresh_detection_count = 0
        self.rotten_detection_count = 0
        
        self.setup_ui()
        self.cap = None
        self.is_streaming = False
        
        # Try to connect to Arduino on COM8
        self.connect_arduino_fixed()
    
    def connect_arduino_fixed(self):
        """Connect to Arduino on fixed COM8 port"""
        try:
            self.arduino = serial.Serial(self.arduino_port, 9600, timeout=1)
            time.sleep(2)  # Wait for connection to establish
            self.arduino_connected = True
            self.port_label.config(text=f"Port: {self.arduino_port}")
            self.connect_btn.config(text="Disconnect", bg="#F44336")
            self.status_bar.config(text=f"Arduino connected on {self.arduino_port}")
        except Exception as e:
            self.arduino_connected = False
            self.port_label.config(text="Port: Not Connected")
            self.connect_btn.config(text="Connect Arduino", bg="#2196F3")
            self.status_bar.config(text=f"Failed to connect to Arduino on {self.arduino_port}")
    
    def setup_ui(self):
        # Header Frame
        header_frame = tk.Frame(self.root, bg="#4CAF50", height=80)
        header_frame.pack(fill=tk.X, padx=10, pady=10)
        
        self.title_label = tk.Label(
            header_frame, 
            text="Fresh & Rotten Fruits Detection System", 
            font=("Arial", 20, "bold"), 
            bg="#4CAF50", 
            fg="white"
        )
        self.title_label.pack(pady=20)
        
        # Main Content Frame
        content_frame = tk.Frame(self.root, bg="#f0f0f0")
        content_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=10)
        
        # Left Frame (Image Display)
        left_frame = tk.Frame(content_frame, bg="white", bd=2, relief=tk.RIDGE)
        left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        self.image_label = tk.Label(left_frame, bg="white")
        self.image_label.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Right Frame (Controls and Results)
        right_frame = tk.Frame(content_frame, bg="#f0f0f0", width=300)
        right_frame.pack(side=tk.RIGHT, fill=tk.Y, padx=10, pady=10)
        
        # Arduino Connection Frame
        arduino_frame = tk.Frame(right_frame, bg="#f0f0f0", bd=2, relief=tk.RIDGE)
        arduino_frame.pack(fill=tk.X, pady=5)
        
        self.arduino_title = tk.Label(
            arduino_frame, 
            text="Arduino Connection (COM8)", 
            font=("Arial", 12, "bold"), 
            bg="#607D8B", 
            fg="white"
        )
        self.arduino_title.pack(fill=tk.X, pady=(0, 5))
        
        self.port_label = tk.Label(
            arduino_frame, 
            text="Port: Not Connected", 
            font=("Arial", 10), 
            bg="#f0f0f0",
            anchor="w"
        )
        self.port_label.pack(fill=tk.X, padx=5, pady=2)
        
        self.connect_btn = tk.Button(
            arduino_frame, 
            text="Connect Arduino", 
            command=self.toggle_arduino_connection,
            bg="#2196F3",
            fg="white",
            font=("Arial", 10),
            width=15
        )
        self.connect_btn.pack(pady=5)
        
        # Buttons Frame
        btn_frame = tk.Frame(right_frame, bg="#f0f0f0")
        btn_frame.pack(fill=tk.X, pady=10)
        
        self.select_btn = tk.Button(
            btn_frame, 
            text="Select Image", 
            command=self.select_image,
            bg="#2196F3",
            fg="white",
            font=("Arial", 12),
            width=15
        )
        self.select_btn.pack(side=tk.LEFT, padx=5)
        
        self.webcam_btn = tk.Button(
            btn_frame, 
            text="Start Webcam", 
            command=self.toggle_webcam,
            bg="#FF5722",
            fg="white",
            font=("Arial", 12),
            width=15
        )
        self.webcam_btn.pack(side=tk.RIGHT, padx=5)
        
        # Results Frame
        results_frame = tk.Frame(right_frame, bg="white", bd=2, relief=tk.RIDGE)
        results_frame.pack(fill=tk.BOTH, expand=True, pady=10)
        
        self.result_title = tk.Label(
            results_frame, 
            text="Detection Results", 
            font=("Arial", 14, "bold"), 
            bg="#607D8B", 
            fg="white"
        )
        self.result_title.pack(fill=tk.X, pady=(0, 10))
        
        self.class_label = tk.Label(
            results_frame, 
            text="Fruit Type: -", 
            font=("Arial", 12), 
            bg="white",
            anchor="w"
        )
        self.class_label.pack(fill=tk.X, padx=10, pady=5)
        
        self.freshness_label = tk.Label(
            results_frame, 
            text="Freshness: -", 
            font=("Arial", 12), 
            bg="white",
            anchor="w"
        )
        self.freshness_label.pack(fill=tk.X, padx=10, pady=5)
        
        self.price_label = tk.Label(
            results_frame, 
            text="Price: -", 
            font=("Arial", 12), 
            bg="white",
            anchor="w"
        )
        self.price_label.pack(fill=tk.X, padx=10, pady=5)
        
        self.confidence_label = tk.Label(
            results_frame, 
            text="Confidence: -", 
            font=("Arial", 12), 
            bg="white",
            anchor="w"
        )
        self.confidence_label.pack(fill=tk.X, padx=10, pady=5)
        
        # LED Status Frame
        led_frame = tk.Frame(right_frame, bg="white", bd=2, relief=tk.RIDGE)
        led_frame.pack(fill=tk.X, pady=10)
        
        self.led_title = tk.Label(
            led_frame, 
            text="LED Status", 
            font=("Arial", 12, "bold"), 
            bg="#607D8B", 
            fg="white"
        )
        self.led_title.pack(fill=tk.X, pady=(0, 5))
        
        self.green_led = tk.Label(
            led_frame, 
            text="Green LED: OFF", 
            font=("Arial", 10), 
            bg="white",
            anchor="w"
        )
        self.green_led.pack(fill=tk.X, padx=10, pady=2)
        
        self.red_led = tk.Label(
            led_frame, 
            text="Red LED: OFF", 
            font=("Arial", 10), 
            bg="white",
            anchor="w"
        )
        self.red_led.pack(fill=tk.X, padx=10, pady=2)
        
        # Status Bar
        self.status_bar = tk.Label(
            self.root, 
            text="Ready", 
            bd=1, 
            relief=tk.SUNKEN, 
            anchor=tk.W,
            font=("Arial", 10),
            bg="#4CAF50",
            fg="white"
        )
        self.status_bar.pack(fill=tk.X, padx=10, pady=(0, 10))
    
    def toggle_arduino_connection(self):
        """Toggle Arduino connection on COM8"""
        if not self.arduino_connected:
            self.connect_arduino_fixed()
        else:
            try:
                self.arduino.close()
                self.arduino_connected = False
                self.port_label.config(text="Port: Not Connected")
                self.connect_btn.config(text="Connect Arduino", bg="#2196F3")
                self.status_bar.config(text="Arduino disconnected")
                # Turn off LEDs when disconnecting
                self.green_led.config(text="Green LED: OFF")
                self.red_led.config(text="Red LED: OFF")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to disconnect: {str(e)}")
    
    def control_led(self, led, state):
        """Send command to Arduino to control LED"""
        if self.arduino_connected and self.arduino:
            try:
                command = f"{led}_{state}\n"
                self.arduino.write(command.encode())
                
                # Update LED status in UI
                if led == "GREEN":
                    self.green_led.config(text=f"Green LED: {state}")
                elif led == "RED":
                    self.red_led.config(text=f"Red LED: {state}")
                
            except Exception as e:
                self.status_bar.config(text=f"Error controlling LED: {str(e)}")
    
    def track_detection(self, class_name):
        """Track continuous detection of fresh or rotten fruits"""
        current_time = time.time()
        
        # Check if detection has changed
        if self.current_detection != class_name:
            self.current_detection = class_name
            if "fresh" in class_name:
                self.fresh_detection_start = current_time
                self.rotten_detection_start = None
                self.rotten_detection_count = 0
                self.fresh_detection_count = 1
            elif "rotten" in class_name:
                self.rotten_detection_start = current_time
                self.fresh_detection_start = None
                self.fresh_detection_count = 0
                self.rotten_detection_count = 1
            return
        
        # Count continuous detections
        if "fresh" in class_name:
            self.fresh_detection_count += 1
            if self.fresh_detection_start is None:
                self.fresh_detection_start = current_time
                
            # Check for 10 seconds of continuous fresh detection
            if (current_time - self.fresh_detection_start) >= 10:
                self.control_led("GREEN", "ON")
                self.control_led("RED", "OFF")
                self.fresh_detection_start = current_time  # Reset timer but keep counting
                
        elif "rotten" in class_name:
            self.rotten_detection_count += 1
            if self.rotten_detection_start is None:
                self.rotten_detection_start = current_time
                
            # Check for 15 seconds of continuous rotten detection
            if (current_time - self.rotten_detection_start) >= 15:
                self.control_led("RED", "ON")
                self.control_led("GREEN", "OFF")
                self.rotten_detection_start = current_time  # Reset timer but keep counting
    
    def select_image(self):
        if self.is_streaming:
            self.toggle_webcam()
            
        file_path = filedialog.askopenfilename()
        if file_path:
            image = Image.open(file_path)
            self.classify_image(image)
    
    def toggle_webcam(self):
        if not self.is_streaming:
            self.cap = cv2.VideoCapture(0)
            if not self.cap.isOpened():
                messagebox.showerror("Error", "Unable to open webcam.")
                return
            
            self.is_streaming = True
            self.webcam_btn.config(text="Stop Webcam", bg="#F44336")
            self.update_webcam()
        else:
            self.is_streaming = False
            if self.cap:
                self.cap.release()
            self.webcam_btn.config(text="Start Webcam", bg="#FF5722")
    
    def update_webcam(self):
        if self.is_streaming:
            ret, frame = self.cap.read()
            if ret:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                image = Image.fromarray(frame)
                self.classify_image(image)
                
                # Schedule the next update
                self.root.after(50, self.update_webcam)
            else:
                self.toggle_webcam()
                messagebox.showerror("Error", "Unable to read frame from webcam.")
    
    def classify_image(self, image):
        # Resize the image
        size = (224, 224)
        image = ImageOps.fit(image, size, Image.LANCZOS)
        
        # Display the input image
        img_tk = ImageTk.PhotoImage(image)
        self.image_label.config(image=img_tk)
        self.image_label.image = img_tk
        
        # Convert to numpy array and normalize
        image_array = np.asarray(image)
        normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1
        
        # Prepare data for prediction
        data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
        data[0] = normalized_image_array
        
        # Make prediction
        prediction = model.predict(data)
        index = np.argmax(prediction)
        confidence = prediction[0][index]
        
        if index < len(class_names):
            class_name = class_names[index]
            
            # Track continuous detection
            self.track_detection(class_name)
            
            # Update results
            if "_" in class_name:
                fruit_type, freshness = class_name.split("_")
                self.class_label.config(text=f"Fruit Type: {fruit_type.capitalize()}")
                self.freshness_label.config(text=f"Freshness: {freshness.capitalize()}")
                
                # Set price based on freshness
                price = price_dict.get(class_name, 0)
                self.price_label.config(text=f"Price: ₹{price}")
                
                # Check if rotten and send email
                if "rotten" in class_name.lower():
                    self.send_email_notification(image, class_name)
                    self.status_bar.config(text=f"Rotten {fruit_type} detected! Email sent.")
                else:
                    self.status_bar.config(text=f"Fresh {fruit_type} detected.")
            else:
                self.class_label.config(text=f"Fruit Type: {class_name.capitalize()}")
                self.freshness_label.config(text="Freshness: -")
                self.price_label.config(text="Price: -")
                self.status_bar.config(text=f"{class_name} detected.")
            
            self.confidence_label.config(text=f"Confidence: {confidence*100:.2f}%")
    
    def send_email_notification(self, image, class_name):
        try:
            # Save the image temporarily
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"rotten_fruit_{timestamp}.jpg"
            image.save(filename)
            
            # Create email message
            msg = MIMEMultipart()
            msg['From'] = EMAIL_SENDER
            msg['To'] = EMAIL_RECEIVER
            msg['Subject'] = f"Rotten Fruit Detected: {class_name}"
            
            body = f"""
            <html>
                <body>
                    <h2>Rotten Fruit Detection Alert</h2>
                    <p>A rotten fruit has been detected by the system.</p>
                    <p><strong>Details:</strong></p>
                    <ul>
                        <li>Fruit Type: {class_name.split('_')[0].capitalize()}</li>
                        <li>Status: Rotten</li>
                        <li>Detection Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</li>
                    </ul>
                    <p>Please see the attached image for reference.</p>
                </body>
            </html>
            """
            
            msg.attach(MIMEText(body, 'html'))
            
            # Attach the image
            with open(filename, 'rb') as attachment:
                part = MIMEBase('application', 'octet-stream')
                part.set_payload(attachment.read())
                encoders.encode_base64(part)
                part.add_header(
                    'Content-Disposition',
                    f'attachment; filename= {filename}',
                )
                msg.attach(part)
            
            # Send email
            with smtplib.SMTP('smtp.gmail.com', 587) as server:
                server.starttls()
                server.login(EMAIL_SENDER, EMAIL_PASSWORD)
                server.send_message(msg)
            
            # Remove temporary file
            os.remove(filename)
            
        except Exception as e:
            messagebox.showerror("Email Error", f"Failed to send email: {str(e)}")
    
    def on_closing(self):
        if self.cap:
            self.cap.release()
        if self.arduino and self.arduino_connected:
            self.arduino.close()
        self.root.destroy()

if __name__ == "__main__":
    root = tk.Tk()
    app = FruitDetectionApp(root)
    root.protocol("WM_DELETE_WINDOW", app.on_closing)
    root.mainloop()
