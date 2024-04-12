import cv2
import numpy as np
import pandas as pd
import os
import sys
import math
import time
import shutil
from PyQt5.QtWidgets import QApplication, QPushButton, QFileDialog, QLabel, QTextEdit, QWidget, QVBoxLayout, QSizePolicy, QStackedWidget
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QPixmap, QFont, QPalette, QIcon
from split_image import split_image

def calculate_fft_features_time(channel): # Calculate statistics for time domain
    feature_vector = []

    mean_real = np.mean(channel)
    std_real = np.std(channel)
    h, w = channel.shape
    x, y = np.meshgrid(np.arange(w), np.arange(h))
    epsilon = 1e-8  # small positive value
    total_weight_real = np.sum(channel) + epsilon
    centroid_x_real = np.sum(x * channel) / total_weight_real
    centroid_y_real = np.sum(y * channel) / total_weight_real
    variance_x_real = max(np.sum(((x - centroid_x_real) ** 2) * channel) / total_weight_real, epsilon)
    variance_y_real = max(np.sum(((y - centroid_y_real) ** 2) * channel) / total_weight_real, epsilon)
    dispersion_x_real = np.sqrt(variance_x_real) if variance_x_real >= 0 else 0
    dispersion_y_real = np.sqrt(variance_y_real) if variance_y_real >= 0 else 0
    feature_vector.extend([mean_real, std_real, dispersion_x_real, dispersion_y_real])       # mean, std, horizontal distribution, vertical distribution for real part
    
    return feature_vector

def feature_extraction_time(image):  # Obtain Feature Vector that contains 24 feature values from image for time domain
    # Convert the image to float for more precise calculations
    image = image.astype(np.float64)

    img_blur = cv2.GaussianBlur(image, (3, 3), 0, borderType=0)
    B0, G0, R0 = cv2.split(img_blur)

    sobel_x = cv2.Sobel(img_blur, cv2.CV_64F, 1, 0, ksize=3, borderType=0)
    sobel_y = cv2.Sobel(img_blur, cv2.CV_64F, 0, 1, ksize=3, borderType=0)

    sobel_x = np.absolute(sobel_x)
    sobel_y = np.absolute(sobel_y)

    sobel_x = np.uint8(sobel_x)
    sobel_y = np.uint8(sobel_y)

    # Combine the gradients
    sobel_combined = cv2.addWeighted(sobel_x, 0.5, sobel_y, 0.5, 0)
    B1, G1, R1 = cv2.split(sobel_combined)

    # Calculate features
    r0_features = calculate_fft_features_time(R0)
    g0_features = calculate_fft_features_time(G0)
    b0_features = calculate_fft_features_time(B0)

    r1_features = calculate_fft_features_time(R1)
    g1_features = calculate_fft_features_time(G1)
    b1_features = calculate_fft_features_time(B1)

    # Combine the features into a single feature vector
    feature_vector = np.concatenate([r0_features, g0_features, b0_features, r1_features, g1_features, b1_features]) # 24 feature values

    return feature_vector

def calculate_fft_features_freq(channel):  # Calculate statistics for frequency domain
    # Initialize the feature vector
    feature_vector = []

    # Compute the FFT of the channel
    f = np.fft.fft2(channel)

    # Split the FFT output into its real and imaginary parts
    real_part = f.real
    imaginary_part = f.imag

    # Set a threshold
    threshold = 10 
    # Apply the threshold to the real and imaginary parts
    real_part[real_part < threshold] = 0
    imaginary_part[imaginary_part < threshold] = 0

    # Calculate statistics for the real part
    mean_real = np.mean(real_part)
    std_real = np.std(real_part)
    h, w = real_part.shape
    x, y = np.meshgrid(np.arange(w), np.arange(h))
    epsilon = 1e-8  # small positive value
    total_weight_real = np.sum(real_part) + epsilon      
    centroid_x_real = np.sum(x * real_part) / total_weight_real
    centroid_y_real = np.sum(y * real_part) / total_weight_real
    variance_x_real = max(np.sum(((x - centroid_x_real) ** 2) * real_part) / total_weight_real, epsilon)
    variance_y_real = max(np.sum(((y - centroid_y_real) ** 2) * real_part) / total_weight_real, epsilon)
    dispersion_x_real = np.sqrt(variance_x_real) if variance_x_real >= 0 else 0
    dispersion_y_real = np.sqrt(variance_y_real) if variance_y_real >= 0 else 0
    feature_vector.extend([mean_real, std_real, dispersion_x_real, dispersion_y_real])       # mean, std, horizontal distribution, vertical distribution for real part

    # Calculate statistics for the imaginary part
    mean_imag = np.mean(imaginary_part)
    std_imag = np.std(imaginary_part)
    total_weight_imag = np.sum(imaginary_part) + 1e-8
    centroid_x_imag = np.sum(x * imaginary_part) / total_weight_imag
    centroid_y_imag = np.sum(y * imaginary_part) / total_weight_imag
    variance_x_imag = max(np.sum(((x - centroid_x_imag) ** 2) * imaginary_part) / total_weight_imag, epsilon)
    variance_y_imag = max(np.sum(((y - centroid_y_imag) ** 2) * imaginary_part) / total_weight_imag, epsilon)
    dispersion_x_imag = np.sqrt(variance_x_imag) if variance_x_imag >= 0 else 0
    dispersion_y_imag = np.sqrt(variance_y_imag) if variance_y_imag >= 0 else 0
    feature_vector.extend([mean_imag, std_imag, dispersion_x_imag, dispersion_y_imag])       # mean, std, horizontal distribution, vertical distribution for imaginary part

    return feature_vector

def feature_extraction_freq(image): # Obtain Feature Vector that contains 24 feature values from image for frequency domain
    # Convert the image to float for more precise calculations
    image = image.astype(np.float64)

    # Separate the RGB channels
    b_channel, g_channel, r_channel = cv2.split(image)

    # Calculate FFT features for each channel
    r_fft_features = calculate_fft_features_freq(r_channel)
    g_fft_features = calculate_fft_features_freq(g_channel)
    b_fft_features = calculate_fft_features_freq(b_channel)

    # Combine the features into a single feature vector
    feature_vector = np.concatenate([r_fft_features, g_fft_features, b_fft_features]) # 24 feature values

    return feature_vector

def resize_and_pad_image(image, target_size): # Resize image while maintaining aspect ratio and pad the remaining pixels white
    # Calculate the aspect ratio of the original image
    original_height, original_width, _ = image.shape
    target_width, target_height = target_size
    aspect_ratio = original_width / original_height

    # Calculate the new size while maintaining the aspect ratio
    if aspect_ratio > target_width / target_height:
        new_width = target_width
        new_height = int(target_width / aspect_ratio)
    else:
        new_width = int(target_height * aspect_ratio)
        new_height = target_height

    # Resize the image while maintaining its aspect ratio
    resized_image = cv2.resize(image, (new_width, new_height))

    # Create a canvas of the target size and place the resized image on it
    #canvas = np.zeros((target_height, target_width, 3), dtype=np.uint8) # black padding
    canvas = np.ones((target_height, target_width, 3), dtype=np.uint8)*255 # white padding

    x_offset = (target_width - new_width) // 2
    y_offset = (target_height - new_height) // 2
    canvas[y_offset:y_offset + new_height, x_offset:x_offset + new_width] = resized_image

    return canvas
    
def extract_features_from_directory(directory_path):  # Obtain an array of Feature Vectors of images in a folder
    # Your code for extracting features from directory goes here
    all_files = os.listdir(directory_path)
    image_files = [file for file in all_files if file.lower().endswith((".jpg", ".jpeg", ".png", ".jfif"))]
    feature_vectors = []
    for image_file in image_files:
        image_path = os.path.join(directory_path, image_file)
        image = cv2.imread(image_path)
        target_size = (640, 640)
        resized_and_padded_image = resize_and_pad_image(image, target_size)
        folder_name_train = os.path.basename(directory_path) 

        if folder_name_train == "LED_Black" or folder_name_train == "LED_Red":  
            feature_vector = feature_extraction_freq(resized_and_padded_image)          
            feature_vectors.append(feature_vector)

        elif folder_name_train == "LED_Blue" or folder_name_train == "LED_Green":  
            feature_vector = feature_extraction_time(resized_and_padded_image)          
            feature_vectors.append(feature_vector)

    return np.array(feature_vectors), image_files

def get_mean_and_variance(feature_vectors): # Calculate Mean Feature Vector and Variance for a given set of Feature Vectors
    mean_feature_vector = np.mean(feature_vectors, axis=0)
    error_vectors = feature_vectors - mean_feature_vector
    squared_distances = np.sum(error_vectors**2, axis=1)
    variance = np.mean(squared_distances)
    return mean_feature_vector, variance


class ImageRecognitionGUI(QWidget):

    def __init__(self):
        super().__init__()
        
        self.folder_names_list = []  # Initialize an empty list to store folder names

        # Global Font and Size
        font = QFont("Arial", 12)
        app.setFont(font)

        # Setting Window Properties
        self.setWindowTitle('Task 5: Localisation')
        self.resize(800, 800)        # GUI Size
        self.setWindowIcon(QIcon('player.png'))

        p =self.palette()
        p.setColor(QPalette.Window, Qt.white)
        self.setPalette(p)

        self.initUI()

        self.stop_sampling = False

        self.class_features = {}  # {"ClassName": {"mean": ..., "variance": ...}, ...}

        # self.layout = QGridLayout()
        self.show()

    def initUI(self):       # UI Function

        # Add Training Sets Button
        self.load_train_btn = QPushButton('Add Training Sets', self)
        self.load_train_btn.resize(180, 30)
        self.load_train_btn.move(50, 50)
        self.load_train_btn.clicked.connect(self.load_training_set)
        self.load_train_btn.setStyleSheet("""
            QPushButton {
                background-color: #e8e8e8;
                border: 2px solid #1e90ff;
                border-radius: 10px;
                color: black;
                padding: 5px;
            }
            QPushButton:hover {
                background-color: #1e90ff;
                color: white;
            }
        """)

        # Excel Training Set Button
        self.excel_training_set_btn = QPushButton('Load Training Sets', self)
        self.excel_training_set_btn.resize(180, 30)
        self.excel_training_set_btn.move(50, 90)
        self.excel_training_set_btn.clicked.connect(self.excel_training_set)
        self.excel_training_set_btn.setStyleSheet("""
            QPushButton {
                background-color: #e8e8e8;
                border: 2px solid #ff311e;
                border-radius: 10px;
                color: black;
                padding: 5px;
            }
            QPushButton:hover {
                background-color: #ff311e;
                color: white;
            }
        """)

        # Image Sampling Button
        self.sampling_btn = QPushButton('Image Sampling', self)
        self.sampling_btn.resize(180, 30)
        self.sampling_btn.move(50, 170)
        self.sampling_btn.clicked.connect(self.sampling)

        # Display Result Box
        self.result_label = QTextEdit(self)
        self.result_label.move(250, 50)
        self.result_label.setFixedSize(500, 700)     # blue box size 
        self.result_label.setAlignment(Qt.AlignTop)
        self.result_label.setStyleSheet("""
            QTextEdit {
                background-color: white;
                #background-color: #f0f0f0;
                border: 2px solid #1e90ff;
                border-radius: 10px;
                color: #000000;
                padding: 5px;
            }
        """)

        self.img_preview = QLabel()

        # create label
        self.label = QLabel()
        self.label.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        
        # Take a Picture Button
        self.take_image_btn = QPushButton('Take a Picture', self)
        self.take_image_btn.resize(180, 30)
        self.take_image_btn.move(50, 130)
        self.take_image_btn.clicked.connect(self.take_image)

        self.stack1 = QWidget()
        self.Stack = QStackedWidget (self)
        self.Stack.addWidget (self.stack1)

        self.show()

        # Initialization data
        self.new_images_features = []


    def load_training_set(self):    # Display Mean Feature Vector and Variance on GUI for each training set
        self.Stack.setCurrentIndex(0)
        blank_layout = QVBoxLayout()

        folder_name = QFileDialog.getExistingDirectory(self, 'Add Training Sets')
        if folder_name:
            self.folder_names_list.append(folder_name)  
            # Using folder names as category names
            class_name = os.path.basename(os.path.normpath(folder_name))

            features, images = extract_features_from_directory(folder_name)
            mean_vector, variance_val = get_mean_and_variance(features)
            self.class_features[class_name] = {"mean": mean_vector, "variance": variance_val}
            
            # Save mean_vector and variance_val into excel
            df = pd.DataFrame({'mean_vector': mean_vector, 'variance_val': variance_val}) # Create a DataFrame for mean_vector and variance_val
            excel_file_path = os.path.join(r"C:\Users\User\Desktop\GUI_Windows", f'{class_name}_mean_variance_data.xlsx') # Generate the Excel file path based on class_name 
            df.to_excel(excel_file_path, index=False) # Save the DataFrame to the Excel file

            # Add feature vectors to existing text and display them on the GUI
            # Fetch current displayed text
            current_displayed_text = self.result_label.toHtml()

            # Format the new data
            header = f"<font size='5' color='red'><b>{class_name}:</b></font><br>"

            mean_str = ', '.join([f"{val:.3f}" for val in mean_vector])    

            new_data_strings = [
            header,
            f"<font size='5'<b>Overall Mean of Features:</b></font><br>",
            f"<font size='5'<b>{mean_str}</b></font><br>",
            f"<font size='5'<b>Overall Variance: {variance_val:.3f}</b></font><br>",
            "<hr>"  # Add a line for visual separation
            ]

            # Add new data to current displayed text and set the text
            self.result_label.setText(current_displayed_text + "<br>".join(new_data_strings))

        self.stack1.setLayout(blank_layout)


    def excel_training_set(self):     # Extract Mean Feature Vector and Variance from Excel file and Display on GUI for each training set
        self.Stack.setCurrentIndex(1)
        blank_layout = QVBoxLayout()

        class_list = ["LED_Black", "LED_Blue", "LED_Green", "LED_Red"]
        for class_name in class_list:
            # Using folder names as category names
            excel_file_path = os.path.join(r"C:\Users\User\Desktop\GUI_Windows", f'{class_name}_mean_variance_data.xlsx') 
            df = pd.read_excel(excel_file_path)
            # Extract mean_vector and variance_val columns from the DataFrame
            mean_vector = np.array(df['mean_vector'])
            variance_val = float(df['variance_val'].iloc[0])

            self.class_features[class_name] = {"mean": mean_vector, "variance": variance_val}

            # Add feature vectors to existing text and display them on the GUI
            # Fetch current displayed text
            current_displayed_text = self.result_label.toHtml()

            # Format the new data
            header = f"<font size='5' color='red'><b>{class_name}:</b></font><br>"
            mean_str = ', '.join([f"{val:.3f}" for val in mean_vector])    

            new_data_strings = [
            header,
            f"<font size='5'<b>Overall Mean of Features:</b></font><br>",
            f"<font size='5'<b>{mean_str}</b></font><br>",
            f"<font size='5'<b>Overall Variance: {variance_val:.3f}</b></font><br>",
            "<hr>"  # Add a line for visual separation
            ]
            # Add new data to current displayed text and set the text
            self.result_label.setText(current_displayed_text + "<br>".join(new_data_strings))
            
        self.stack1.setLayout(blank_layout)


    def select_new_images(self):    # Display each image and filename on GUI

        folder_name = r"C:\Users\User\Desktop\GUI_Windows\split_images"

        if folder_name:
            self.new_images_features = []
            self.new_images_filenames = []

            all_files = os.listdir(folder_name)
            #image_files = [file for file in all_files if file.lower().endswith((".jpg", ".jpeg", ".png", ".jfif")) and file!="image1_0.jpg"]
            image_files = [file for file in all_files if file.lower().endswith((".jpg", ".jpeg", ".png", ".jfif"))]

            image_layout = QVBoxLayout()

            #self.result_label.setLayout(image_layout)

            for i, image_file in enumerate(image_files):
                image_path = os.path.join(folder_name, image_file)
  
                # Image
                pixmap = QPixmap(image_path)
                pixmap = pixmap.scaled(110, 100)  # Adjust the size of the image
                label = QLabel()
                label.setPixmap(pixmap)
                image_layout.addWidget(label)  # Add the image label to the vertical layout

                # File Name
                filename_label = QLabel(image_file)
                filename_label.setAlignment(Qt.AlignVCenter)  # Center align the filename
                image_layout.addWidget(filename_label)
            

            self.result_label.setLayout(image_layout)
            self.show()


    def take_image(self):      # Take 1 Image and save as image1.jpg (does not work for Arducam 64MP)
        output_folder = "captured_images"
        #interval = 1
        total_images = 1
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)

        # Initialize the camera
        camera = cv2.VideoCapture(0)

        # Check if the camera is opened successfully
        if not camera.isOpened():
            print("Error: Failed to open camera.")
            return

        # Start capturing images
        for i in range(total_images):
            # Capture image
            ret, frame = camera.read()
            if not ret:
                print("Failed to capture image")
                break

            # Save image
            image_name = f"image{i+1}.jpg"
            image_path = os.path.join(output_folder, image_name)
            cv2.imwrite(image_path, frame)
            print(f"{image_name} captured and saved.")

            # Wait for the specified interval
            # time.sleep(interval)

        # Release the camera
        camera.release()


    def splitting_image(self, rows, cols):    # Split Images and move the images to split_images folder 

        folder_path = r"C:\Users\User\Desktop\GUI_Windows\split_images"
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        source_dir = r"C:\Users\User\Desktop\GUI_Windows"
        destination_dir = r"C:\Users\User\Desktop\GUI_Windows\split_images"

        # self.remove_images()
        # time.sleep(1)

        split_image("image1.jpg", rows, cols, False, False)

        image_filenames = []

        for row in range(rows):
            for col in range(cols):
                image_filenames.append(f"image1_{row * cols + col}.jpg")

        for filename in image_filenames:
            source_path = os.path.join(source_dir, filename)
            destination_path = os.path.join(destination_dir, filename)
            shutil.move(source_path, destination_path)


    def calculate_possibility(self):    # Calculate Possibility for each image in Time and Freq domain
        
        folder_name = r"C:\Users\User\Desktop\GUI_Windows\split_images"

        if folder_name:
            self.new_images_features = []
            self.new_images_filenames = []

            all_files = os.listdir(folder_name)
            image_files = [file for file in all_files if file.lower().endswith((".jpg", ".jpeg", ".png", ".jfif"))]

            for i, image_file in enumerate(image_files):
                image_path = os.path.join(folder_name, image_file)
                image = cv2.imread(image_path)
                target_size = (640, 640)
                #resized_and_padded_image = resize_and_pad_image(image, target_size)     # pad
                
                resized_image = cv2.resize(image, target_size)          # no pad

                feature_vector_time = feature_extraction_time(resized_image)     
                feature_vector_freq= feature_extraction_freq(resized_image) 

                #feature_vector = feature_extraction(resized_and_padded_image)     # pad

                self.new_images_features.append(feature_vector_time)
                self.new_images_features.append(feature_vector_freq)

                self.new_images_filenames.append(image_file)

        self.new_images_possibilities = {}

        for classname, data in self.class_features.items():
            mu_f = data['mean']
            variance_f = data['variance']
            possibilities_time, possibilities_freq= [], [] 

            time_features = self.new_images_features[::2]  # Select every other element starting from index 0
            freq_features = self.new_images_features[1::2]  # Select every other element starting from index 1

            for f_new_time, f_new_freq in zip(time_features, freq_features):
                error_time = f_new_time - mu_f
                error_freq = f_new_freq - mu_f
                
                exponent_time = -0.5 * np.dot(error_time, error_time) / variance_f
                exponent_freq = -0.5 * np.dot(error_freq, error_freq) / variance_f
                
                possibility_time = math.exp(exponent_time)
                possibility_freq = math.exp(exponent_freq)
                
                possibilities_time.append(possibility_time)
                possibilities_freq.append(possibility_freq)

            self.new_images_possibilities[classname] = {"time": possibilities_time, "freq": possibilities_freq}  

        self.recognize_image()  # Call the recognize_image method at the end of the calculate_possibility method
    
    def recognize_image(self):  # Between the prob of time and freq domain, choose the prob that is > 0.15
                                # Once prob >0.4, stops image sampling, outputs the prob in GUI
        results = []

        for idx, filename in enumerate(self.new_images_filenames):
            result = "none"
            highest_similarity = -np.inf
            probs_strings = []

            for classname, probs in self.new_images_possibilities.items():
                prob_time = probs["time"]
                prob_freq = probs["freq"]

                prob = prob_time[idx] if any(p > 0.15 for p in prob_time) else prob_freq[idx]
                probs_strings.append(f"{classname}: {prob:.4f}")

                if prob > highest_similarity:  # add training sets with Blue > Green > Red > Black
                    highest_similarity = prob

                    if classname == "LED_Black" and highest_similarity > 0.45:    
                        result = "LED Panel"
                        self.stop_sampling = True
                        break
                    elif classname == "LED_Blue" and highest_similarity > 0.4: 
                        result = "LED Panel"
                        self.stop_sampling = True
                        break
                    elif classname == "LED_Red" and highest_similarity > 0.4: 
                        result = "LED Panel"
                        self.stop_sampling = True
                        break
                    elif classname == "LED_Green" and highest_similarity > 0.4:     
                        result = "LED Panel"
                        self.stop_sampling = True
                        break   

            # if highest_similarity < 0.03:  # Check the threshold
            #     result = "none"
            
            prob_str = f"<br>  &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; ".join(probs_strings)

            # Display possibilities and best match in separate lines
            results.append(
                #f"<span style='color: white; font-size: 16px;'>dadadaddadadaadadada<span style='color: black; font-size: 16px;'>Possibility: {prob_str}<br>" # prob_str 
                f"<span style='color: white; font-size: 16px;'>dadadaddadadaadadada<span style='color: black; font-size: 16px;'>Possibility: {prob:.3f}<br>" # prob_str 
                f"<span style='color: white; font-size: 16px;'>dadadaddadadaadadada<span style='color: red; font-size: 16px;'><b>Result: {result}</b></span>"
                f"<br>"
                f"<br>"
                f"<br>"
                f"<br>"
                f"<br>"
            )
        # Combine all image results
        self.result_label.setText("<br><br>".join(results))


    def remove_images(self):    # Remove images in split_images folder
        folder_path = r"C:\Users\User\Desktop\GUI_Windows\split_images"
        if os.path.exists(folder_path):
            # Get the list of files in the directory
            all_files = os.listdir(folder_path)
        
            # Iterate over the files and remove them
            for file_name in all_files:
                file_path = os.path.join(folder_path, file_name)
                os.remove(file_path)


    def sampling(self):        # Image Sampling
        sk_list = [[1,1],[2,1],[1,2],[3,1],[1,3],[4,1],[1,4],[2,2],[5,1],[1,5]]
        #sk_list = [[1,4]]

        for rows, cols in sk_list:
            if self.stop_sampling == True:
                break
            self.remove_images()
            time.sleep(1)
            self.splitting_image(rows,cols)
            time.sleep(1)
            self.calculate_possibility()
            if self.stop_sampling == True:
                break

        self.select_new_images()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = ImageRecognitionGUI()
    sys.exit(app.exec_())
