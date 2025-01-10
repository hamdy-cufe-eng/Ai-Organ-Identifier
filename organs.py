import os
import sys
import torch
import torchvision.transforms as transforms
from torchvision import models
from torch.utils.data import DataLoader, Dataset
from PIL import Image
from PyQt5 import QtWidgets, QtGui, QtCore
from PyQt5.QtWidgets import QLabel,QFrame ,QToolBar, QStatusBar,QMainWindow,QApplication # Import QLabel
import torch.nn as nn
import torch.optim as optim

# Custom Dataset
class OrganDataset(Dataset):
    def __init__(self, images, labels, transform=None):
        self.images = images
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)

        return image, label

# Load images and labels from the dataset
def load_data(dataset_path):
    images = []
    labels = []
    organ_classes = ['Liver', 'Brain','Heart','Limbs']
#C:/Users/Hamdy/Downloads/Liver_tumor/images/Limbs
    #Limbs
    for organ in organ_classes:
        organ_path = os.path.join(dataset_path, organ)
        if not os.path.exists(organ_path):
            print(f"Directory not found: {organ_path}")
            continue

        for img_name in os.listdir(organ_path):
            img_path = os.path.join(organ_path, img_name)
            try:
                img = Image.open(img_path).convert("RGB")  # Ensure image is in RGB
                images.append(img)
                labels.append(organ_classes.index(organ))  # Use index as label
            except Exception as e:
                print(f"Error loading image {img_path}: {e}")

    if not images:
        raise ValueError("No images found. Please check the dataset path.")

    return images, labels

# Load and prepare model
def load_model(num_classes):
    model = models.resnet18(pretrained=True)
    model.fc = nn.Linear(model.fc.in_features, num_classes)  # Adjust for number of classes
    return model

# Train the model
def train_model(model, train_loader, criterion, optimizer, num_epochs=5):
    model.train()  # Set the model to training mode
    for epoch in range(num_epochs):
        running_loss = 0.0
        for images, labels in train_loader:
            optimizer.zero_grad()  # Zero the parameter gradients
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()  # Backward pass
            optimizer.step()  # Optimize the weights
            running_loss += loss.item()

        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_loader):.4f}")

# Function to predict an organ from an image
def predict_image(model, image_path, transform):
    model.eval()
    try:
        image = Image.open(image_path).convert("RGB")
        image = transform(image).unsqueeze(0)  # Add batch dimension
    except Exception as e:
        print(f"Error opening image: {e}")
        return None, None

    with torch.no_grad():
        outputs = model(image)
        _, predicted = torch.max(outputs, 1)

    organ_classes = ['Liver', 'Brain','Heart','Limbs']
    return organ_classes[predicted.item()], image.squeeze(0)  # Remove the batch dimension for display

# PyQt5 GUI application
class OrganClassifierApp(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("Organ Classification")
        self.setGeometry(100, 100, 800, 600)

        # Layout
        layout = QtWidgets.QVBoxLayout()

        # Button to browse dataset directory
        self.browse_button = QtWidgets.QPushButton("Browse Dataset Directory")
        self.browse_button.clicked.connect(self.browse_directory)
        layout.addWidget(self.browse_button)

        # Label to display selected dataset path
        self.dataset_label = QLabel("Dataset Directory: None")
        self.dataset_label.setStyleSheet("font-size: 14px;")
        self.dataset_label.setFixedHeight(30)
        layout.addWidget(self.dataset_label)

        line_dataset = QFrame()
        line_dataset.setFrameShape(QFrame.HLine)  # Horizontal line
        line_dataset.setFrameShadow(QFrame.Sunken)
        line_dataset.setStyleSheet(" background-color: rgb(83, 82, 237);")  # Set line color and background
        layout.addWidget(line_dataset)
        # Button to train the model
        self.train_button = QtWidgets.QPushButton("Train Model")
        self.train_button.clicked.connect(self.train_model)
        layout.addWidget(self.train_button)

        # Button to save the model
        self.save_button = QtWidgets.QPushButton("Save Model")
        self.save_button.clicked.connect(self.save_model)
        layout.addWidget(self.save_button)

        # Button to load a model
        self.load_button = QtWidgets.QPushButton("Load Model")
        self.load_button.clicked.connect(self.load_model)
        layout.addWidget(self.load_button)

        # Upload button for image prediction
        self.upload_button = QtWidgets.QPushButton("Upload Image")
        self.upload_button.clicked.connect(self.upload_image)
        layout.addWidget(self.upload_button)

        line_prediction = QFrame()
        line_prediction.setFrameShape(QFrame.HLine)  # Horizontal line
        line_prediction.setFrameShadow(QFrame.Sunken)
        line_prediction.setStyleSheet(" background-color: rgb(83, 82, 237);")  # Set line color and background
        layout.addWidget(line_prediction)

        # Label to display prediction
        self.result_label = QLabel("Predicted Organ: None")
        self.result_label.setStyleSheet("font-size: 14px;")
        self.result_label.setFixedHeight(30)
        layout.addWidget(self.result_label)

        # QLabel to display the uploaded image
        self.image_label = QLabel(self)
        self.image_label.setAlignment(QtCore.Qt.AlignCenter)
        self.image_label.setMinimumSize(224, 300)
        self.image_label.setStyleSheet("border: 1px solid rgb(83, 82, 237); background-color: black;")
        layout.addWidget(self.image_label)
        self.status_bar = QStatusBar()

        status_separator = QFrame()
        status_separator.setFrameShape(QFrame.HLine)  # Vertical line
        status_separator.setFrameShadow(QFrame.Sunken)
        status_separator.setStyleSheet("color: (83, 82, 237); background-color: rgb(9, 132, 227);")

        self.status_bar.addPermanentWidget(status_separator)
        self.status_bar.setStyleSheet("background-color: rgb(9, 132, 227);")
        layout.addWidget(self.status_bar)
        self.setLayout(layout)

        self.dataset_path = None  # Initialize dataset path
        self.images = []
        self.labels = []
        self.model = None

        # Define transformations
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),  # Resize images to 224x224
            transforms.ToTensor(),  # Convert to tensor
        ])

        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = None

    def browse_directory(self):
        self.dataset_path = QtWidgets.QFileDialog.getExistingDirectory(self, "Select Dataset Directory")
        if self.dataset_path:
            self.dataset_label.setText(f"Dataset Directory: {self.dataset_path}")
            self.dataset_label.setStyleSheet("color: rgb(123, 237, 159);font-weight: bold;")
            print(f"Selected dataset directory: {self.dataset_path}")  # Debugging statement
            self.status_bar.showMessage(f"Selected dataset directory: {self.dataset_path}")
            # Load dataset
            try:
                self.images, self.labels = load_data(self.dataset_path)
                # Load model
                self.model = load_model(len(set(self.labels)))  # Number of classes
                self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)  # Set optimizer
                print(f"Loaded {len(self.images)} images and {len(self.labels)} labels.")
                self.status_bar.showMessage(f"Loaded {len(self.images)} images and {len(self.labels)} labels.")
            except Exception as e:
                print(f"Error loading dataset: {e}")
                self.status_bar.showMessage(f"Error loading dataset: {e}")
    def train_model(self):
        if self.model is None or self.images is None or self.labels is None:
            print("Model or dataset not loaded.")
            self.status_bar.showMessage("Model or dataset not loaded.")
            return

        # Create dataset and data loader
        dataset = OrganDataset(self.images, self.labels, transform=self.transform)
        train_loader = DataLoader(dataset, batch_size=8, shuffle=True)

        # Train the model
        train_model(self.model, train_loader, self.criterion, self.optimizer, num_epochs=5)

    def save_model(self):
        if self.model is None:
            print("No model to save.")
            self.status_bar.showMessage("No model to save.")
            return

        file_path, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Save Model", "",
                                                              "PyTorch Model Files (*.pt *.pth)")
        if file_path:
            torch.save(self.model.state_dict(), file_path)
            print(f"Model saved to {file_path}")
            self.status_bar.showMessage(f"Model saved to {file_path}")

    def load_model(self):
        file_path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Load Model", "",
                                                             "PyTorch Model Files (*.pt *.pth)")
        if file_path:
            self.model = load_model(len(set(self.labels)))  # Reinitialize the model
            self.model.load_state_dict(torch.load(file_path,weights_only=True))
            self.model.eval()  # Set the model to evaluation mode
            print(f"Model loaded from {file_path}")
            self.status_bar.showMessage(f"Model loaded from {file_path}")

    def upload_image(self):
        file_path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Select Image", "",
                                                             "Images (*.png *.jpg *.jpeg *.bmp)")
        if file_path and self.model:
            print(f"Selected image: {file_path}")  # Debugging statement
            predicted_organ, image = predict_image(self.model, file_path, self.transform)
            if predicted_organ:
                self.result_label.setText(f"Predicted Organ: {predicted_organ}")
                self.result_label.setStyleSheet("color: rgb(123, 237, 159);font-weight: bold;font-size: 14px;")
                print(f"Predicted Organ: {predicted_organ}")
                self.status_bar.showMessage(f"Prediction complete: {predicted_organ}")
                # Convert the image to a format suitable for QLabel
                try:
                    image = image.permute(1, 2, 0).numpy()  # Convert from CxHxW to HxWxC
                    image = (image * 255).astype('uint8')  # Scale to [0, 255]

                    # Create a QImage from the numpy array
                    height, width, channel = image.shape
                    bytes_per_line = 3 * width
                    qt_image = QtGui.QImage(image.tobytes(), width, height, bytes_per_line, QtGui.QImage.Format_RGB888)

                    # Display the image in QLabel
                    self.image_label.setPixmap(QtGui.QPixmap.fromImage(qt_image))  # Set pixmap in QLabel

                   # Set minimum size for better layout
                except Exception as e:
                    print(f"Error processing image for display: {e}")
            else:
                self.result_label.setText("Error in prediction: Unable to process image.")

stylesheet = """ 
QWidget{ background-color: rgb(47, 53, 66);color: White;}
QLabel{ color: White;}
QPushButton {color: White; }
QTabWidget  {color: White; }
"""


# Run the application
if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setStyle('Fusion')
    app.setStyleSheet(stylesheet)
    classifier_app = OrganClassifierApp()
    classifier_app.show()
    sys.exit(app.exec_())
