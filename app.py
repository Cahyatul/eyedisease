import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import numpy as np

# Definisi ulang ModifiedLeNet (harus sama persis)
class ModifiedLeNet(nn.Module):
    def __init__(self, num_classes=4):
        super(ModifiedLeNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, kernel_size=5)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5)
        self.conv3 = nn.Conv2d(16, 32, kernel_size=3)

        self.dropout_conv = nn.Dropout2d(0.25)
        self.dropout_fc = nn.Dropout(0.5)

        self.fc1 = nn.Linear(32 * 29 * 29, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, num_classes)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2)
        x = self.dropout_conv(x)

        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)
        x = self.dropout_conv(x)

        x = F.relu(self.conv3(x))
        x = F.max_pool2d(x, 2)
        x = self.dropout_conv(x)

        x = x.view(-1, 32 * 29 * 29)
        x = F.relu(self.fc1(x))
        x = self.dropout_fc(x)

        x = F.relu(self.fc2(x))
        x = self.dropout_fc(x)

        x = self.fc3(x)
        return x

# Load model
@st.cache_resource
def ModifiedLeNet():
    model = ModifiedLeNet(num_classes=4)
    model.load_state_dict(torch.load("ModifiedLeNet_model.pth", map_location=torch.device('cpu')))
    model.eval()
    return model

model = ModifiedLeNet()

classes = ["Normal", "Diabetic Retinopathy", "Cataract", "Glaucoma"]

# Streamlit UI
st.title("Prediksi Penyakit Mata (LeNet Modifikasi - PyTorch)")
uploaded_file = st.file_uploader("Upload gambar fundus", type=["jpg", "png", "jpeg"])

if uploaded_file:
    img = Image.open(uploaded_file).convert("RGB")
    st.image(img, caption="Gambar diupload", use_column_width=True)

    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor()
    ])

    img_tensor = transform(img).unsqueeze(0)  # tambah batch dim
    outputs = model(img_tensor)
    _, predicted = torch.max(outputs, 1)
    probs = torch.softmax(outputs, dim=1).detach().numpy()

    st.subheader("Hasil Prediksi:")
    st.success(f"{classes[predicted.item()]}")

    st.subheader("Probabilitas Tiap Kelas:")
    for i, c in enumerate(classes):
        st.write(f"{c}: {probs[0][i]*100:.2f}%")
