import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image

# -------------------------------
# Model Modified LeNet (versi kamu)
# -------------------------------
class ModifiedLeNet(nn.Module):
    def __init__(self, num_classes=4):
        super(ModifiedLeNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, kernel_size=5)     # (6, 252, 252)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5)    # (16, 122, 122)
        self.conv3 = nn.Conv2d(16, 32, kernel_size=3)   # (32, 59, 59)

        self.dropout_conv = nn.Dropout2d(0.25)  # Dropout conv
        self.dropout_fc = nn.Dropout(0.5)       # Dropout FC

        # Hitungan flatten: 32 * 29 * 29
        self.fc1 = nn.Linear(32 * 29 * 29, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, num_classes)

    def forward(self, x):
        x = F.relu(self.conv1(x))         # (6, 252, 252)
        x = F.max_pool2d(x, 2)            # (6, 126, 126)
        x = self.dropout_conv(x)

        x = F.relu(self.conv2(x))         # (16, 122, 122)
        x = F.max_pool2d(x, 2)            # (16, 61, 61)
        x = self.dropout_conv(x)

        x = F.relu(self.conv3(x))         # (32, 59, 59)
        x = F.max_pool2d(x, 2)            # (32, 29, 29)
        x = self.dropout_conv(x)

        x = x.view(-1, 32 * 29 * 29)      # flatten
        x = F.relu(self.fc1(x))
        x = self.dropout_fc(x)

        x = F.relu(self.fc2(x))
        x = self.dropout_fc(x)

        x = self.fc3(x)
        return x


# -------------------------------
# Load Model
# -------------------------------
@st.cache_resource
def load_model():
    model = ModifiedLeNet(num_classes=4)
    model.load_state_dict(torch.load("lenet_model.pth", map_location="cpu"))
    model.eval()
    return model

model = load_model()

# -------------------------------
# Preprocessing
# -------------------------------
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor()
])

# -------------------------------
# Streamlit UI
# -------------------------------
st.title("🔍 Klasifikasi Penyakit Mata dengan LeNet Modifikasi")
st.write("Upload citra fundus untuk diprediksi menggunakan CNN LeNet Modifikasi dengan Dropout.")

uploaded_file = st.file_uploader("Upload gambar fundus", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Citra fundus diupload", use_column_width=True)

    # Preprocess
    img_tensor = transform(image).unsqueeze(0)

    # Predict
    with torch.no_grad():
        outputs = model(img_tensor)
        _, predicted = torch.max(outputs, 1)

    labels = ["Normal", "Cataract", "Glaucoma", "Retina Disease"]
    st.success(f"**Prediksi:** {labels[predicted.item()]}")

    # Probabilitas tiap kelas
    probs = torch.softmax(outputs, dim=1).numpy()[0]
    st.write("### Probabilitas Prediksi:")
    for i, label in enumerate(labels):
        st.write(f"{label}: {probs[i]*100:.2f}%")
