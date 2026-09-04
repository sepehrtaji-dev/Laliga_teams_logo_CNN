import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image, ImageTk
import tkinter as tk
from tkinter import filedialog, messagebox

IMG_SIZE = 128
NUM_CLASSES = 20
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class_to_idx = {'Athletic Bilbao': 0, 'Atletico Madrid': 1, 'Barcelona': 2, 'Celta Vigo': 3,
                 'Deportivo Alaves': 4, 'Espanyol': 5, 'Getafe': 6, 'Girona': 7, 'Las Palmas': 8,
                 'Leganes': 9, 'Mallorca': 10, 'Osasuna': 11, 'Rayo Vallecano': 12, 'Real Betis': 13,
                 'Real Madrid': 14, 'Real Sociedad': 15, 'Real Valladolid': 16, 'Sevilla': 17,
                 'Valencia': 18, 'Villarreal': 19}
idx_to_class = {v: k for k, v in class_to_idx.items()}

class LaLigaCNN(nn.Module):
    def __init__(self, num_classes=20):
        super(LaLigaCNN, self).__init__()

        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 8 * 8, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                          std=[0.229, 0.224, 0.225])
])

model = LaLigaCNN(num_classes=NUM_CLASSES).to(DEVICE)
model.load_state_dict(torch.load("best_laliga_model.pth", map_location=DEVICE))
model.eval()

def predict(image_path):
    img = Image.open(image_path).convert("RGB")
    tensor = transform(img).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        outputs = model(tensor)
        probs = torch.softmax(outputs, dim=1)
        conf, pred = torch.max(probs, 1)

    return idx_to_class[pred.item()], conf.item()

class LogoApp:
    def __init__(self, root):
        self.root = root
        self.root.title("La Liga Logo Classifier")
        self.root.geometry("400x500")
        self.root.resizable(False, False)

        self.title_label = tk.Label(root, text="La Liga Logo Classifier", font=("Arial", 16, "bold"))
        self.title_label.pack(pady=10)

        self.image_frame = tk.Frame(root, bg="#eeeeee", width=300, height=300)
        self.image_frame.pack(pady=10)
        self.image_frame.pack_propagate(False)

        self.image_label = tk.Label(self.image_frame, bg="#eeeeee", text="No image selected")
        self.image_label.pack(expand=True)

        self.browse_btn = tk.Button(root, text="Browse Image", command=self.browse_image,
                                     font=("Arial", 12), bg="#4CAF50", fg="white", padx=10, pady=5)
        self.browse_btn.pack(pady=10)

        self.result_label = tk.Label(root, text="logo : —", font=("Arial", 14, "bold"))
        self.result_label.pack(pady=10)

        self.conf_label = tk.Label(root, text="", font=("Arial", 10), fg="gray")
        self.conf_label.pack()

    def browse_image(self):
        file_path = filedialog.askopenfilename(
            filetypes=[("Image files", "*.png *.jpg *.jpeg *.bmp *.webp")]
        )
        if not file_path:
            return

        try:
            img = Image.open(file_path).convert("RGB")
            display_img = img.copy()
            display_img.thumbnail((280, 280))
            photo = ImageTk.PhotoImage(display_img)
            self.image_label.config(image=photo, text="")
            self.image_label.image = photo

            label, conf = predict(file_path)
            self.result_label.config(text=f"logo : {label}")
            self.conf_label.config(text=f"confidence: {conf*100:.1f}%")

        except Exception as e:
            messagebox.showerror("Error", f"Could not process image:\n{e}")

if __name__ == "__main__":
    root = tk.Tk()
    app = LogoApp(root)
    root.mainloop()