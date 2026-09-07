import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision import models
from PIL import Image, ImageTk
import tkinter as tk
from tkinter import filedialog, messagebox

IMG_SIZE = 224
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Fallback class mapping (will be overwritten by checkpoint if available)
class_to_idx = {
    'Athletic Bilbao': 0, 'Atletico Madrid': 1, 'Barcelona': 2, 'Celta Vigo': 3,
    'Deportivo Alaves': 4, 'Espanyol': 5, 'Getafe': 6, 'Girona': 7, 'Las Palmas': 8,
    'Leganes': 9, 'Mallorca': 10, 'Osasuna': 11, 'Rayo Vallecano': 12, 'Real Betis': 13,
    'Real Madrid': 14, 'Real Sociedad': 15, 'Real Valladolid': 16, 'Sevilla': 17,
    'Valencia': 18, 'Villarreal': 19
}
idx_to_class = {v: k for k, v in class_to_idx.items()}
NUM_CLASSES = 20


def create_model(num_classes):
    model = models.resnet18(weights=None)  # weights=None because we load our fine-tuned weights
    in_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(0.4),
        nn.Linear(in_features, 256),
        nn.ReLU(inplace=True),
        nn.Dropout(0.3),
        nn.Linear(256, num_classes)
    )
    return model


transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# Load model
model = create_model(NUM_CLASSES).to(DEVICE)

try:
    checkpoint = torch.load("best_laliga_model.pth", map_location=DEVICE)

    # Support both old and new checkpoint formats
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
        if 'idx_to_class' in checkpoint:
            idx_to_class = checkpoint['idx_to_class']
            class_to_idx = checkpoint.get('class_to_idx', {v: k for k, v in idx_to_class.items()})
            NUM_CLASSES = checkpoint.get('num_classes', len(idx_to_class))
        print("Loaded new-format checkpoint with class mapping.")
    else:
        # Old format (just state_dict)
        model.load_state_dict(checkpoint)
        print("Loaded old-format checkpoint.")

except Exception as e:
    print(f"Warning: could not load best_laliga_model.pth → {e}")
    print("Trying laliga_logo_cnn_final.pth ...")
    checkpoint = torch.load("laliga_logo_cnn_final.pth", map_location=DEVICE)
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
        if 'idx_to_class' in checkpoint:
            idx_to_class = checkpoint['idx_to_class']
    else:
        model.load_state_dict(checkpoint)

model.eval()
print(f"Model ready on {DEVICE} | {NUM_CLASSES} classes")


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
        self.root.title("La Liga Logo Classifier (ResNet18)")
        self.root.geometry("420x520")
        self.root.resizable(False, False)

        self.title_label = tk.Label(root, text="La Liga Logo Classifier", font=("Arial", 16, "bold"))
        self.title_label.pack(pady=10)

        self.image_frame = tk.Frame(root, bg="#eeeeee", width=320, height=320)
        self.image_frame.pack(pady=10)
        self.image_frame.pack_propagate(False)

        self.image_label = tk.Label(self.image_frame, bg="#eeeeee", text="No image selected")
        self.image_label.pack(expand=True)

        self.browse_btn = tk.Button(
            root, text="Browse Image",
            command=self.browse_image,
            font=("Arial", 12), bg="#4CAF50", fg="white", padx=12, pady=6
        )
        self.browse_btn.pack(pady=10)

        self.result_label = tk.Label(root, text="logo : —", font=("Arial", 14, "bold"))
        self.result_label.pack(pady=8)

        self.conf_label = tk.Label(root, text="", font=("Arial", 11), fg="gray")
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
            display_img.thumbnail((300, 300))
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
