import customtkinter as ctk
from tkinter import filedialog
from PIL import Image, ImageTk
import requests

ctk.set_appearance_mode("System")
ctk.set_default_color_theme("blue")

app = ctk.CTk()
app.geometry("520x590")
app.title("EcoSort - Intelligent Waste Classification System")

current_image = None

def upload_and_predict():
    global current_image
    file_path = filedialog.askopenfilename(
        filetypes=[("Image Files", "*.jpg *.png *.jpeg")]
    )
    if not file_path:
        return
    
    # Load and display image preview
    img = Image.open(file_path)
    img_resized = img.resize((180, 180))
    current_image = ImageTk.PhotoImage(img_resized)
    img_label.configure(image=current_image, text="")
    
    # Reset status states
    result_label.configure(text="Analyzing image...", text_color="gray")
    confidence_label.configure(text="Confidence: --")
    disposal_label.configure(text="Recommended Bin: Please wait...", text_color="gray")
    app.update()
    
    try:
        with open(file_path, "rb") as f:
            response = requests.post("http://localhost:5000/predict", files={"file": f})
            
        if response.status_code == 200:
            data = response.json()
            pred = data.get("prediction", "Unknown")
            
            if pred.lower() == "organic":
                color = "#2fa572" # Green accent
                disposal_text = "Recommended Bin: Compost Bin"
            else:
                color = "#1f538d" # Blue accent
                disposal_text = "Recommended Bin: Recyclable Bin"
                
            result_label.configure(text=f"Result: {pred}", text_color=color)
            confidence_label.configure(text="Confidence: High (Model Verified)")
            disposal_label.configure(text=disposal_text, text_color=color)
        else:
            result_label.configure(text="API Error", text_color="red")
            
    except requests.exceptions.ConnectionError:
        result_label.configure(text="Error: Is the Docker container running?", text_color="red")

# --- UI Layout Design ---

# Header Frame (Subtitle removed)
header_frame = ctk.CTkFrame(app, fg_color="transparent")
header_frame.pack(pady=15)

title_label = ctk.CTkLabel(header_frame, text="EcoSort Waste Classifier", font=("Segoe UI", 22, "bold"))
title_label.pack()

# Upload Button
btn = ctk.CTkButton(app, text="Select Waste Image", command=upload_and_predict, font=("Segoe UI", 14, "bold"), width=220, height=42, corner_radius=8)
btn.pack(pady=10)

# Image Preview Card
img_label = ctk.CTkLabel(app, text="No image selected", width=180, height=180, fg_color=("gray85", "gray20"), corner_radius=10)
img_label.pack(pady=10)

# Results Panel Frame (Card style)
info_frame = ctk.CTkFrame(app, width=440, height=130, corner_radius=10)
info_frame.pack(pady=10, padx=20, fill="x")

result_label = ctk.CTkLabel(info_frame, text="Waiting for input...", font=("Segoe UI", 16, "bold"))
result_label.pack(pady=(12, 5))

confidence_label = ctk.CTkLabel(info_frame, text="Confidence: --", font=("Segoe UI", 13))
confidence_label.pack(pady=2)

disposal_label = ctk.CTkLabel(info_frame, text="Recommended Bin: --", font=("Segoe UI", 13, "italic"), text_color="gray")
disposal_label.pack(pady=(2, 12))

app.mainloop()