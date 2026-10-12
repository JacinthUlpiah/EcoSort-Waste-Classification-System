import customtkinter as ctk
from tkinter import filedialog
import requests
from PIL import Image

ctk.set_appearance_mode("System")
ctk.set_default_color_theme("blue")

app = ctk.CTk()
app.geometry("700x450")
app.title("EcoSort Waste Classifier")

# --- MASTER CENTER FRAME ---
# This frame acts as a wrapper that stays pinned to the dead-center of the app window
main_frame = ctk.CTkFrame(app, fg_color="transparent")
main_frame.place(relx=0.5, rely=0.5, anchor="center")

# --- LAYOUT FRAMES ---
# Left side for the button (now placed inside main_frame)
left_frame = ctk.CTkFrame(main_frame, fg_color="transparent")
left_frame.pack(side="left", padx=20, pady=20, expand=True)

# Right side for the image and prediction (now placed inside main_frame)
right_frame = ctk.CTkFrame(main_frame, fg_color="transparent")
right_frame.pack(side="right", padx=20, pady=20, expand=True)

def upload_and_predict():
    file_path = filedialog.askopenfilename(
        filetypes=[("Image Files", "*.jpg *.png *.jpeg")]
    )
    if not file_path:
        return
    
    # --- DISPLAY THE IMAGE ---
    img = Image.open(file_path)
    img.thumbnail((300, 300)) # Resize to fit the UI while maintaining aspect ratio
    ctk_image = ctk.CTkImage(light_image=img, dark_image=img, size=img.size)
    image_label.configure(image=ctk_image, text="") # Remove placeholder text
    
    result_label.configure(text="Analyzing image...", text_color="gray")
    app.update()
    
    # --- SEND TO BACKEND ---
    try:
        with open(file_path, "rb") as f:
            response = requests.post("http://localhost:8000/predict", files={"file": f})
            
        if response.status_code == 200:
            pred = response.json()["prediction"]
            color = "#2ecc71" if pred == "Organic" else "#3498db"
            result_label.configure(text=f"Result: {pred}", text_color=color)
        else:
            result_label.configure(text="API Error", text_color="red")
            
    except requests.exceptions.ConnectionError:
        result_label.configure(text="Connection Error. Is Docker running?", text_color="red")

# --- UI ELEMENTS ---
# expand=True dynamically centers the button vertically in the left_frame
btn = ctk.CTkButton(left_frame, text="Select Waste Image", command=upload_and_predict, font=("Arial", 16))
btn.pack(expand=True) 

# Placeholder box for the image
image_label = ctk.CTkLabel(right_frame, text="No image selected", width=300, height=300, fg_color="gray20", corner_radius=10)
image_label.pack(pady=10, expand=True)

# Prediction text under the image
result_label = ctk.CTkLabel(right_frame, text="Waiting for input...", font=("Arial", 22, "bold"))
result_label.pack(pady=10)

app.mainloop()