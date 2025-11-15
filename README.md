# README.md for Flower CNN Classification
# This file is ready to be used on GitHub without any personal information.

# Flower CNN Classification

This repository contains a simple CNN training code using a small number of flower images.
The code is provided without any personal information and is safe to share on GitHub.

---

## How to Run

### 1. Prepare the Image Data

1. Create a folder named `FlowerImages` in the same directory as this repository.
2. Organize the folder structure as follows (create a folder for each class and put images inside):

```
FlowerImages/
    daisy/
        image1.jpg
        image2.jpg
        ...
    rose/
        image1.jpg
        image2.jpg
        ...
    tulip/
        image1.jpg
        image2.jpg
        ...
```

- Supported image formats are `.jpg` or `.png`.
- Having at least 2 images per class is recommended for stable training.

---

### 2. Required Python Packages

```bash
pip install tensorflow matplotlib
```

---

### 3. Run the Training

```bash
python FlowerCNN.py
```

- This will train the CNN using the training data and display graphs of training accuracy and loss.
- Currently, there is **no validation data** included.

---

### 4. Notes

- The code does **not contain any personal information** (e.g., user names or full paths).
- The image data is **not included in this repository**, so you must prepare your own `FlowerImages` folder.
