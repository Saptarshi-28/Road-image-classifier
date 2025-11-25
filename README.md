# 🚗 Road Condition Classification System

A deep learning-based image classification system that automatically identifies and categorizes road conditions from images to enhance road safety and maintenance.

![Road Classification](https://img.shields.io/badge/Road-Classification-brightgreen)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.13.0-orange)
![Python](https://img.shields.io/badge/Python-3.8+-blue)
![Hugging Face](https://img.shields.io/badge/Hugging%20Face-Spaces-yellow)

## 🌐 Live Demo

**🚀 Try it now**: [Road Classification API on Hugging Face](https://huggingface.co/spaces/saptarshi-28/road-classification-api)

## 📋 Project Overview

This project leverages computer vision and deep learning to classify road images into five distinct safety categories. The system helps in automated road monitoring, safety assessment, and maintenance prioritization.

## 🏷️ Dataset & Classes

### Dataset Composition
- **Total Images**: Handpicked from various internet sources
- **Labeling**: Manually annotated by human reviewers
- **Quality**: Curated for diverse road scenarios and lighting conditions

### Classification Categories

| Category | Label | Description | Priority | Color Code |
|----------|-------|-------------|----------|------------|
| 🟢 Safe Area | `safe` | Well-maintained, clear roads with no visible issues | Low | Green |
| 🟡 Damaged Road | `caution` | Roads with potholes, cracks, or surface damage | Medium | Yellow |
| 🟡 Road Blockage | `caution` | Obstructions, construction, or blocked pathways | Medium | Yellow |
| 🟡 No Surveillance | `caution` | Areas lacking proper monitoring or lighting | Medium | Yellow |
| 🔴 Accident | `danger` | Traffic accidents, collisions, or emergency situations | High | Red |

## 🧠 Model Architecture

### Base Model
- **Architecture**: MobileNetV2
- **Input Size**: 224×224×3
- **Weights**: ImageNet pre-trained

### Fine-Tuning Details
```python
# Model Configuration
base_model = MobileNetV2(
    weights='imagenet',
    include_top=False,
    input_shape=(224, 224, 3)
)
base_model.trainable = False

# Custom Head
model = Sequential([
    base_model,
    GlobalAveragePooling2D(),
    Dropout(0.3),
    Dense(128, activation='relu'),
    Dropout(0.3),
    Dense(5, activation='softmax')  # 5 classes
])
