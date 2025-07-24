# Medical Image Diagnosis using Deep Learning
This repository contains a comprehensive demonstration of developing medical image diagnosis algorithms using Convolutional Neural Networks (CNN). It is designed for educational purposes to help students understand the complete pipeline of medical AI development and includes advanced features for scientific research.

## Project Overview
This project demonstrates a complete pipeline for medical image diagnosis (both classification and regression tasks), including:

1. Image Preprocessing: Histogram equalization, Contrast enhancement, Gaussian smoothing, Image resizing, Visualization of preprocessing effects

2. Dataset Management: Custom dataset class for medical images, Train/validation/test split, Data augmentation, DataLoader implementation

3. Model Architecture: Custom CNN architecture, Training pipeline, Early stopping, Model checkpointing, Class name mapping for better visualization

4. Hyperparameter Tuning: Grid search implementation, Parameter optimization, Results visualization, Best model selection

5. Model Comparison & Benchmarking: Deep Learning (CNN) vs Traditional ML methods, SVM, Random Forest, Logistic Regression comparison, Feature extraction from pre-trained CNN, Scientific-quality performance visualization

6. Advanced Evaluation and Visualization: Training history plots, Confusion matrix with custom class names, Classification metrics (Accuracy, Precision, Recall, F1), ROC and Precision-Recall curves, Scientific paper-quality figures, Performance comparison charts

# Model Comparison Features
🤖 Supported Models
Deep Learning: Custom CNN architecture
Traditional ML:
Support Vector Machine (SVM)
Random Forest
Logistic Regression
Linear Regression (for regression tasks)
📊 Generated Outputs
Figures (saved to ./results/model_evaluation/):
aupr_comparison.png - Precision-Recall curves
auc_comparison.png - ROC curves
accuracy_comparison.png - Accuracy comparison
precision_comparison.png - Precision comparison
recall_comparison.png - Recall comparison
f1_comparison.png - F1-score comparison
model_comparison_results.json - Numerical results
🎯 Scientific Paper Integration
All figures are optimized for direct use in scientific papers:

High Resolution: 300 DPI for print quality
Professional Fonts: Times New Roman serif fonts
Compact Size: 4×3 inches for bar charts, 6×5 for curves
Clean Styling: Minimal, professional aesthetics
Proper Labeling: Bold labels and appropriate sizing
Educational Applications
This repository is particularly valuable for teaching:

Medical Image Analysis: Preprocessing and feature extraction
Deep Learning: CNN architecture and training
Model Comparison: Traditional ML vs Deep Learning
Scientific Visualization: Research-quality figure generation
Performance Evaluation: Comprehensive metrics and interpretation
Best Practices: Modern AI development workflows
Requirements
Python 3.10+
PyTorch 2.0.0+
OpenCV 4.5+
NumPy 1.21+
Pandas 1.3+
Matplotlib 3.5+
scikit-learn 1.0+
seaborn 0.11+
Jupyter Notebook
Output Quality Standards
Figure Specifications
Resolution: 300 DPI (publication ready)
Format: PNG with transparent backgrounds
Fonts: Times New Roman (scientific standard)
Colors: Professional, colorblind-friendly palette
Size: Optimized for journal submissions
Performance Metrics
Classification: Accuracy, Precision, Recall, F1, AUC, AUPR
Regression: MSE, MAE, R²
Statistical Significance: Consistent evaluation protocols
Cross-Model Comparison: Fair benchmarking methodology
Contributing
We welcome contributions that enhance the educational value or research capabilities:

Bug fixes and improvements
Additional model architectures
New visualization features
Documentation enhancements
License
This project is licensed under the MIT License - see the LICENSE file for details.

Citation
If you use this code in your research, please cite:

@software{AI_medicine,
  title={When AI meets medicine – algorithms are transforming healthcare issues},
  author={Hauser Zhang},
  year={2025},
  url={https://github.com/zhanghaoyu9931/AI_medicine}
}
Acknowledgments
This project is designed for educational and research purposes, demonstrating state-of-the-art practices in medical AI development. It incorporates best practices from both academic research and industry applications.
