# **Combinational Depth Prediction Using AI**

## **Girl Hackathon 2025 - Silicon Track**

### **Table of Contents**
1. [Introduction](#introduction)
2. [Problem Statement](#problem-statement)
3. [Solution Approach](#solution-approach)
4. [Installation & Setup](#installation--setup)
5. [Dataset Preparation](#dataset-preparation)
6. [Model Architecture](#model-architecture)
7. [Training & Evaluation](#training--evaluation)
8. [Results & Observations](#results--observations)
9. [How to Use](#how-to-use)
10. [Example Usage](#example-usage)
11. [Future Work](#future-work)
12. [References](#references)

---

## **1. Introduction**
Timing analysis is a critical step in chip design, but it is often **slow** due to reliance on synthesis tools. This project aims to **predict combinational depth** of signals using AI/ML, reducing dependency on full synthesis and accelerating the timing verification process.

---

## **2. Problem Statement**
Traditional synthesis tools generate **timing reports** only after the entire design has been processed. This is **time-consuming**, leading to **delays in project execution**. The objective is to develop an AI-based approach that can predict the **combinational depth** of a signal in RTL **before synthesis**.

### **Definitions:**
- **Combinational Depth**: The longest path of logic gates (AND, OR, NAND, etc.) required to generate a signal.
- **Timing Violation**: When the combinational depth exceeds what the clock period allows.

### **Inputs & Outputs:**
- **Input**: RTL code & target signal
- **Output**: Predicted combinational depth

---

## **3. Solution Approach**
### **Steps Involved:**
1. **Dataset Creation**: Extract combinational depth from RTL synthesis reports.
2. **Feature Engineering**: Identify key RTL parameters influencing depth (Fan-In, Fan-Out, gate types).
3. **Model Selection**: Fine-tune **CodeBERT** (`microsoft/codebert-base`) for regression.
4. **Training**: Use supervised learning to predict combinational depth.
5. **Evaluation**: Compare AI-predicted depth with actual synthesis reports.

---

## **4. Installation & Setup**
### **Prerequisites**
- Python 3.8+
- PyTorch
- Hugging Face Transformers
- Scikit-learn
- Pandas & NumPy

### **Installation**
```bash
pip install torch transformers datasets scikit-learn numpy pandas
```

---

## **5. Dataset Preparation**
- Used **MetRex dataset** containing RTL implementations.
- Extracted **critical path depth** using regex from timing reports.
- Filtered & formatted data for ML training.

---

## **6. Model Architecture**
- **Pretrained Model**: `microsoft/codebert-base` (originally for code understanding).
- **Regression Head**: Modified for predicting combinational depth.
- **Features Used**: Raw RTL text & extracted features.

---

## **7. Training & Evaluation**
### **Training Configuration:**
- **Loss Function**: Mean Squared Error (MSE)
- **Optimizer**: AdamW
- **Learning Rate**: `1e-4`
- **Batch Size**: `8`
- **Epochs**: `3`

### **Metrics Used:**
- **MSE**: Measures prediction error.
- **Test Accuracy**: *Currently 16.54%* (Needs improvement with better features).

---

## **8. Results & Observations**
- **Current Accuracy: 16.54%** (low due to lack of graph-based features).
- **Freezing CodeBERT backbone improved training speed.**
- **Future Improvements**:
  - Use **Fan-In/Fan-Out as additional features**.
  - Increase **dataset size**.

---

## **9. How to Use**
### **Training**
```bash
python train.py
```

### **Inference**
```bash
python predict.py --input "path/to/rtl/file"
```

### **Generate Predictions**
```bash
python evaluate.py
```

---

## **10. Example Usage**
### **Example Input RTL Code:**
```verilog
module example(input a, b, c, output y);
  wire w1, w2;
  assign w1 = a & b;
  assign w2 = w1 | c;
  assign y = ~w2;
endmodule
```

### **Predicted Output:**
```bash
Predicted combinational depth: 3
```

---

## **11. Future Work**
- Explore **Graph Neural Networks (GNNs)** for RTL representation.
- Incorporate **timing reports from synthesis tools** for better predictions.
- Reduce **prediction run-time** while improving accuracy.

---

## **12. References**
- [Hugging Face Transformers](https://huggingface.co/transformers/)
- [CodeBERT: A Pretrained Model for Programming Languages](https://arxiv.org/abs/2002.08155)
- [MetRex Dataset](https://huggingface.co/datasets/scale-lab/MetRex)
