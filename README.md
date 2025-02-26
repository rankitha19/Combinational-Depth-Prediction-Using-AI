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
11. [References](#references)

---

## **1. Introduction**
Timing analysis is a critical step in chip design, but it is often **slow** due to reliance on synthesis tools. This project aims to **predict combinational depth** of signals using AI/ML, reducing dependency on full synthesis and accelerating the timing verification process.

---

## **2. Problem Statement**
Traditional synthesis tools generate **timing reports** only after the entire design has been processed. This is **time-consuming**, leading to **delays in project execution**. The objective is to develop an AI-based approach that can predict the **combinational depth** of a signal in RTL **before synthesis**.

### **Definitions:**
- **Combinational Depth**: The longest path of combinational logic gates (AND, OR, NAND, etc.) required to generate a signal.
- **Timing Violation**: When the combinational depth exceeds what the clock period allows.

### **Inputs & Outputs:**
- **Input**: RTL code & target signal
- **Output**: Predicted combinational depth

---

## **3. Solution Approach**
### **Steps Involved:**
*Dataset Selection*
   - Used an open-source dataset (MetRex) containing RTL designs and their respective timing reports.
   - The dataset already includes critical path information, allowing us to extract combinational depth directly.

*Feature Extraction*
   - Applied regex-based parsing to extract the number of gates in the critical path from the delay column.
   - The extracted count includes all elements in the critical path. We assume that all elements are combinational, as the number of sequential elements is negligible. Future improvements can refine this by distinguishing between combinational and sequential elements.
   - Filtered out samples where depth extraction was not possible.
   
*Data Formatting*
   - Reformatted the dataset to a supervised learning format.
   - Created input-output pairs where the RTL code serves as input, and the extracted combinational depth is the target label.

*Model Selection*
   - Chose CodeBERT (microsoft/codebert-base) as the pretrained model.
   - Pretrained models like CodeBERT have been trained on large-scale GitHub repositories, making them well-suited for understanding RTL code.
   - Fine-tuning an existing model is computationally efficient compared to training from scratch.
   
*Data Preprocessing*
   - Tokenized RTL code using CodeBERT’s tokenizer.
   - Converted target depth values to float for regression.

*Model Training*
   - Used supervised learning with a regression-based approach.
   - Fine-tuned only the classification head while freezing other layers to reduce training time.
   - Used MSE loss as the evaluation metric.
  
     
*Evaluation*
   - Split dataset into training (90%) and testing (10%).
   - Computed accuracy by comparing predicted depth with true depth.
   - Test accuracy achieved: 36.54% (indicating room for improvement with better feature engineering).

*Prediction and Output Generation*
   - Generated predictions on the test set.
   - Saved outputs as CSV files containing RTL, true labels, and predicted values.

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
- Extracted **combinational depth** and other parameters using regex from timing reports.
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
- **Test Accuracy**: *Currently 36.54%* (Needs improvement with better features).

---

  ## *8. Results & Observations*
- *Current Accuracy: 36.54%* 
- *Freezing CodeBERT backbone improved training speed.*
- *Future Improvements*:
  - Use *Fan-In/Fan-Out as additional features*.
  - Increase *dataset size*.
  - Accuracy can be further improved by training on a larger dataset.
  - We have used only *3 epochs* for training, which can be increased.
  - A *larger model like CodeLlama-7B* can yield better accuracy.
  - Full fine-tuning (instead of parameter-efficient tuning) can further enhance accuracy.
  
### *Why We Didn't Use Full Fine-Tuning?*
- *RAM and GPU limitations* restricted the ability to fine-tune all layers.
- Training other layers would have taken significantly more time due to *computational constraints*.
- Instead, we opted for *parameter-efficient fine-tuning*, training only the last layer to optimize available resources.

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



---

## **11. References**
- [Hugging Face Transformers](https://huggingface.co/transformers/)
- [CodeBERT: A Pretrained Model for Programming Languages](https://arxiv.org/abs/2002.08155)
- [MetRex Dataset](https://huggingface.co/datasets/scale-lab/MetRex)
#The cleaned train data is available for download from [Google Drive](https://drive.google.com/file/d/17GrhDF3yk_567xrWQgdTOFDLVvOL0pgU/view?usp=sharing).  
(Note: The file is too large to be added to Git.)
  This project provides a foundation for AI-driven timing analysis in RTL design. Future work can improve accuracy by incorporating additional RTL features like fan-in, fan-out, and synthesis optimizations
