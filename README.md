# Quantum Generative Adversarial Networks (qGANs) for Advanced AI

## 🌟 **Quantum Generative Adversarial Networks: Integrating Quantum Computing with Artificial Intelligence** 🌟

Quantum computing is revolutionizing the landscape of artificial intelligence by offering unparalleled processing power and efficiency. This project harnesses **Quantum Generative Adversarial Networks (qGANs)** to push the boundaries of AI capabilities. By integrating quantum algorithms with traditional AI frameworks, qGANs enable faster data generation, more complex pattern recognition, and enhanced model training.

This innovative approach allows us to tackle intricate problems in various fields such as healthcare, finance, and creative industries—areas where classical AI struggles with computational limitations. For example, qGANs can accelerate drug discovery by simulating molecular interactions with greater accuracy or enhance financial models by analyzing vast datasets in real-time.

By leveraging the strengths of quantum computing, this project not only advances the state-of-the-art in AI but also paves the way for future technological breakthroughs. Join me in exploring how qGANs are transforming AI, making it more powerful, efficient, and capable of solving the complex challenges of tomorrow.

## 📜 **Table of Contents**

- [Introduction](#introduction)
- [Features](#features)
- [Architecture](#architecture)
- [Installation](#installation)
- [Usage](#usage)
- [Dataset](#dataset)
- [Training](#training)
- [Visualization](#visualization)
- [Contributing](#contributing)
- [License](#license)
- [Contact](#contact)
- [References](#references)

## 🎯 **Introduction**

Generative Adversarial Networks (GANs) have transformed fields such as image synthesis, data augmentation, and creative content generation. By infusing these networks with the principles of quantum computing, qGANs unlock capabilities that transcend the limitations of classical systems. This project demonstrates how quantum-enhanced AI can tackle intricate problems across various industries, including healthcare, finance, and creative arts.

## 🚀 **Features**

- **Quantum Generator:** Utilizes Qiskit and PyTorch to create a quantum-based generator capable of producing high-quality data samples.
- **Classical Discriminator:** A traditional neural network that distinguishes between real and generated data.
- **Hybrid Training Loop:** Alternates between training the generator and discriminator to improve model performance.
- **Real-World Applications:** Demonstrates the potential of qGANs in areas like drug discovery, financial modeling, and cybersecurity.
- **Visualization:** Generates and visualizes data samples to monitor training progress.

## 🛠️ **Architecture**

1. **Quantum Circuit Creation:** Builds parametrized quantum circuits using Qiskit, incorporating both data and ancilla qubits.
2. **Quantum Generator:** Implements the generator using `SamplerQNN` from Qiskit Machine Learning, connected via `TorchConnector` to integrate with PyTorch.
3. **Classical Discriminator:** Utilizes a neural network built with PyTorch to evaluate the authenticity of generated data.
4. **Training Loop:** Employs a hybrid quantum-classical training loop to iteratively improve both the generator and discriminator.

## 🛠 **Installation**

### **Prerequisites**

- Python 3.8+
- [PyTorch](https://pytorch.org/)
- [Qiskit](https://qiskit.org/)
- [Qiskit Machine Learning](https://qiskit.org/documentation/machine-learning/)
- [Matplotlib](https://matplotlib.org/)
- [Torchvision](https://pytorch.org/vision/stable/index.html)
- [Pandas](https://pandas.pydata.org/)
- [Numpy](https://numpy.org/)

### **Clone the Repository**

```bash
git clone https://github.com/tu_usuario/qGANs-Quantum-AI.git
cd qGANs-Quantum-AI
```

### **Install Dependencies**

It's recommended to use a virtual environment:

```bash
python -m venv venv
source venv/bin/activate  # On Windows use `venv\Scripts\activate`
pip install -r requirements.txt
```

*If `requirements.txt` is not provided, install manually:*

```bash
pip install torch qiskit qiskit-ibm-provider qiskit-machine-learning matplotlib torchvision pandas numpy
```

## 💻 **Usage**

### **Preparing the Data**

Ensure you have the `optdigits.tra` dataset in your project directory.

### **Running the Training**

```bash
python train_qGAN.py
```

### **Monitoring the Training**

The training script will output loss values every 10 iterations and visualize generated images every 100 iterations.

## 📃 **Dataset**

This project uses the [Optical Recognition of Handwritten Digits Dataset](https://archive.ics.uci.edu/ml/datasets/Optical+Recognition+of+Handwritten+Digits). Ensure the dataset file `optdigits.tra` is placed in the project directory.

## 🏋️‍♂️ **Training**

The training loop alternates between training the classical discriminator and the quantum generator. It employs a fixed noise vector for consistent visualization of generated samples throughout the training process.

## 📈 **Visualization**

Generated images are visualized periodically to monitor the progress and quality of the qGAN. These visualizations help in assessing the convergence and performance of the generative model.

## 🤝 **Contributing**

Contributions are welcome! Please fork the repository and submit a pull request for any enhancements or bug fixes.

## 📜 **License**

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## 📨 **Contact**

For any questions or suggestions, please contact [wil-19-60@live.com](wil-19-60@live.com).

## 📑 **References**

1. **Goodfellow et al.**
   - *Generative Adversarial Networks.*
   - arXiv:1406.2661 (2014).
   - [Link](https://arxiv.org/abs/1406.2661)

2. **Huang et al.**
   - *Experimental Quantum Generative Adversarial Networks for Image Generation.*
   - arXiv:2010.06201 (2020).
   - [Link](https://arxiv.org/abs/2010.06201)

3. **Pennylane Quantum GAN Tutorial**
   - [Link](https://pennylane.ai/qml/demos/tutorial_quantum_gans)
