# 🔺 Triangle ML

A modern machine learning experimentation platform with an intuitive web interface for hyperparameter tuning, dataset exploration, and model evaluation.

![Triangle ML](https://img.shields.io/badge/Machine%20Learning-Streamlit-blue)
![Python](https://img.shields.io/badge/Python-3.8%2B-green)
![License](https://img.shields.io/badge/License-MIT-yellow)

## ✨ Features

- **🎯 Interactive Hyperparameter Tuning**: Adjust model parameters with real-time sliders and dropdowns
- **📊 Multiple Datasets**: Built-in support for popular datasets (Iris, Breast Cancer) plus synthetic data generation
- **🤖 Multiple Algorithms**: Support for Random Forest, SVM, Logistic Regression, XGBoost, and LightGBM
- **📈 Real-time Visualizations**: Interactive charts for model performance and feature importance
- **🎨 Modern UI**: Clean, responsive interface built with Streamlit
- **⚡ Fast Prototyping**: Quick experimentation without complex setup

## 🚀 Quick Start

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/your-username/triangle-ml.git
   cd triangle-ml
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the application**
   ```bash
   streamlit run main.py
   ```

4. **Open your browser** and navigate to `http://localhost:8501`

## 📖 Usage

### Dataset Selection
Choose from built-in datasets or upload your own CSV files:
- **Iris**: Classic multi-class classification dataset
- **Breast Cancer**: Binary classification with medical features
- **Synthetic**: Generated datasets for testing and experimentation

### Algorithm Configuration
Select from popular machine learning algorithms:
- **Random Forest**: Ensemble method with customizable trees and depth
- **SVM**: Support Vector Machines with kernel selection
- **Logistic Regression**: Linear model with regularization options
- **XGBoost**: Gradient boosting with optimized performance
- **LightGBM**: Microsoft's fast gradient boosting

### Hyperparameter Tuning
Adjust parameters using intuitive sliders and dropdowns:
- Number of estimators/trees
- Learning rates
- Regularization parameters
- Kernel types
- Maximum depths

### Model Evaluation
View comprehensive results including:
- Accuracy metrics
- Classification reports
- Feature importance plots
- Target distribution charts

## 🏗️ Project Structure

```
triangle-ml/
│
├── main.py                 # Main Streamlit application
├── requirements.txt        # Python dependencies
├── README.md              # Project documentation
├── .gitignore             # Git ignore rules
│
├── data/                  # Dataset storage (future)
├── models/                # Saved models (future)
├── utils/                 # Utility functions (future)
└── notebooks/             # Jupyter notebooks (future)
```

## 🔧 Development

### Adding New Algorithms

1. Add algorithm to the selectbox in `main.py`
2. Implement hyperparameter controls
3. Add model instantiation and training logic
4. Update visualization components if needed

### Adding New Datasets

1. Add dataset loading logic to `load_sample_datasets()`
2. Update the dataset selection dropdown
3. Ensure proper feature/target extraction

### Custom Styling

Modify the CSS in the `st.markdown()` section to customize the appearance.

## 📊 Supported Algorithms & Parameters

| Algorithm | Parameters |
|-----------|------------|
| Random Forest | n_estimators, max_depth, min_samples_split |
| SVM | C, kernel, gamma |
| Logistic Regression | C, penalty, solver |
| XGBoost | n_estimators, learning_rate, max_depth |
| LightGBM | n_estimators, learning_rate, max_depth |

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Built with [Streamlit](https://streamlit.io/)
- Machine learning powered by [scikit-learn](https://scikit-learn.org/)
- Interactive charts by [Plotly](https://plotly.com/)

## 📞 Support

If you find this project helpful, please give it a ⭐️ on GitHub!

For questions or issues, please open an [issue](https://github.com/your-username/triangle-ml/issues) on GitHub.

---

**Happy Machine Learning! 🚀**