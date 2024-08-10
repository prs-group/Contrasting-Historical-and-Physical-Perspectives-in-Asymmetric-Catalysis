
# Contrasting Historical and Physical Perspectives in Asymmetric Catalysis

<!-- ![Banner](https://your-image-link-here) Optional: Add a banner image  -->

## Overview

Welcome to the repository for **Contrasting Historical and Physical Perspectives in Asymmetric Catalysis**. This project presents a detailed examination of asymmetric catalysis, contrasting traditional quantitative structure-activity relationship approaches with machine learning methods. Designed for researchers, students, and professionals in the field, this repository serves as a supplementary resource for understanding both the historical evolution and the current state-of-the-art in asymmetric catalysis.

## Table of Contents

- [Background](#background)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Key Features](#key-features)
- [Contributing](#contributing)
- [License](#license)
- [Contact](#contact)

## Background

Asymmetric catalysis is a cornerstone of modern chemistry, enabling the selective production of chiral molecules—a critical aspect in pharmaceuticals, agrochemicals, and beyond. This project delves into the historical context, tracing the development of key concepts and techniques, while also presenting contemporary methodologies grounded in physical principles.

## Project Structure

```bash
├── data_preprocessing.py   # Scripts for preparing and cleaning datasets
├── calc.py                 # Core calculations for the analysis
├── config.py               # Configuration files for project settings
├── get_metrics.py          # Scripts to compute and compare metrics
├── graph_preprocessing.py  # Scripts for preparing graph data structures
├── hpo.py                  # Hyperparameter optimization scripts
├── model.py                # Machine learning model definitions
├── n_fold.py               # N-fold cross-validation scripts
├── train_val.py            # Training and validation pipeline scripts
└── README.md               # You are here
```

## Installation

To get started, clone the repository and install the necessary dependencies:

```bash
git clone https://github.com/prs-group/Contrasting-Historical-and-Physical-Perspectives-in-Asymmetric-Catalysis.git
cd Contrasting-Historical-and-Physical-Perspectives-in-Asymmetric-Catalysis
pip install -r requirements.txt
```

Make sure to have Python 3.7+ installed. If you need to set up a virtual environment:

```bash
python3 -m venv venv
source venv/bin/activate  # On Windows use `venv\Scripts\activate`
```

## Usage

This repository is modular, allowing you to use each component independently or together as part of a larger analysis pipeline.

### Data Preprocessing
```bash
python data_preprocessing.py --input your_data.csv --output preprocessed_data.csv
```

### Model Training
```bash
python train_val.py --config config.yaml --data preprocessed_data.csv
```

### Hyperparameter Optimization
```bash
python hpo.py --config config.yaml --trials 100
```

For more detailed usage instructions, please refer to the respective script's docstrings or the [documentation](docs/documentation.md).

## Key Features

- **Comprehensive Data Processing:** Includes robust preprocessing tools tailored for asymmetric catalysis datasets.
- **Modeling and Evaluation:** Implements state-of-the-art machine learning models, with support for cross-validation and performance metrics.
- **Hyperparameter Optimization:** Easily configurable scripts for optimizing model performance using Optuna.
- **Extensive Documentation:** Detailed explanations of each step, from historical context to cutting-edge methodologies.

## Contributing

We welcome contributions from the community! If you would like to contribute, please fork the repository and submit a pull request. For major changes, please open an issue first to discuss what you would like to change.

1. Fork the repo
2. Create a new branch (`git checkout -b feature/your-feature`)
3. Commit your changes (`git commit -m 'Add some feature'`)
4. Push to the branch (`git push origin feature/your-feature`)
5. Open a pull request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contact

For any questions, suggestions, or issues, please reach out to the project maintainers:

- **Primary Maintainer:** [Marcel Ruth](https://www.linkedin.com/in/marcel-ruth-319b04218/)
- **GitHub Issues:** Feel free to open an issue on the [GitHub repository](https://github.com/prs-group/Contrasting-Historical-and-Physical-Perspectives-in-Asymmetric-Catalysis/issues).

---

Thank you for exploring the intersection of history and physical science in asymmetric catalysis with us! Happy researching! 🚀

The initial publication of this code and related findings can be accessed at this [link](http://dx.doi.org/10.22029/jlupub-17918).