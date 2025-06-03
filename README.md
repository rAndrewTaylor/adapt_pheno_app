# Design-Weighted Validation Simulation App

This Streamlit application simulates a stratified, design-weighted validation of a predictive model across a range of prevalence, sensitivity, and specificity values. It is designed for rapid scenario analysis and reporting in clinical or research settings.

## Features

- Interactive simulation of validation metrics
- Configurable prevalence, sensitivity, and specificity ranges
- Dynamic visualization of PPV and NPV across different scenarios
- Automatic generation of HTML reports and CSV exports
- Design-weighted calculations for accurate population estimates

## Installation

1. Clone this repository:
```bash
git clone [your-repository-url]
cd [repository-name]
```

2. Install the required packages:
```bash
pip install -r requirements.txt
```

## Usage

Run the Streamlit app:
```bash
streamlit run app.py
```

## Configuration

The app allows you to configure:
- Population size
- Sample size per stratum
- Prevalence range
- Model sensitivity and specificity ranges
- Rule performance parameters
- Number of simulation replicates

## Output

The app generates:
- Interactive plots of PPV and NPV vs prevalence
- Summary metrics table
- HTML report with detailed results
- CSV export of simulation results

## License

[Your chosen license]

## Author
Andrew Taylor

---
For questions or suggestions, open an issue or contact via [GitHub](https://github.com/rAndrewTaylor/adapt_pheno_app).