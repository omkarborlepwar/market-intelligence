from setuptools import setup, find_packages

setup(
    name="market-intelligence",
    version="1.0.0",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    python_requires=">=3.10",
    install_requires=[
        "yfinance>=0.2.0",
        "pandas>=2.0.0",
        "numpy>=1.24.0",
        "scikit-learn>=1.3.0",
        "statsmodels>=0.14.0",
        "textblob>=0.17.1",
        "feedparser>=6.0.0",
        "streamlit>=1.28.0",
        "plotly>=5.18.0",
    ],
)
