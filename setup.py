from setuptools import setup, find_packages

setup(
    name="hierarchical-streaming",
    version="1.0.0",
    author="Nikita Zmanovskii",
    author_email="zmanovskiy.n.v@gmail.com",
    description="Optimal hierarchical aggregation for multi-scale time series",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/username/hierarchical-streaming",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
    install_requires=[
        "numpy>=1.21.0",
        "pandas>=1.3.0",
        "scipy>=1.7.0",
        "matplotlib>=3.4.0",
        "pyyaml>=6.0",
        "tqdm>=4.62.0",
    ],
    entry_points={
        "console_scripts": [
            "hstream=scripts.run_experiments:main",
        ],
    },
)