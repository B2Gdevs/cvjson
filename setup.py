from setuptools import setup, find_packages

setup(
    name='cvjson',
    version='0.1',
    description="Computer Vision JSON Utility Library",
    author="Benjamin Garrard",
    author_email="bag131@txstate.edu",
    packages=find_packages(),
    install_requires=["numpy",
                     "seaborn",
                    "opencv-python",
                    "tqdm",
                    "imgaug",
                    "scikit-image",
                    "pandas"
                    ]
)