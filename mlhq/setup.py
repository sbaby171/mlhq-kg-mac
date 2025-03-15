from setuptools import setup, find_packages

setup(
    name="mlhq",
    version="0.1.0",
    packages=find_packages(),
    include_package_data=True,  # Ensures non-Python files are included
    package_data={
        "mlhq": ["model-registry.json"],  # Specify the JSON file to include
    },
    install_requires=[
        "pyfiglet",
    ],  
)
