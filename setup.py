from setuptools import setup, find_packages

setup(
    name="statarb-helpers",
    version="0.1",
    packages=[
        "reshape_tools",
        "sample_data"
    ],
    install_requires=["pandas", "numpy", "nptyping"],
)
