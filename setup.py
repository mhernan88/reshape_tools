from setuptools import setup

setup(
    name="reshape-tools",
    version="0.1.1",
    packages=[
        "reshape_tools",
        "reshape_logging",
        "sample_data"
    ],
    install_requires=["pandas", "numpy", "nptyping"],
)
