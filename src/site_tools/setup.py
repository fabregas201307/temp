from setuptools import find_packages, setup

setup(
    name="site-tools",
    packages=find_packages(exclude=["*tests*"]),
    version="0.0.1",
    description="Tools to be used in the site project.",
    author="Crayon",
    license="",
    python_requires=">=3.6",
    install_requires=[
        "pyexiftool==0.1.1",
        "Pillow",
        "requests",
        "numpy",
        "scipy",
        "pandas",
        "boto3>=1.14",
        "opencv-python",
        "scikit-learn",
    ],
    dependency_links=[
        "git+https://github.com/stevenwalton/pydensecrf.git",
    ]
)
