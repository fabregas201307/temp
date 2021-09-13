from setuptools import find_packages, setup

setup(
    name="site-cli",
    packages=find_packages(exclude=["*package_data*"]),
    version="0.0.1",
    description="CLI for the SITE AI automation project.",
    author="Crayon",
    license="",
    python_requires=">=3.7",
    install_requires=["click>=7.1", "sagemaker>=2.5"],
    package_data={
        "site_cli.inference.package_data.scripts": ["preprocess.py", "postprocess.py"],
        "site_cli.inference.package_data.config": [
            "global.json",
            "p1_preprocess.json",
            "p2_preprocess.json",
            "p1_transform.json",
            "p2_transform.json",
            "p1_postprocess.json",
            "p2_postprocess.json",
        ],
    },
    entry_points="""
        [console_scripts]
        stcrayon=site_cli.cli:entry_point
    """,
)
