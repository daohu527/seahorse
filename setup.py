import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="seahorse",
    version="1.0.0",
    author="daohu527",
    author_email="daohu527@gmail.com",
    description="A pure python autonomous driving framework",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/daohu527/seahorse",
    project_urls={
        "Bug Tracker": "https://github.com/daohu527/seahorse/issues",
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
    ],
    package_dir={"": "."},
    packages=setuptools.find_packages(where="."),
    package_data={"": [
        'perception/config.yaml',
    ]},
    install_requires=[
        "torch",
        "pyyaml",
    ],
    entry_points={
        'console_scripts': [
            'seahorse = seahorse.command:main',
        ],
    },
    python_requires=">=3.6",
)