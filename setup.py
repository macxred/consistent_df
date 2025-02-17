from setuptools import find_packages, setup

setup(
    name="consistent_df",
    version="v0.0.1",
    description=("Python package designed to provide shared DataFrame operations"),
    url='https://github.com/macxred/consistent_df',
    author="Lukas Elmiger, Oleksandr Stepanenko",
    python_requires='>3.9',
    install_requires=['pandas'],
    packages=find_packages(exclude=('tests', 'examples')),
    extras_require={
        "dev": [
            "flake8",
            "bandit",
            "pytest"
        ]
    }
)
