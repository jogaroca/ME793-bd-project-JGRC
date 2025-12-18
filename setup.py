from setuptools import setup, find_packages
import os


def get_version() -> str:
    init_path = os.path.join(os.path.dirname(__file__), "Utility", "__init__.py")
    with open(init_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.startswith("__version__"):
                return line.split("=")[1].strip().strip('"').strip("'")
    return "0.0.0"


def read_requirements(filename: str = "requirements.txt") -> list[str]:
    req_path = os.path.join(os.path.dirname(__file__), filename)
    if not os.path.exists(req_path):
        return []

    reqs: list[str] = []
    with open(req_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            # Defensive: setuptools no acepta flags tipo `-r ...` en install_requires
            if line.startswith("-"):
                continue
            reqs.append(line)
    return reqs


setup(
    name="me793-bd-project-jgrc",
    version=get_version(),
    packages=find_packages(),  # includes Utility
    install_requires=read_requirements(),
    python_requires=">=3.9",
    description="ME793 Bd chemostat project utilities (observability + nonlinear estimation)",
    author="JGRC",
)
