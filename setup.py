from setuptools import find_packages, setup

setup(
    name = 'Medical Chatbot',
    version= '0.0.0',
    author= 'Siddhant Dnyane',
    author_email= 'siddhantdnyane2003@gmail.com',
    install_requires = ["langchain", "langchain-chroma", "langchain-community", "langchain-ollama", "ollama", "flask", "pypdf", "python-dotenv"],
    packages= find_packages()
)