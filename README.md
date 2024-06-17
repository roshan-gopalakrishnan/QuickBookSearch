# QuickBookSearch
Quick Book Search (QBS) is a chatbot built for querying PDF textbooks for students. It is powered by NVIDIA-NIM and LangChain.

![streamlit_app](https://github.com/roshan-gopalakrishnan/QuickBookSearch/streamlit_app.png)


Follow the steps below to use this repo.

# Requirements
Python3 >= 3.8

Create a virtual environment:
python3 -m venv QuickBookSearch
source QuickBookSearch/bin/activate

# Installation

pip install -r requirements.txt

# Run

streamlit run main.py

You need to upload a PDF textbook for the first time usage. A sample PDF file is shared in the repo (downloaded from https://electrovolt.ir/wp-content/uploads/2014/08/Design-of-Analog-CMOS-Integrated-Circuit-2nd-Edition-ElectroVolt.ir_.pdf).

For the first time PDF upload, you need to wait for few minutes to index the file and save.

Once the upload is ready then there will be a chat interface to ask questions. 
