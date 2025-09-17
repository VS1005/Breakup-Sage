# Author :
## NAME : Vatsalya singh
## BRANCH : Mechanical Engineering
## UNIVERSITY : Indian Institute of Technology Guwahati ( IIT Guwahati )

# 💔 Breakup-Sage

*Intelligent healing for every heartbreak*

## Demo video can be accessed here on YouTube : https://youtu.be/yNK_ZEyuWx8

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/release/python-380/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-red.svg)](https://streamlit.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**Breakup-Sage** is an intelligent breakup recovery assistant powered by a fine-tuned LLaVA (Large Language and Vision Assistant) model. It uses hybrid AI analysis to automatically select the most appropriate recovery agents based on your situation, providing personalized support through emotional healing, closure guidance, practical planning, and honest feedback.

---

## Features

* **Hybrid AI Agent Selection**: Combines rule-based analysis with LLM intelligence for optimal support matching
* **4 Specialized Recovery Agents**:

  * **Therapist**: Emotional support and healing guidance
  * **Closure Specialist**: Understanding and acceptance assistance
  * **Routine Planner**: Practical life rebuilding strategies
  * **Honest Feedback**: Direct perspective and accountability
* 📸 **Vision Support**: Analyze chat screenshots and images for context
* ⚙️ **Customizable**: Adjustable sensitivity and analysis weights
* 🔄 **Auto Checkpoint Selection**: Automatically uses the best-performing model checkpoint
* 📊 **Transparent Analysis**: See exactly why certain agents were selected

---

## Quick Start

### ⚠ Important: Run on Google Colab

Most users don’t have a CUDA GPU locally. To make it simple, **we recommend running Breakup-Sage directly on Google Colab**. The app uses **ngrok** to create a public link so you can access the Streamlit interface in your browser.

### Steps

1. **Open Colab Notebook**

   * [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/yourusername/breakup-sage/blob/main/training_notebook.ipynb)

2. **Clone the repository**

   ```bash
   !git clone https://github.com/yourusername/breakup-sage.git
   %cd breakup-sage
   ```

3. **Install dependencies**

   ```bash
   !pip install -r requirements.txt
   ```

4. **Mount Google Drive (for checkpoints)**

   ```python
   from google.colab import drive
   drive.mount('/content/drive')
   ```

5. **Download checkpoints**

   * Place your fine-tuned model checkpoints inside:

     ```
     /content/drive/MyDrive/llava_lora_finetune/
     ```
     Drive link for downloading checkpoints folder : https://drive.google.com/drive/folders/17GCH6H-yBgYjvzu7AYh993TqUcX0F111?usp=sharing

6. **Run Streamlit with ngrok**

   ```bash
   !pip install pyngrok
   from pyngrok import ngrok
   public_url = ngrok.connect(8501)
   print("App URL:", public_url)
   !streamlit run app.py --server.port 8501
   ```

7. **Click the ngrok URL** → Access the app in your browser 🎉

---

## Usage

* **Launch on Colab** (as shown above)
* **Describe your situation** or upload chat screenshots
* **Breakup-Sage** will auto-select the best recovery agents

Example:

* *"I don’t understand why this happened so suddenly..."* → **Closure Specialist**
* *"I need to rebuild my life and create new routines..."* → **Routine Planner**
* *"Tell me honestly, what did I do wrong?"* → **Honest Feedback**

---

## Configuration

Update model paths in `app.py`:

```python
LOCAL_MODEL_PATH = "/content/drive/MyDrive/llava_lora_finetune"
BASE_MODEL_NAME = "unsloth/llava-1.5-7b-hf-bnb-4bit"
```

Checkpoint selection order:

1. Lowest validation loss
2. Training loss (secondary)
3. Latest checkpoint (fallback)

---

## Project Structure

```
breakup-sage/
├── app.py
├── finetune_llava.py
├── training_notebook.ipynb
├── requirements.txt
├── README.md
├── data/
│   ├── sample_data.json
│   └── test_prompts.md
├── models/
│   └── checkpoints/   # Download separately
└── docs/
    ├── TRAINING.md
    └── API.md
```

---

## Performance

✅ Emotional state recognition
✅ Context-appropriate agent selection
✅ Multi-modal (text + images) understanding
✅ Personalized response generation

---

## License

This project is licensed under the **MIT License** - see the [LICENSE](LICENSE) file.

