import json
import os

notebook = {
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# 🛡️ PhishShield - Exécution sur Google Colab\n",
    "Bienvenue dans le notebook de déploiement de **PhishShield** pour ta soutenance PFE.\n",
    "Exécute les cellules ci-dessous une par une en cliquant sur le bouton *Play*."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "# 1. Connexion au Google Drive\n",
    "from google.colab import drive\n",
    "drive.mount('/content/drive')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "# 2. Décompression du projet PhishShield\n",
    "!unzip -q \"/content/drive/MyDrive/PFE_Phishing/PhishShield_Colab.zip\" -d \"/content/PhishShield\"\n",
    "print(\"✅ Projet décompressé avec succès !\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "# 3. Installation des bibliothèques IA\n",
    "!pip install -q lightgbm xgboost shap lime fastapi uvicorn pydantic pandas scikit-learn matplotlib seaborn\n",
    "print(\"✅ Bibliothèques installées !\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "# 4. Se placer dans le bon dossier de travail\n",
    "import os\n",
    "os.chdir('/content/PhishShield/xai_project')\n",
    "!pwd"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "# 5. Lancer un test d'entraînement (Machine Learning)\n",
    "!python main.py"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "# 6. Lancer le Deep Learning (Multi-dataset)\n",
    "!python cross_dataset_dl_native.py"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "name": "python3"
  },
  "language_info": {
   "name": "python"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 0
}

path = 'PhishShield_Notebook.ipynb'
with open(path, 'w', encoding='utf-8') as f:
    json.dump(notebook, f, indent=2)

print(f"Notebook créé avec succès à: {os.path.abspath(path)}")
