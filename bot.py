"""
Chatbot de Farmácia com Machine Learning - Versão Telegram
Baseado na Aula 02 - PLN + NLTK
Curso Técnico em Inteligência Artificial - CETAM / IBC
"""

import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
import random
import os
from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes

# ============================================================
# 1. Token do Bot (pegue no @BotFather)
# ============================================================

TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()

# ============================================================
# 2. Download dos recursos NLTK
# ============================================================
nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)
nltk.download('stopwords', quiet=True)

stop_pt = set(stopwords.words('portuguese'))

# ============================================================
# 3. Base de conhecimento da farmácia
# ============================================================


# ============================================================
# 300 MEDICAMENTOS ADICIONAIS
# Preços fictícios para uso em protótipo
# ============================================================

medicamentos = {
     "dipirona": {
        "preco": "R$ 8,90 (500mg - 20 comprimidos)",
        "indicacao": "Dor e febre",
        "estoque": True
    },

    "paracetamol": {
        "preco": "R$ 7,50 (750mg - 20 comprimidos)",
        "indicacao": "Dor e febre",
        "estoque": True
    },

    "ibuprofeno": {
        "preco": "R$ 12,90 (400mg - 20 comprimidos)",
        "indicacao": "Dor, inflamação e febre",
        "estoque": True
    },

    "omeprazol": {
        "preco": "R$ 15,90 (20mg - 28 cápsulas)",
        "indicacao": "Azia, refluxo e proteção gástrica",
        "estoque": True
    },

    "losartana": {
        "preco": "R$ 18,50 (50mg - 30 comprimidos)",
        "indicacao": "Pressão alta",
        "estoque": True
    },

    "vitamina c": {
        "preco": "R$ 22,00 (1g - 30 comprimidos efervescentes)",
        "indicacao": "Suplementação de vitamina C",
        "estoque": True
    },

    "rivotril": {
        "preco": "Medicamento controlado - necessário receita",
        "indicacao": "Transtornos de ansiedade e outras indicações específicas",
        "estoque": True
    },

    "amoxicilina": {
        "preco": "R$ 28,90 (500mg - 21 cápsulas) - necessário receita",
        "indicacao": "Infecções bacterianas",
        "estoque": True
    },

    "loratadina": {
        "preco": "R$ 11,90 (10mg - 12 comprimidos)",
        "indicacao": "Sintomas de alergia",
        "estoque": True
    },

    "buscopan": {
        "preco": "R$ 14,50 (10mg - 20 comprimidos)",
        "indicacao": "Cólicas e espasmos gastrointestinais",
        "estoque": True
    },

    "azitromicina": {
        "preco": "R$ 25,90 (500mg - 5 comprimidos) - necessário receita",
        "indicacao": "Infecções bacterianas",
        "estoque": True
    },

    "cefalexina": {
        "preco": "R$ 24,90 (500mg - 28 cápsulas) - necessário receita",
        "indicacao": "Infecções bacterianas",
        "estoque": True
    },

    "nimesulida": {
        "preco": "R$ 10,90 (100mg - 12 comprimidos)",
        "indicacao": "Dor e inflamação",
        "estoque": True
    },

    "diclofenaco": {
        "preco": "R$ 9,90 (50mg - 20 comprimidos)",
        "indicacao": "Dor e inflamação",
        "estoque": True
    },

    "cetirizina": {
        "preco": "R$ 13,90 (10mg - 12 comprimidos)",
        "indicacao": "Sintomas de alergia",
        "estoque": True
    },

    "prednisona": {
        "preco": "R$ 15,90 (20mg - 20 comprimidos) - necessário receita",
        "indicacao": "Processos inflamatórios e alérgicos",
        "estoque": True
    },

    "prednisolona": {
        "preco": "R$ 18,90 (20mg - 20 comprimidos) - necessário receita",
        "indicacao": "Processos inflamatórios e alérgicos",
        "estoque": True
    },

    "metformina": {
        "preco": "R$ 12,90 (850mg - 30 comprimidos)",
        "indicacao": "Controle do diabetes tipo 2",
        "estoque": True
    },

    "atenolol": {
        "preco": "R$ 9,90 (50mg - 30 comprimidos)",
        "indicacao": "Pressão alta e algumas condições cardíacas",
        "estoque": True
    },

    "enalapril": {
        "preco": "R$ 10,90 (10mg - 30 comprimidos)",
        "indicacao": "Pressão alta e insuficiência cardíaca",
        "estoque": True
    },

    "hidroclorotiazida": {
        "preco": "R$ 8,90 (25mg - 30 comprimidos)",
        "indicacao": "Pressão alta e retenção de líquidos",
        "estoque": True
    },

    "sinvastatina": {
        "preco": "R$ 12,90 (20mg - 30 comprimidos)",
        "indicacao": "Controle do colesterol",
        "estoque": True
    },

    "pantoprazol": {
        "preco": "R$ 18,90 (40mg - 28 comprimidos)",
        "indicacao": "Azia, refluxo e problemas relacionados à acidez gástrica",
        "estoque": True
    },

    "domperidona": {
        "preco": "R$ 17,90 (10mg - 30 comprimidos)",
        "indicacao": "Náuseas e distúrbios da motilidade gastrointestinal",
        "estoque": True
    },

    "metoclopramida": {
        "preco": "R$ 8,90 (10mg - 20 comprimidos)",
        "indicacao": "Náuseas e vômitos",
        "estoque": True
    },

    "simeticona": {
        "preco": "R$ 10,90 (40mg - 20 comprimidos)",
        "indicacao": "Gases e desconforto abdominal",
        "estoque": True
    },

    "loperamida": {
        "preco": "R$ 9,90 (2mg - 12 comprimidos)",
        "indicacao": "Controle sintomático da diarreia",
        "estoque": True
    },

    "sulfato ferroso": {
        "preco": "R$ 15,90 (40mg - 30 comprimidos)",
        "indicacao": "Suplementação de ferro",
        "estoque": True
    },

    "vitamina d": {
        "preco": "R$ 19,90 (7.000 UI - 8 cápsulas)",
        "indicacao": "Suplementação de vitamina D",
        "estoque": True
    },

    "ácido fólico": {
        "preco": "R$ 8,90 (5mg - 30 comprimidos)",
        "indicacao": "Suplementação de ácido fólico",
        "estoque": True
    },

    "albendazol": {
        "preco": "R$ 9,90 (400mg - 1 comprimido)",
        "indicacao": "Tratamento de algumas verminoses",
        "estoque": True
    },

    "ivermectina": {
        "preco": "R$ 18,90 (6mg - 4 comprimidos)",
        "indicacao": "Tratamento de parasitoses específicas",
        "estoque": True
    },

    "fluconazol": {
        "preco": "R$ 12,90 (150mg - 1 cápsula)",
        "indicacao": "Tratamento de algumas infecções fúngicas",
        "estoque": True
    },

    "nistatina": {
        "preco": "R$ 14,90 (creme 60g)",
        "indicacao": "Infecções por fungos, conforme apresentação",
        "estoque": True
    },

    "aciclovir": {
        "preco": "R$ 19,90 (200mg - 25 comprimidos)",
        "indicacao": "Tratamento de algumas infecções por herpes",
        "estoque": True
    },

    "salbutamol": {
        "preco": "R$ 18,90 (100mcg - aerossol)",
        "indicacao": "Alívio do broncoespasmo em condições respiratórias",
        "estoque": True
    },

    "budesonida": {
        "preco": "R$ 29,90 (apresentação inalatória)",
        "indicacao": "Controle de doenças inflamatórias das vias respiratórias",
        "estoque": True
    },

    "ambroxol": {
        "preco": "R$ 16,90 (xarope 30mg/5ml)",
        "indicacao": "Auxílio na fluidificação das secreções respiratórias",
        "estoque": True
    },

    "acetilcisteína": {
        "preco": "R$ 19,90 (600mg - 16 envelopes)",
        "indicacao": "Fluidificação de secreções respiratórias",
        "estoque": True
    },

    "guaifenesina": {
        "preco": "R$ 17,90 (xarope)",
        "indicacao": "Expectoração e secreções respiratórias",
        "estoque": True
    },

    "dexametasona": {
        "preco": "R$ 12,90 (4mg - 10 comprimidos) - necessário receita",
        "indicacao": "Processos inflamatórios e alérgicos específicos",
        "estoque": True
    },

    "hidroxizina": {
        "preco": "R$ 18,90 (25mg - 30 comprimidos)",
        "indicacao": "Sintomas alérgicos e coceira",
        "estoque": True
    },

    "prometazina": {
        "preco": "R$ 12,90 (25mg - 20 comprimidos)",
        "indicacao": "Sintomas alérgicos e náuseas",
        "estoque": True
    },

    "dimeticona": {
        "preco": "R$ 11,90 (40mg/ml - gotas)",
        "indicacao": "Alívio de gases",
        "estoque": True
    },

    "hidróxido de alumínio": {
        "preco": "R$ 12,90 (suspensão oral)",
        "indicacao": "Azia e acidez estomacal",
        "estoque": True
    },

    "carbonato de cálcio": {
        "preco": "R$ 15,90 (500mg - 60 comprimidos)",
        "indicacao": "Suplementação de cálcio",
        "estoque": True
    },

    "levotiroxina": {
        "preco": "R$ 18,90 (50mcg - 30 comprimidos)",
        "indicacao": "Reposição hormonal da tireoide",
        "estoque": True
    },

    "amitriptilina": {
        "preco": "Medicamento sujeito a controle especial - necessário receita",
        "indicacao": "Algumas condições de dor e transtornos depressivos",
        "estoque": True
    },

    "sertralina": {
        "preco": "Medicamento sujeito a controle especial - necessário receita",
        "indicacao": "Transtornos depressivos e de ansiedade",
        "estoque": True
    },

    "fluoxetina": {
        "preco": "Medicamento sujeito a controle especial - necessário receita",
        "indicacao": "Transtornos depressivos e outras condições específicas",
        "estoque": True
    },

    "clonazepam": {
        "preco": "Medicamento controlado - necessário receita",
        "indicacao": "Alguns transtornos de ansiedade e convulsões",
        "estoque": True
    },

    "diazepam": {
        "preco": "Medicamento controlado - necessário receita",
        "indicacao": "Alguns transtornos de ansiedade e espasmos musculares",
        "estoque": True
    },


    "acetilsalicilico": {
        "preco": "R$ 9,90 (100mg - 30 comprimidos)",
        "indicacao": "Prevenção cardiovascular conforme orientação médica",
        "estoque": True
    },
   
    "naproxeno": {
        "preco": "R$ 16,90 (550mg - 10 comprimidos)",
        "indicacao": "Dor e inflamação",
        "estoque": True
    },

    "diclofenaco sodico": {
        "preco": "R$ 13,90 (50mg - 20 comprimidos)",
        "indicacao": "Dor e inflamação",
        "estoque": True
    },

    "diclofenaco potassico": {
        "preco": "R$ 15,90 (50mg - 20 comprimidos)",
        "indicacao": "Dor e inflamação",
        "estoque": True
    },

    "meloxicam": {
        "preco": "R$ 18,90 (15mg - 10 comprimidos)",
        "indicacao": "Dor e inflamação",
        "estoque": True
    },

    "celecoxibe": {
        "preco": "R$ 29,90 (200mg - 10 cápsulas)",
        "indicacao": "Dor e inflamação",
        "estoque": False
    },

    "cetoprofeno": {
        "preco": "R$ 17,90 (100mg - 20 comprimidos)",
        "indicacao": "Dor e inflamação",
        "estoque": True
    },

    "nimesulida": {
        "preco": "R$ 12,90 (100mg - 12 comprimidos)",
        "indicacao": "Dor e inflamação",
        "estoque": True
    },

    "prednisona": {
        "preco": "R$ 14,90 (20mg - 20 comprimidos) - necessário receita",
        "indicacao": "Processos inflamatórios e alérgicos",
        "estoque": True
    },

    "prednisolona": {
        "preco": "R$ 19,90 (20mg - 20 comprimidos) - necessário receita",
        "indicacao": "Processos inflamatórios e alérgicos",
        "estoque": True
    },

    "dexametasona": {
        "preco": "R$ 12,90 (4mg - 10 comprimidos) - necessário receita",
        "indicacao": "Processos inflamatórios e alérgicos",
        "estoque": True
    },

    "cetirizina": {
        "preco": "R$ 10,90 (10mg - 12 comprimidos)",
        "indicacao": "Alergias e rinite",
        "estoque": True
    },

    "desloratadina": {
        "preco": "R$ 16,90 (5mg - 10 comprimidos)",
        "indicacao": "Alergias e rinite",
        "estoque": True
    },

    "fexofenadina": {
        "preco": "R$ 21,90 (120mg - 10 comprimidos)",
        "indicacao": "Alergias e rinite",
        "estoque": True
    },

    "levocetirizina": {
        "preco": "R$ 18,90 (5mg - 10 comprimidos)",
        "indicacao": "Alergias e rinite",
        "estoque": True
    },

    "hidroxizina": {
        "preco": "R$ 15,90 (25mg - 30 comprimidos) - necessário receita",
        "indicacao": "Alergias e coceira",
        "estoque": False
    },

    "prometazina": {
        "preco": "R$ 13,90 (25mg - 20 comprimidos)",
        "indicacao": "Alergias e náuseas",
        "estoque": True
    },

    "domperidona": {
        "preco": "R$ 14,90 (10mg - 30 comprimidos)",
        "indicacao": "Náuseas e distúrbios gastrointestinais",
        "estoque": True
    },

    "metoclopramida": {
        "preco": "R$ 9,90 (10mg - 20 comprimidos)",
        "indicacao": "Náuseas e vômitos",
        "estoque": True
    },

    "ondansetrona": {
        "preco": "R$ 24,90 (8mg - 10 comprimidos)",
        "indicacao": "Prevenção e tratamento de náuseas e vômitos",
        "estoque": True
    },

    "simeticona": {
        "preco": "R$ 11,90 (40mg - 20 comprimidos)",
        "indicacao": "Gases e desconforto abdominal",
        "estoque": True
    },

    "loperamida": {
        "preco": "R$ 10,90 (2mg - 12 cápsulas)",
        "indicacao": "Diarreia aguda",
        "estoque": True
    },

    "sais_reidratacao": {
        "preco": "R$ 5,90 (envelope para solução oral)",
        "indicacao": "Reposição de líquidos e eletrólitos",
        "estoque": True
    },

    "lactulose": {
        "preco": "R$ 18,90 (667mg/mL - 120mL)",
        "indicacao": "Constipação intestinal",
        "estoque": True
    },

    "bisacodil": {
        "preco": "R$ 12,90 (5mg - 20 comprimidos)",
        "indicacao": "Constipação intestinal",
        "estoque": True
    },

    "polietilenoglicol": {
        "preco": "R$ 29,90 (sachês)",
        "indicacao": "Constipação intestinal",
        "estoque": False
    },

    "pantoprazol": {
        "preco": "R$ 19,90 (40mg - 28 comprimidos)",
        "indicacao": "Refluxo e redução da acidez gástrica",
        "estoque": True
    },

    "esomeprazol": {
        "preco": "R$ 27,90 (40mg - 28 cápsulas)",
        "indicacao": "Refluxo e redução da acidez gástrica",
        "estoque": True
    },

    "lansoprazol": {
        "preco": "R$ 22,90 (30mg - 28 cápsulas)",
        "indicacao": "Refluxo e problemas relacionados à acidez",
        "estoque": True
    },

    "famotidina": {
        "preco": "R$ 18,90 (20mg - 20 comprimidos)",
        "indicacao": "Azia e excesso de acidez",
        "estoque": False
    },

    "hidroxido_magnesio": {
        "preco": "R$ 12,90 (suspensão oral - 120mL)",
        "indicacao": "Azia e constipação",
        "estoque": True
    },

    "carbonato_calcio": {
        "preco": "R$ 21,90 (500mg - 60 comprimidos)",
        "indicacao": "Suplementação de cálcio",
        "estoque": True
    },

    "enalapril": {
        "preco": "R$ 13,90 (10mg - 30 comprimidos)",
        "indicacao": "Pressão alta e insuficiência cardíaca",
        "estoque": True
    },

    "captopril": {
        "preco": "R$ 10,90 (25mg - 30 comprimidos)",
        "indicacao": "Pressão alta e insuficiência cardíaca",
        "estoque": True
    },

    "ramipril": {
        "preco": "R$ 22,90 (5mg - 30 comprimidos)",
        "indicacao": "Pressão alta e doenças cardiovasculares",
        "estoque": True
    },

    "lisinopril": {
        "preco": "R$ 18,90 (10mg - 30 comprimidos)",
        "indicacao": "Pressão alta",
        "estoque": True
    },

    "valsartana": {
        "preco": "R$ 23,90 (80mg - 30 comprimidos)",
        "indicacao": "Pressão alta",
        "estoque": True
    },

    "candesartana": {
        "preco": "R$ 26,90 (8mg - 30 comprimidos)",
        "indicacao": "Pressão alta",
        "estoque": True
    },

    "irbesartana": {
        "preco": "R$ 24,90 (150mg - 30 comprimidos)",
        "indicacao": "Pressão alta",
        "estoque": True
    },

    "telmisartana": {
        "preco": "R$ 29,90 (40mg - 30 comprimidos)",
        "indicacao": "Pressão alta",
        "estoque": True
    },

    "anlodipino": {
        "preco": "R$ 12,90 (5mg - 30 comprimidos)",
        "indicacao": "Pressão alta e angina",
        "estoque": True
    },

    "nifedipino": {
        "preco": "R$ 18,90 (20mg - 30 comprimidos)",
        "indicacao": "Pressão alta",
        "estoque": True
    },

    "atenolol": {
        "preco": "R$ 11,90 (50mg - 30 comprimidos)",
        "indicacao": "Pressão alta e controle cardíaco",
        "estoque": True
    },

    "propranolol": {
        "preco": "R$ 10,90 (40mg - 30 comprimidos)",
        "indicacao": "Pressão alta e algumas condições cardíacas",
        "estoque": True
    },

    "metoprolol": {
        "preco": "R$ 19,90 (50mg - 30 comprimidos)",
        "indicacao": "Controle da pressão e frequência cardíaca",
        "estoque": True
    },

    "carvedilol": {
        "preco": "R$ 21,90 (6,25mg - 30 comprimidos)",
        "indicacao": "Insuficiência cardíaca e pressão alta",
        "estoque": True
    },

    "furosemida": {
        "preco": "R$ 8,90 (40mg - 20 comprimidos)",
        "indicacao": "Retenção de líquidos e edema",
        "estoque": True
    },

    "hidroclorotiazida": {
        "preco": "R$ 9,90 (25mg - 30 comprimidos)",
        "indicacao": "Pressão alta e retenção de líquidos",
        "estoque": True
    },

    "espironolactona": {
        "preco": "R$ 17,90 (25mg - 30 comprimidos)",
        "indicacao": "Retenção de líquidos e algumas doenças cardíacas",
        "estoque": True
    },

    "clortalidona": {
        "preco": "R$ 15,90 (25mg - 30 comprimidos)",
        "indicacao": "Pressão alta",
        "estoque": False
    },

    "sinvastatina": {
        "preco": "R$ 14,90 (20mg - 30 comprimidos)",
        "indicacao": "Controle do colesterol",
        "estoque": True
    },

    "atorvastatina": {
        "preco": "R$ 24,90 (20mg - 30 comprimidos)",
        "indicacao": "Controle do colesterol",
        "estoque": True
    },

    "rosuvastatina": {
        "preco": "R$ 29,90 (20mg - 30 comprimidos)",
        "indicacao": "Controle do colesterol",
        "estoque": True
    },

    "pravastatina": {
        "preco": "R$ 21,90 (20mg - 30 comprimidos)",
        "indicacao": "Controle do colesterol",
        "estoque": False
    },

    "fenofibrato": {
        "preco": "R$ 31,90 (160mg - 30 comprimidos)",
        "indicacao": "Controle de triglicerídeos e colesterol",
        "estoque": True
    },

    "metformina": {
        "preco": "R$ 12,90 (500mg - 30 comprimidos)",
        "indicacao": "Controle do diabetes tipo 2",
        "estoque": True
    },

    "metformina_850": {
        "preco": "R$ 16,90 (850mg - 30 comprimidos)",
        "indicacao": "Controle do diabetes tipo 2",
        "estoque": True
    },

    "glibenclamida": {
        "preco": "R$ 9,90 (5mg - 30 comprimidos)",
        "indicacao": "Controle do diabetes tipo 2",
        "estoque": True
    },

    "gliclazida": {
        "preco": "R$ 18,90 (30mg - 30 comprimidos)",
        "indicacao": "Controle do diabetes tipo 2",
        "estoque": True
    },

    "glimepirida": {
        "preco": "R$ 17,90 (2mg - 30 comprimidos)",
        "indicacao": "Controle do diabetes tipo 2",
        "estoque": True
    },

    "empagliflozina": {
        "preco": "R$ 89,90 (10mg - 30 comprimidos)",
        "indicacao": "Diabetes tipo 2 e condições cardiovasculares",
        "estoque": False
    },

    "dapagliflozina": {
        "preco": "R$ 79,90 (10mg - 30 comprimidos)",
        "indicacao": "Diabetes tipo 2 e algumas condições cardíacas",
        "estoque": True
    },

    "sitagliptina": {
        "preco": "R$ 69,90 (100mg - 28 comprimidos)",
        "indicacao": "Controle do diabetes tipo 2",
        "estoque": True
    },

    "pioglitazona": {
        "preco": "R$ 24,90 (30mg - 30 comprimidos)",
        "indicacao": "Controle do diabetes tipo 2",
        "estoque": False
    },

    "levotiroxina": {
        "preco": "R$ 15,90 (50mcg - 30 comprimidos)",
        "indicacao": "Reposição de hormônio da tireoide",
        "estoque": True
    },

    "levotiroxina_100": {
        "preco": "R$ 18,90 (100mcg - 30 comprimidos)",
        "indicacao": "Reposição de hormônio da tireoide",
        "estoque": True
    },

    "metimazol": {
        "preco": "R$ 25,90 (10mg - 30 comprimidos)",
        "indicacao": "Tratamento do hipertireoidismo",
        "estoque": False
    },

    "amoxicilina_clavulanato": {
        "preco": "R$ 39,90 (875mg + 125mg - 14 comprimidos) - necessário receita",
        "indicacao": "Infecções bacterianas",
        "estoque": True
    },

    "azitromicina": {
        "preco": "R$ 24,90 (500mg - 5 comprimidos) - necessário receita",
        "indicacao": "Infecções bacterianas",
        "estoque": True
    },

    "cefalexina": {
        "preco": "R$ 27,90 (500mg - 28 cápsulas) - necessário receita",
        "indicacao": "Infecções bacterianas",
        "estoque": True
    },

    "cefuroxima": {
        "preco": "R$ 44,90 (500mg - 14 comprimidos) - necessário receita",
        "indicacao": "Infecções bacterianas",
        "estoque": False
    },

    "ciprofloxacino": {
        "preco": "R$ 19,90 (500mg - 14 comprimidos) - necessário receita",
        "indicacao": "Algumas infecções bacterianas",
        "estoque": True
    },

    "levofloxacino": {
        "preco": "R$ 29,90 (500mg - 7 comprimidos) - necessário receita",
        "indicacao": "Algumas infecções bacterianas",
        "estoque": True
    },

    "clindamicina": {
        "preco": "R$ 32,90 (300mg - 16 cápsulas) - necessário receita",
        "indicacao": "Infecções bacterianas",
        "estoque": False
    },

    "doxiciclina": {
        "preco": "R$ 21,90 (100mg - 15 cápsulas) - necessário receita",
        "indicacao": "Algumas infecções bacterianas",
        "estoque": True
    },

    "metronidazol": {
        "preco": "R$ 14,90 (250mg - 20 comprimidos) - necessário receita",
        "indicacao": "Infecções bacterianas e parasitárias",
        "estoque": True
    },

    "nitrofurantoina": {
        "preco": "R$ 18,90 (100mg - 28 cápsulas) - necessário receita",
        "indicacao": "Algumas infecções urinárias",
        "estoque": True
    },

    "sulfametoxazol_trim": {
        "preco": "R$ 15,90 (400mg + 80mg - 20 comprimidos) - necessário receita",
        "indicacao": "Algumas infecções bacterianas",
        "estoque": True
    },

    "fluconazol": {
        "preco": "R$ 14,90 (150mg - 1 cápsula)",
        "indicacao": "Infecções por fungos",
        "estoque": True
    },

    "itraconazol": {
        "preco": "R$ 39,90 (100mg - 15 cápsulas)",
        "indicacao": "Infecções fúngicas",
        "estoque": False
    },

    "cetoconazol": {
        "preco": "R$ 18,90 (200mg - 10 comprimidos)",
        "indicacao": "Infecções fúngicas",
        "estoque": True
    },

    "aciclovir": {
        "preco": "R$ 24,90 (200mg - 25 comprimidos)",
        "indicacao": "Infecções causadas por alguns vírus herpes",
        "estoque": True
    },

    "oseltamivir": {
        "preco": "R$ 79,90 (75mg - 10 cápsulas) - necessário receita",
        "indicacao": "Tratamento da influenza conforme orientação médica",
        "estoque": False
    },

    "salbutamol": {
        "preco": "R$ 22,90 (100mcg - inalador)",
        "indicacao": "Alívio do broncoespasmo",
        "estoque": True
    },

    "budesonida": {
        "preco": "R$ 39,90 (inalador)",
        "indicacao": "Controle de inflamação das vias respiratórias",
        "estoque": True
    },

    "beclometasona": {
        "preco": "R$ 34,90 (inalador)",
        "indicacao": "Controle de inflamação das vias respiratórias",
        "estoque": True
    },

    "formoterol": {
        "preco": "R$ 49,90 (inalador)",
        "indicacao": "Controle de sintomas respiratórios",
        "estoque": False
    },

    "montelucaste": {
        "preco": "R$ 29,90 (10mg - 30 comprimidos)",
        "indicacao": "Controle de asma e rinite alérgica",
        "estoque": True
    },

    "acetilcisteina": {
        "preco": "R$ 18,90 (600mg - 16 sachês)",
        "indicacao": "Fluidificação de secreções respiratórias",
        "estoque": True
    },

    "ambroxol": {
        "preco": "R$ 15,90 (xarope - 120mL)",
        "indicacao": "Auxílio na eliminação de secreções respiratórias",
        "estoque": True
    },

    "bromexina": {
        "preco": "R$ 13,90 (xarope - 120mL)",
        "indicacao": "Fluidificação de secreções respiratórias",
        "estoque": True
    },

    "guaifenesina": {
        "preco": "R$ 17,90 (xarope - 120mL)",
        "indicacao": "Auxílio na eliminação de secreções",
        "estoque": True
    },

    "oximetazolina": {
        "preco": "R$ 16,90 (spray nasal)",
        "indicacao": "Congestão nasal",
        "estoque": True
    },

    "budesonida_nasal": {
        "preco": "R$ 31,90 (spray nasal)",
        "indicacao": "Rinite alérgica",
        "estoque": True
    },

    "beclometasona_nasal": {
        "preco": "R$ 28,90 (spray nasal)",
        "indicacao": "Rinite e inflamação nasal",
        "estoque": False
    },

    "soro_fisiologico": {
        "preco": "R$ 7,90 (solução nasal 30mL)",
        "indicacao": "Higiene e hidratação nasal",
        "estoque": True
    },

    "clotrimazol": {
        "preco": "R$ 13,90 (creme dermatológico)",
        "indicacao": "Infecções fúngicas da pele",
        "estoque": True
    },

    "miconazol": {
        "preco": "R$ 15,90 (creme dermatológico)",
        "indicacao": "Infecções fúngicas da pele",
        "estoque": True
    },

    "terbinafina": {
        "preco": "R$ 24,90 (creme dermatológico)",
        "indicacao": "Infecções fúngicas da pele",
        "estoque": True
    },

    "nitrato_miconazol": {
        "preco": "R$ 17,90 (creme dermatológico)",
        "indicacao": "Micoses superficiais",
        "estoque": True
    },

    "hidrocortisona": {
        "preco": "R$ 11,90 (creme dermatológico)",
        "indicacao": "Inflamações e irritações da pele",
        "estoque": True
    },

    "betametasona": {
        "preco": "R$ 18,90 (creme dermatológico)",
        "indicacao": "Inflamações da pele",
        "estoque": False
    },

    "neomicina_bacitracina": {
        "preco": "R$ 14,90 (pomada dermatológica)",
        "indicacao": "Algumas infecções superficiais da pele",
        "estoque": True
    },

    "permanganato_potassio": {
        "preco": "R$ 8,90 (comprimidos para solução)",
        "indicacao": "Uso externo conforme orientação",
        "estoque": False
    },

    "calamina": {
        "preco": "R$ 16,90 (loção 100mL)",
        "indicacao": "Alívio de irritações e coceira da pele",
        "estoque": True
    },

    "ureia_creme": {
        "preco": "R$ 19,90 (creme 10%)",
        "indicacao": "Hidratação de pele ressecada",
        "estoque": True
    },

    "dexpantenol": {
        "preco": "R$ 17,90 (pomada dermatológica)",
        "indicacao": "Hidratação e proteção da pele",
        "estoque": True
    },

    "isotretinoina": {
        "preco": "R$ 89,90 (20mg - 30 cápsulas) - necessário receita especial",
        "indicacao": "Tratamento de acne grave sob acompanhamento médico",
        "estoque": False
    },

    "adapaleno": {
        "preco": "R$ 32,90 (gel dermatológico)",
        "indicacao": "Tratamento tópico da acne",
        "estoque": True
    },

    "peroxido_benzoila": {
        "preco": "R$ 27,90 (gel dermatológico)",
        "indicacao": "Tratamento tópico da acne",
        "estoque": True
    },

    "tretinoina": {
        "preco": "R$ 34,90 (creme dermatológico) - necessário receita",
        "indicacao": "Tratamento dermatológico sob orientação médica",
        "estoque": False
    },

    "finasterida": {
        "preco": "R$ 29,90 (1mg - 30 comprimidos)",
        "indicacao": "Tratamento da queda de cabelo em homens",
        "estoque": True
    },

    "tansulosina": {
        "preco": "R$ 39,90 (0,4mg - 30 cápsulas)",
        "indicacao": "Sintomas urinários relacionados à próstata",
        "estoque": True
    },

    "doxazosina": {
        "preco": "R$ 21,90 (2mg - 30 comprimidos)",
        "indicacao": "Sintomas urinários e pressão alta",
        "estoque": True
    },

    "solifenacina": {
        "preco": "R$ 49,90 (5mg - 30 comprimidos)",
        "indicacao": "Sintomas de bexiga hiperativa",
        "estoque": False
    },

    "oxibutinina": {
        "preco": "R$ 24,90 (5mg - 30 comprimidos)",
        "indicacao": "Sintomas de bexiga hiperativa",
        "estoque": True
    },

    "finasterida_5mg": {
        "preco": "R$ 35,90 (5mg - 30 comprimidos)",
        "indicacao": "Tratamento de sintomas da próstata",
        "estoque": True
    },

    "sertralina": {
        "preco": "R$ 29,90 (50mg - 30 comprimidos) - necessário receita",
        "indicacao": "Tratamento de depressão e transtornos de ansiedade",
        "estoque": True
    },

    "fluoxetina": {
        "preco": "R$ 18,90 (20mg - 30 cápsulas) - necessário receita",
        "indicacao": "Tratamento de depressão e alguns transtornos",
        "estoque": True
    },

    "escitalopram": {
        "preco": "R$ 34,90 (10mg - 30 comprimidos) - necessário receita",
        "indicacao": "Tratamento de depressão e ansiedade",
        "estoque": True
    },

    "paroxetina": {
        "preco": "R$ 27,90 (20mg - 30 comprimidos) - necessário receita",
        "indicacao": "Tratamento de depressão e ansiedade",
        "estoque": False
    },

    "venlafaxina": {
        "preco": "R$ 39,90 (75mg - 30 cápsulas) - necessário receita",
        "indicacao": "Tratamento de depressão e ansiedade",
        "estoque": True
    },

    "duloxetina": {
        "preco": "R$ 44,90 (30mg - 30 cápsulas) - necessário receita",
        "indicacao": "Tratamento de depressão e algumas dores crônicas",
        "estoque": True
    },

    "amitriptilina": {
        "preco": "R$ 16,90 (25mg - 30 comprimidos) - necessário receita",
        "indicacao": "Depressão e algumas condições de dor",
        "estoque": True
    },

    "nortriptilina": {
        "preco": "R$ 24,90 (25mg - 30 cápsulas) - necessário receita",
        "indicacao": "Depressão e algumas condições de dor",
        "estoque": False
    },

    "trazodona": {
        "preco": "R$ 31,90 (50mg - 30 comprimidos) - necessário receita",
        "indicacao": "Depressão e transtornos do sono",
        "estoque": True
    },

    "quetiapina": {
        "preco": "R$ 39,90 (25mg - 30 comprimidos) - necessário receita",
        "indicacao": "Transtornos psiquiátricos específicos",
        "estoque": True
    },

    "risperidona": {
        "preco": "R$ 24,90 (1mg - 30 comprimidos) - necessário receita",
        "indicacao": "Transtornos psiquiátricos específicos",
        "estoque": True
    },

    "olanzapina": {
        "preco": "R$ 39,90 (5mg - 30 comprimidos) - necessário receita",
        "indicacao": "Transtornos psiquiátricos específicos",
        "estoque": False
    },

    "haloperidol": {
        "preco": "R$ 15,90 (5mg - 20 comprimidos) - necessário receita",
        "indicacao": "Transtornos psiquiátricos específicos",
        "estoque": True
    },

    "carbamazepina": {
        "preco": "R$ 21,90 (200mg - 20 comprimidos) - necessário receita",
        "indicacao": "Epilepsia e algumas condições neurológicas",
        "estoque": True
    },

    "valproato_sodio": {
        "preco": "R$ 29,90 (500mg - 30 comprimidos) - necessário receita",
        "indicacao": "Epilepsia e transtornos específicos",
        "estoque": True
    },

    "lamotrigina": {
        "preco": "R$ 39,90 (100mg - 30 comprimidos) - necessário receita",
        "indicacao": "Epilepsia e transtornos específicos",
        "estoque": False
    },

    "gabapentina": {
        "preco": "R$ 34,90 (300mg - 30 cápsulas) - necessário receita",
        "indicacao": "Algumas condições neurológicas e dores",
        "estoque": True
    },

    "pregabalina": {
        "preco": "R$ 49,90 (75mg - 30 cápsulas) - necessário receita",
        "indicacao": "Algumas dores neuropáticas e condições específicas",
        "estoque": True
    },

    "clonazepam": {
        "preco": "Medicamento controlado - necessário receita especial",
        "indicacao": "Alguns transtornos de ansiedade e epilepsia",
        "estoque": True
    },

    "diazepam": {
        "preco": "Medicamento controlado - necessário receita especial",
        "indicacao": "Ansiedade, espasmos musculares e outras condições",
        "estoque": False
    },

    "alprazolam": {
        "preco": "Medicamento controlado - necessário receita especial",
        "indicacao": "Transtornos de ansiedade",
        "estoque": True
    },

    "lorazepam": {
        "preco": "Medicamento controlado - necessário receita especial",
        "indicacao": "Transtornos de ansiedade",
        "estoque": False
    },

    "zolpidem": {
        "preco": "Medicamento controlado - necessário receita especial",
        "indicacao": "Tratamento de insônia",
        "estoque": True
    },

    "biperideno": {
        "preco": "R$ 19,90 (2mg - 20 comprimidos) - necessário receita",
        "indicacao": "Sintomas motores específicos",
        "estoque": True
    },

    "levodopa_benserazida": {
        "preco": "R$ 49,90 (comprimidos) - necessário receita",
        "indicacao": "Tratamento da doença de Parkinson",
        "estoque": False
    },

    "pramipexol": {
        "preco": "R$ 59,90 (0,25mg - 30 comprimidos) - necessário receita",
        "indicacao": "Doença de Parkinson e condições específicas",
        "estoque": True
    },

    "donepezila": {
        "preco": "R$ 49,90 (10mg - 30 comprimidos) - necessário receita",
        "indicacao": "Tratamento sintomático da doença de Alzheimer",
        "estoque": True
    },

    "memantina": {
        "preco": "R$ 59,90 (10mg - 30 comprimidos) - necessário receita",
        "indicacao": "Tratamento sintomático da doença de Alzheimer",
        "estoque": False
    },

    "levetiracetam": {
        "preco": "R$ 44,90 (500mg - 30 comprimidos) - necessário receita",
        "indicacao": "Tratamento de epilepsia",
        "estoque": True
    },

    "topiramato": {
        "preco": "R$ 34,90 (50mg - 30 comprimidos) - necessário receita",
        "indicacao": "Epilepsia e prevenção de enxaqueca",
        "estoque": True
    },

    "sumatriptana": {
        "preco": "R$ 29,90 (50mg - 4 comprimidos)",
        "indicacao": "Tratamento de crises de enxaqueca",
        "estoque": True
    },

    "naratriptana": {
        "preco": "R$ 34,90 (2,5mg - 4 comprimidos)",
        "indicacao": "Tratamento de crises de enxaqueca",
        "estoque": False
    },

    "propranolol_enxaqueca": {
        "preco": "R$ 15,90 (40mg - 30 comprimidos)",
        "indicacao": "Prevenção de enxaqueca conforme orientação médica",
        "estoque": True
    },

    "ciclobenzaprina": {
        "preco": "R$ 18,90 (10mg - 15 comprimidos) - necessário receita",
        "indicacao": "Espasmos e dores musculares",
        "estoque": True
    },

    "tizanidina": {
        "preco": "R$ 29,90 (2mg - 20 comprimidos) - necessário receita",
        "indicacao": "Espasmos musculares",
        "estoque": False
    },

    "baclofeno": {
        "preco": "R$ 24,90 (10mg - 20 comprimidos) - necessário receita",
        "indicacao": "Espasticidade muscular",
        "estoque": True
    },

    "tramadol": {
        "preco": "Medicamento controlado - necessário receita especial",
        "indicacao": "Tratamento de dores moderadas a intensas",
        "estoque": True
    },

    "codeina": {
        "preco": "Medicamento controlado - necessário receita especial",
        "indicacao": "Tratamento de algumas dores",
        "estoque": False
    },

    "morfina": {
        "preco": "Medicamento controlado - necessário receita especial",
        "indicacao": "Tratamento de dores intensas",
        "estoque": False
    },

    "oxicodona": {
        "preco": "Medicamento controlado - necessário receita especial",
        "indicacao": "Tratamento de dores intensas",
        "estoque": False
    },

    "lidocaina": {
        "preco": "R$ 19,90 (gel anestésico)",
        "indicacao": "Anestesia local",
        "estoque": True
    },

    "benzocaina": {
        "preco": "R$ 14,90 (gel tópico)",
        "indicacao": "Anestesia local superficial",
        "estoque": True
    },

    "clorexidina": {
        "preco": "R$ 15,90 (solução antisséptica)",
        "indicacao": "Antissepsia da pele",
        "estoque": True
    },

    "alcool_70": {
        "preco": "R$ 8,90 (100mL)",
        "indicacao": "Higienização e antissepsia",
        "estoque": True
    },

    "iodopovidona": {
        "preco": "R$ 18,90 (solução antisséptica)",
        "indicacao": "Antissepsia",
        "estoque": True
    },

    "sulfadiazina_prata": {
        "preco": "R$ 29,90 (creme dermatológico)",
        "indicacao": "Cuidados específicos com queimaduras",
        "estoque": False
    },

    "colagenase": {
        "preco": "R$ 39,90 (pomada dermatológica)",
        "indicacao": "Tratamento de feridas específicas",
        "estoque": True
    },

    "povidona": {
        "preco": "R$ 17,90 (solução tópica)",
        "indicacao": "Antissepsia",
        "estoque": True
    },

    "carboximetilcelulose": {
        "preco": "R$ 24,90 (colírio)",
        "indicacao": "Lubrificação ocular",
        "estoque": True
    },

    "hipromelose": {
        "preco": "R$ 19,90 (colírio)",
        "indicacao": "Lubrificação dos olhos",
        "estoque": True
    },

    "timolol": {
        "preco": "R$ 29,90 (colírio) - necessário receita",
        "indicacao": "Tratamento de glaucoma",
        "estoque": False
    },

    "latanoprosta": {
        "preco": "R$ 49,90 (colírio) - necessário receita",
        "indicacao": "Tratamento de glaucoma",
        "estoque": True
    },

    "brimonidina": {
        "preco": "R$ 39,90 (colírio) - necessário receita",
        "indicacao": "Controle da pressão intraocular",
        "estoque": False
    },

    "dorzolamida": {
        "preco": "R$ 44,90 (colírio) - necessário receita",
        "indicacao": "Tratamento de glaucoma",
        "estoque": True
    },

    "tobramicina": {
        "preco": "R$ 27,90 (colírio) - necessário receita",
        "indicacao": "Infecções bacterianas oculares",
        "estoque": True
    },

    "moxifloxacino_colirio": {
        "preco": "R$ 34,90 (colírio) - necessário receita",
        "indicacao": "Infecções bacterianas oculares",
        "estoque": False
    },

    "cloranfenicol": {
        "preco": "R$ 18,90 (colírio)",
        "indicacao": "Infecções oculares específicas",
        "estoque": True
    },

    "cetoprofeno_gel": {
        "preco": "R$ 22,90 (gel tópico)",
        "indicacao": "Dor e inflamação localizada",
        "estoque": True
    },

    "diclofenaco_gel": {
        "preco": "R$ 19,90 (gel tópico)",
        "indicacao": "Dor e inflamação localizada",
        "estoque": True
    },

    "arnica_gel": {
        "preco": "R$ 18,90 (gel tópico)",
        "indicacao": "Alívio de desconforto muscular",
        "estoque": True
    },

    "mentol_canfora": {
        "preco": "R$ 14,90 (pomada tópica)",
        "indicacao": "Alívio temporário de dores musculares",
        "estoque": True
    },

    "vitamina_d": {
        "preco": "R$ 19,90 (1000UI - 30 cápsulas)",
        "indicacao": "Suplementação de vitamina D",
        "estoque": True
    },

    "vitamina_d3": {
        "preco": "R$ 24,90 (2000UI - 30 cápsulas)",
        "indicacao": "Suplementação de vitamina D",
        "estoque": True
    },

    "vitamina_b12": {
        "preco": "R$ 22,90 (1000mcg - 30 comprimidos)",
        "indicacao": "Suplementação de vitamina B12",
        "estoque": True
    },

    "complexo_b": {
        "preco": "R$ 18,90 (30 comprimidos)",
        "indicacao": "Suplementação de vitaminas do complexo B",
        "estoque": True
    },

    "vitamina_e": {
        "preco": "R$ 21,90 (400UI - 30 cápsulas)",
        "indicacao": "Suplementação de vitamina E",
        "estoque": True
    },

    "zinco": {
        "preco": "R$ 16,90 (30 comprimidos)",
        "indicacao": "Suplementação de zinco",
        "estoque": True
    },

    "magnesio": {
        "preco": "R$ 23,90 (60 comprimidos)",
        "indicacao": "Suplementação de magnésio",
        "estoque": True
    },

    "ferro": {
        "preco": "R$ 17,90 (30 comprimidos)",
        "indicacao": "Suplementação de ferro",
        "estoque": True
    },

    "acido_folico": {
        "preco": "R$ 9,90 (5mg - 30 comprimidos)",
        "indicacao": "Suplementação de ácido fólico",
        "estoque": True
    },

    "sulfato_ferroso": {
        "preco": "R$ 12,90 (40mg - 30 comprimidos)",
        "indicacao": "Reposição de ferro",
        "estoque": True
    },

    "vitamina_a": {
        "preco": "R$ 15,90 (cápsulas)",
        "indicacao": "Suplementação de vitamina A",
        "estoque": False
    },

    "vitamina_k": {
        "preco": "R$ 29,90 (cápsulas)",
        "indicacao": "Suplementação de vitamina K",
        "estoque": True
    },

    "polivitaminico": {
        "preco": "R$ 29,90 (30 comprimidos)",
        "indicacao": "Suplementação de vitaminas e minerais",
        "estoque": True
    },

    "omega_3": {
        "preco": "R$ 34,90 (60 cápsulas)",
        "indicacao": "Suplementação nutricional",
        "estoque": True
    },

    "coenzima_q10": {
        "preco": "R$ 49,90 (30 cápsulas)",
        "indicacao": "Suplementação nutricional",
        "estoque": False
    },

    "melatonina": {
        "preco": "R$ 24,90 (3mg - 30 comprimidos)",
        "indicacao": "Auxílio na regulação do sono",
        "estoque": True
    },

    "carisoprodol": {
        "preco": "Medicamento controlado - necessário receita",
        "indicacao": "Relaxamento muscular em condições específicas",
        "estoque": False
    },

    "orfenadrina": {
        "preco": "R$ 19,90 (comprimidos) - necessário receita",
        "indicacao": "Espasmos e dores musculares",
        "estoque": True
    },

    "metocarbamol": {
        "preco": "R$ 29,90 (500mg - comprimidos)",
        "indicacao": "Espasmos e desconforto muscular",
        "estoque": True
    },

    "alendronato": {
        "preco": "R$ 24,90 (70mg - 4 comprimidos)",
        "indicacao": "Tratamento e prevenção da osteoporose",
        "estoque": True
    },

    "risedronato": {
        "preco": "R$ 39,90 (35mg - 4 comprimidos)",
        "indicacao": "Tratamento da osteoporose",
        "estoque": False
    },

    "carbonato_calcio_vitamina_d": {
        "preco": "R$ 29,90 (60 comprimidos)",
        "indicacao": "Suplementação de cálcio e vitamina D",
        "estoque": True
    },

    "estradiol": {
        "preco": "R$ 39,90 (comprimidos) - necessário receita",
        "indicacao": "Terapia hormonal sob acompanhamento médico",
        "estoque": False
    },

    "progesterona": {
        "preco": "R$ 49,90 (cápsulas) - necessário receita",
        "indicacao": "Terapia hormonal sob acompanhamento médico",
        "estoque": True
    },

    "etinilestradiol_levonorgestrel": {
        "preco": "R$ 18,90 (cartela) - necessário receita",
        "indicacao": "Contracepção hormonal",
        "estoque": True
    },

    "levonorgestrel": {
        "preco": "R$ 19,90 (comprimido)",
        "indicacao": "Contracepção de emergência",
        "estoque": True
    },

    "medroxiprogesterona": {
        "preco": "R$ 29,90 (comprimidos) - necessário receita",
        "indicacao": "Tratamento hormonal específico",
        "estoque": False
    },

    "metotrexato": {
        "preco": "R$ 49,90 (comprimidos) - necessário receita",
        "indicacao": "Doenças autoimunes e outras condições específicas",
        "estoque": False
    },

    "hidroxicloroquina": {
        "preco": "R$ 34,90 (400mg - 30 comprimidos) - necessário receita",
        "indicacao": "Doenças autoimunes específicas",
        "estoque": True
    },

    "sulfassalazina": {
        "preco": "R$ 39,90 (500mg - 30 comprimidos) - necessário receita",
        "indicacao": "Doenças inflamatórias específicas",
        "estoque": False
    },

    "azatioprina": {
        "preco": "R$ 44,90 (50mg - 30 comprimidos) - necessário receita",
        "indicacao": "Doenças autoimunes específicas",
        "estoque": True
    },

    "colchicina": {
        "preco": "R$ 19,90 (0,5mg - 20 comprimidos)",
        "indicacao": "Tratamento e prevenção de crises de gota",
        "estoque": True
    },

    "alopurinol": {
        "preco": "R$ 17,90 (100mg - 30 comprimidos)",
        "indicacao": "Controle do ácido úrico",
        "estoque": True
    },

    "probenecida": {
        "preco": "R$ 29,90 (500mg - 30 comprimidos)",
        "indicacao": "Tratamento específico da gota",
        "estoque": False
    },

    "mesalazina": {
        "preco": "R$ 69,90 (500mg - 30 comprimidos) - necessário receita",
        "indicacao": "Doenças inflamatórias intestinais",
        "estoque": True
    },

    "saccharomyces_boulardii": {
        "preco": "R$ 21,90 (cápsulas)",
        "indicacao": "Auxílio na recuperação da flora intestinal",
        "estoque": True
    },

    "probióticos": {
        "preco": "R$ 29,90 (cápsulas)",
        "indicacao": "Suplementação da flora intestinal",
        "estoque": True
    },

    "carvao_ativado": {
        "preco": "R$ 15,90 (cápsulas)",
        "indicacao": "Uso específico conforme orientação profissional",
        "estoque": False
    },

    "sucralfato": {
        "preco": "R$ 27,90 (1g - 30 comprimidos)",
        "indicacao": "Proteção da mucosa gástrica",
        "estoque": True
    },

    "mesalazina_sup": {
        "preco": "R$ 79,90 (supositórios) - necessário receita",
        "indicacao": "Doenças inflamatórias intestinais",
        "estoque": False
    },

    "bisacodil_sup": {
        "preco": "R$ 14,90 (supositórios)",
        "indicacao": "Constipação intestinal",
        "estoque": True
    },

    "glicerina_sup": {
        "preco": "R$ 9,90 (supositórios)",
        "indicacao": "Constipação intestinal",
        "estoque": True
    },

    "diosmina_hesperidina": {
        "preco": "R$ 39,90 (comprimidos)",
        "indicacao": "Sintomas relacionados à circulação venosa",
        "estoque": True
    },

    "castanha_india": {
        "preco": "R$ 24,90 (cápsulas)",
        "indicacao": "Suplementação tradicional para circulação venosa",
        "estoque": False
    },

    "pentoxifilina": {
        "preco": "R$ 29,90 (400mg - 20 comprimidos)",
        "indicacao": "Distúrbios circulatórios específicos",
        "estoque": True
    },

    "cilostazol": {
        "preco": "R$ 39,90 (100mg - 30 comprimidos)",
        "indicacao": "Doença arterial periférica específica",
        "estoque": False
    },

    "clopidogrel": {
        "preco": "R$ 24,90 (75mg - 30 comprimidos)",
        "indicacao": "Prevenção de eventos cardiovasculares conforme orientação médica",
        "estoque": True
    },

    "varfarina": {
        "preco": "R$ 19,90 (5mg - 30 comprimidos) - necessário receita",
        "indicacao": "Anticoagulação",
        "estoque": True
    },

    "rivaroxabana": {
        "preco": "R$ 79,90 (20mg - 28 comprimidos) - necessário receita",
        "indicacao": "Anticoagulação",
        "estoque": False
    },

    "apixabana": {
        "preco": "R$ 89,90 (5mg - 60 comprimidos) - necessário receita",
        "indicacao": "Anticoagulação",
        "estoque": True
    },

    "dabigatrana": {
        "preco": "R$ 99,90 (150mg - 60 cápsulas) - necessário receita",
        "indicacao": "Anticoagulação",
        "estoque": False
    },

    "digoxina": {
        "preco": "R$ 16,90 (0,25mg - 30 comprimidos) - necessário receita",
        "indicacao": "Algumas condições cardíacas",
        "estoque": True
    },

    "amiodarona": {
        "preco": "R$ 29,90 (200mg - 30 comprimidos) - necessário receita",
        "indicacao": "Distúrbios específicos do ritmo cardíaco",
        "estoque": True
    },

    "diltiazem": {
        "preco": "R$ 24,90 (60mg - 30 comprimidos)",
        "indicacao": "Pressão alta e algumas condições cardíacas",
        "estoque": True
    },

    "verapamil": {
        "preco": "R$ 19,90 (80mg - 30 comprimidos)",
        "indicacao": "Pressão alta e algumas condições cardíacas",
        "estoque": False
    },

    "isosorbida_mononitrato": {
        "preco": "R$ 18,90 (20mg - 30 comprimidos)",
        "indicacao": "Prevenção de sintomas de angina",
        "estoque": True
    },

    "nitroglicerina": {
        "preco": "R$ 29,90 (apresentação sublingual)",
        "indicacao": "Alívio de crises de angina conforme orientação médica",
        "estoque": False
    },

    "ivabradina": {
        "preco": "R$ 59,90 (5mg - 30 comprimidos) - necessário receita",
        "indicacao": "Condições cardíacas específicas",
        "estoque": True
    },

    "sacubitril_valsartana": {
        "preco": "R$ 89,90 (comprimidos) - necessário receita",
        "indicacao": "Tratamento de insuficiência cardíaca",
        "estoque": False
    },

    "glucagon": {
        "preco": "R$ 149,90 (kit de emergência) - necessário receita",
        "indicacao": "Tratamento de emergência de hipoglicemia grave",
        "estoque": False
    },

    "insulina_nph": {
        "preco": "R$ 39,90 (frasco) - necessário receita",
        "indicacao": "Controle do diabetes",
        "estoque": True
    },

    "insulina_regular": {
        "preco": "R$ 39,90 (frasco) - necessário receita",
        "indicacao": "Controle do diabetes",
        "estoque": True
    },

    "insulina_glargina": {
        "preco": "R$ 99,90 (caneta) - necessário receita",
        "indicacao": "Controle do diabetes",
        "estoque": False
    },

    "insulina_asparte": {
        "preco": "R$ 109,90 (caneta) - necessário receita",
        "indicacao": "Controle do diabetes",
        "estoque": True
    },

    "semaglutida": {
        "preco": "R$ 299,90 (caneta) - necessário receita",
        "indicacao": "Tratamento de diabetes e outras indicações específicas",
        "estoque": False
    },

    "liraglutida": {
        "preco": "R$ 279,90 (caneta) - necessário receita",
        "indicacao": "Tratamento de diabetes e outras indicações específicas",
        "estoque": True
    },

    "orlistat": {
        "preco": "R$ 49,90 (120mg - 42 cápsulas)",
        "indicacao": "Tratamento da obesidade associado a medidas de estilo de vida",
        "estoque": True
    },

    "sene": {
        "preco": "R$ 9,90 (comprimidos)",
        "indicacao": "Constipação ocasional",
        "estoque": True
    },

    "psyllium": {
        "preco": "R$ 24,90 (pó para solução oral)",
        "indicacao": "Fonte de fibras e auxílio no funcionamento intestinal",
        "estoque": True
    },

    "simeticona_gotas": {
        "preco": "R$ 12,90 (gotas 10mL)",
        "indicacao": "Alívio de gases",
        "estoque": True
    },

    "paracetamol_gotas": {
        "preco": "R$ 12,90 (gotas 15mL)",
        "indicacao": "Dor e febre",
        "estoque": True
    },

    "dipirona_gotas": {
        "preco": "R$ 11,90 (gotas 20mL)",
        "indicacao": "Dor e febre",
        "estoque": True
    },

    "ibuprofeno_suspensao": {
        "preco": "R$ 18,90 (suspensão oral)",
        "indicacao": "Dor, febre e inflamação",
        "estoque": True
    },

    "loratadina_xarope": {
        "preco": "R$ 16,90 (xarope 100mL)",
        "indicacao": "Alergias e coriza",
        "estoque": True
    },

    "cetirizina_gotas": {
        "preco": "R$ 17,90 (gotas 20mL)",
        "indicacao": "Alergias",
        "estoque": False
    },

    "ambroxol_xarope": {
        "preco": "R$ 16,90 (xarope 120mL)",
        "indicacao": "Auxílio na eliminação de secreções",
        "estoque": True
    },

    "prednisolona_solucao": {
        "preco": "R$ 19,90 (solução oral) - necessário receita",
        "indicacao": "Processos inflamatórios e alérgicos",
        "estoque": True
    },

    "amoxicilina_suspensao": {
        "preco": "R$ 29,90 (suspensão oral) - necessário receita",
        "indicacao": "Infecções bacterianas",
        "estoque": True
    },

    "azitromicina_suspensao": {
        "preco": "R$ 27,90 (suspensão oral) - necessário receita",
        "indicacao": "Infecções bacterianas",
        "estoque": False
    },

    "cefalexina_suspensao": {
        "preco": "R$ 31,90 (suspensão oral) - necessário receita",
        "indicacao": "Infecções bacterianas",
        "estoque": True
    },

    "metronidazol_suspensao": {
        "preco": "R$ 18,90 (suspensão oral) - necessário receita",
        "indicacao": "Infecções bacterianas e parasitárias",
        "estoque": True
    },

    "nistatina": {
        "preco": "R$ 15,90 (suspensão oral)",
        "indicacao": "Infecções por fungos na boca",
        "estoque": True
    },

    "nistatina_creme": {
        "preco": "R$ 14,90 (creme dermatológico)",
        "indicacao": "Infecções fúngicas da pele",
        "estoque": True
    },

    "mupirocina": {
        "preco": "R$ 24,90 (pomada dermatológica)",
        "indicacao": "Algumas infecções bacterianas da pele",
        "estoque": False
    },

    "fusidato_sodio": {
        "preco": "R$ 29,90 (creme dermatológico)",
        "indicacao": "Infecções bacterianas da pele",
        "estoque": True
    },

    "aciclovir_creme": {
        "preco": "R$ 16,90 (creme dermatológico)",
        "indicacao": "Lesões de herpes labial",
        "estoque": True
    },

    "terconazol": {
        "preco": "R$ 29,90 (creme vaginal)",
        "indicacao": "Infecções fúngicas específicas",
        "estoque": False
    },

    "miconazol_vaginal": {
        "preco": "R$ 22,90 (creme vaginal)",
        "indicacao": "Infecções fúngicas específicas",
        "estoque": True
    },

    "metronidazol_gel": {
        "preco": "R$ 24,90 (gel vaginal)",
        "indicacao": "Infecções vaginais específicas",
        "estoque": True
    },

    "clindamicina_vaginal": {
        "preco": "R$ 39,90 (creme vaginal) - necessário receita",
        "indicacao": "Infecções bacterianas vaginais específicas",
        "estoque": False
    },

    "tinidazol": {
        "preco": "R$ 22,90 (500mg - comprimidos) - necessário receita",
        "indicacao": "Infecções bacterianas e parasitárias específicas",
        "estoque": True
    },

    "secnidazol": {
        "preco": "R$ 19,90 (2g - envelope)",
        "indicacao": "Algumas infecções parasitárias",
        "estoque": True
    },

    "ivermectina": {
        "preco": "R$ 16,90 (6mg - 4 comprimidos)",
        "indicacao": "Tratamento de algumas parasitoses",
        "estoque": True
    },

    "albendazol": {
        "preco": "R$ 12,90 (400mg - 1 comprimido)",
        "indicacao": "Tratamento de algumas verminoses",
        "estoque": True
    },

    "mebendazol": {
        "preco": "R$ 11,90 (100mg - 6 comprimidos)",
        "indicacao": "Tratamento de algumas verminoses",
        "estoque": True
    },

    "praziquantel": {
        "preco": "R$ 39,90 (600mg - comprimidos)",
        "indicacao": "Tratamento de parasitoses específicas",
        "estoque": False
    },

    "nitazoxanida": {
        "preco": "R$ 29,90 (500mg - 6 comprimidos)",
        "indicacao": "Tratamento de algumas parasitoses",
        "estoque": True
    },

    "hidroxizina_xarope": {
        "preco": "R$ 19,90 (xarope) - necessário receita",
        "indicacao": "Alergias e coceira",
        "estoque": False
    },

    "dimenidrinato": {
        "preco": "R$ 14,90 (50mg - 20 comprimidos)",
        "indicacao": "Náuseas, vômitos e enjoo",
        "estoque": True
    },

    "meclizina": {
        "preco": "R$ 18,90 (25mg - 10 comprimidos)",
        "indicacao": "Náuseas e tonturas",
        "estoque": True
    },

    "butilbrometo_escopolamina": {
        "preco": "R$ 13,90 (10mg - 20 comprimidos)",
        "indicacao": "Cólicas e espasmos gastrointestinais",
        "estoque": True
    },

    "escopolamina_dipirona": {
        "preco": "R$ 19,90 (comprimidos)",
        "indicacao": "Cólica e dor abdominal",
        "estoque": True
    },

    "butilbrometo_escopolamina_gotas": {
        "preco": "R$ 17,90 (gotas)",
        "indicacao": "Cólicas e espasmos",
        "estoque": True
    },

    "metamizol": {
        "preco": "R$ 9,90 (500mg - comprimidos)",
        "indicacao": "Dor e febre",
        "estoque": False
    },

    "sulfato_magnesio": {
        "preco": "R$ 12,90 (solução)",
        "indicacao": "Uso específico conforme indicação profissional",
        "estoque": False
    },

    "cloreto_potassio": {
        "preco": "R$ 18,90 (comprimidos)",
        "indicacao": "Reposição de potássio conforme indicação médica",
        "estoque": True
    },

    "bicarbonato_sodio": {
        "preco": "R$ 8,90 (envelope)",
        "indicacao": "Uso específico conforme orientação",
        "estoque": True
    },

    "ringer_lactato": {
        "preco": "R$ 18,90 (solução intravenosa)",
        "indicacao": "Reposição de líquidos e eletrólitos em ambiente de saúde",
        "estoque": False
    },

    "glicose": {
        "preco": "R$ 9,90 (solução intravenosa)",
        "indicacao": "Uso hospitalar para situações específicas",
        "estoque": True
    },

    "sulfato_ferroso_gotas": {
        "preco": "R$ 14,90 (gotas)",
        "indicacao": "Reposição de ferro",
        "estoque": True
    },

    "acido_folico_gotas": {
        "preco": "R$ 12,90 (gotas)",
        "indicacao": "Suplementação de ácido fólico",
        "estoque": True
    },

    "vitamina_b1": {
        "preco": "R$ 14,90 (comprimidos)",
        "indicacao": "Suplementação de vitamina B1",
        "estoque": True
    },

    "vitamina_b6": {
        "preco": "R$ 16,90 (comprimidos)",
        "indicacao": "Suplementação de vitamina B6",
        "estoque": True
    },

    "biotina": {
        "preco": "R$ 24,90 (cápsulas)",
        "indicacao": "Suplementação nutricional",
        "estoque": True
    },

    "selênio": {
        "preco": "R$ 19,90 (comprimidos)",
        "indicacao": "Suplementação de selênio",
        "estoque": False
    },

    "potassio": {
        "preco": "R$ 18,90 (comprimidos)",
        "indicacao": "Suplementação de potássio conforme orientação",
        "estoque": True
    },

    "calcio": {
        "preco": "R$ 19,90 (comprimidos)",
        "indicacao": "Suplementação de cálcio",
        "estoque": True
    },

    "magnésio_dimalato": {
        "preco": "R$ 29,90 (cápsulas)",
        "indicacao": "Suplementação de magnésio",
        "estoque": True
    },

    "condroitina_glicosamina": {
        "preco": "R$ 69,90 (cápsulas)",
        "indicacao": "Suplementação para articulações",
        "estoque": False
    },

    "glucosamina": {
        "preco": "R$ 59,90 (cápsulas)",
        "indicacao": "Suplementação para articulações",
        "estoque": True
    },

    "colageno": {
        "preco": "R$ 49,90 (sachês)",
        "indicacao": "Suplementação nutricional",
        "estoque": True
    },

    "creatina": {
        "preco": "R$ 79,90 (pó - 300g)",
        "indicacao": "Suplementação esportiva",
        "estoque": True
    },

    "proteina_soro_leite": {
        "preco": "R$ 89,90 (pó - 900g)",
        "indicacao": "Suplementação proteica",
        "estoque": False
    },

    "solucao_glicosada": {
        "preco": "R$ 12,90 (solução)",
        "indicacao": "Uso hospitalar conforme indicação",
        "estoque": True
    },

    "manitol": {
        "preco": "R$ 29,90 (solução intravenosa)",
        "indicacao": "Uso hospitalar em condições específicas",
        "estoque": False
    },

    "furosemida_injetavel": {
        "preco": "R$ 14,90 (ampola) - uso profissional",
        "indicacao": "Uso hospitalar em situações específicas",
        "estoque": True
    },

    "metoclopramida_injetavel": {
        "preco": "R$ 12,90 (ampola) - uso profissional",
        "indicacao": "Tratamento de náuseas e vômitos em situações específicas",
        "estoque": True
    },

    "ondansetrona_injetavel": {
        "preco": "R$ 24,90 (ampola) - uso profissional",
        "indicacao": "Prevenção e tratamento de náuseas e vômitos",
        "estoque": False
    },

    "dipirona_injetavel": {
        "preco": "R$ 15,90 (ampola) - uso profissional",
        "indicacao": "Dor e febre em situações específicas",
        "estoque": True
    },

    "diclofenaco_injetavel": {
        "preco": "R$ 18,90 (ampola) - uso profissional",
        "indicacao": "Dor e inflamação",
        "estoque": True
    },

    "cianocobalamina": {
        "preco": "R$ 19,90 (ampola) - uso profissional",
        "indicacao": "Reposição de vitamina B12",
        "estoque": True
    },

    "betametasona_injetavel": {
        "preco": "R$ 21,90 (ampola) - necessário receita",
        "indicacao": "Processos inflamatórios específicos",
        "estoque": False
    },

    "dexametasona_injetavel": {
        "preco": "R$ 18,90 (ampola) - necessário receita",
        "indicacao": "Processos inflamatórios específicos",
        "estoque": True
    },

    "prometazina_injetavel": {
        "preco": "R$ 16,90 (ampola) - uso profissional",
        "indicacao": "Alergias e náuseas em situações específicas",
        "estoque": False
    },

    "ceftriaxona": {
        "preco": "R$ 39,90 (frasco-ampola) - necessário receita",
        "indicacao": "Tratamento de infecções bacterianas",
        "estoque": True
    },

    "benzilpenicilina": {
        "preco": "R$ 24,90 (frasco-ampola) - necessário receita",
        "indicacao": "Tratamento de infecções bacterianas específicas",
        "estoque": False
    },

    "penicilina_benzatina": {
        "preco": "R$ 19,90 (frasco-ampola) - necessário receita",
        "indicacao": "Tratamento de infecções bacterianas específicas",
        "estoque": True
    },

    "gentamicina": {
        "preco": "R$ 16,90 (ampola) - necessário receita",
        "indicacao": "Tratamento de infecções bacterianas específicas",
        "estoque": False
    },

    "amikacina": {
        "preco": "R$ 29,90 (ampola) - necessário receita",
        "indicacao": "Tratamento de infecções bacterianas específicas",
        "estoque": True
    },

    "meropenem": {
        "preco": "R$ 89,90 (frasco-ampola) - uso hospitalar",
        "indicacao": "Tratamento hospitalar de infecções bacterianas graves",
        "estoque": False
    },

    "vancomicina": {
        "preco": "R$ 49,90 (frasco-ampola) - uso hospitalar",
        "indicacao": "Tratamento hospitalar de infecções bacterianas específicas",
        "estoque": True
    },

    "flucloxacilina": {
        "preco": "R$ 39,90 (cápsulas) - necessário receita",
        "indicacao": "Tratamento de infecções bacterianas específicas",
        "estoque": False
    },

    "rifampicina": {
        "preco": "R$ 34,90 (cápsulas) - necessário receita",
        "indicacao": "Tratamento de infecções específicas",
        "estoque": True
    },

    "isoniazida": {
        "preco": "R$ 24,90 (comprimidos) - necessário receita",
        "indicacao": "Tratamento de tuberculose em esquemas específicos",
        "estoque": True
    },

    "etambutol": {
        "preco": "R$ 29,90 (comprimidos) - necessário receita",
        "indicacao": "Tratamento de tuberculose em esquemas específicos",
        "estoque": False
    },

    "pirazinamida": {
        "preco": "R$ 29,90 (comprimidos) - necessário receita",
        "indicacao": "Tratamento de tuberculose em esquemas específicos",
        "estoque": True
    },

    "etionamida": {
        "preco": "R$ 39,90 (comprimidos) - necessário receita",
        "indicacao": "Tratamento específico de tuberculose",
        "estoque": False
    },

    "rifapentina": {
        "preco": "R$ 89,90 (comprimidos) - necessário receita",
        "indicacao": "Tratamento de tuberculose em situações específicas",
        "estoque": False
    },

    "albendazol_suspensao": {
        "preco": "R$ 17,90 (suspensão oral)",
        "indicacao": "Tratamento de algumas verminoses",
        "estoque": True
    },

    "mebendazol_suspensao": {
        "preco": "R$ 14,90 (suspensão oral)",
        "indicacao": "Tratamento de algumas verminoses",
        "estoque": True
    },

    "cloroquina": {
        "preco": "R$ 29,90 (comprimidos) - necessário receita",
        "indicacao": "Tratamento de doenças específicas conforme indicação médica",
        "estoque": False
    },

    "primaquina": {
        "preco": "R$ 19,90 (comprimidos) - necessário receita",
        "indicacao": "Tratamento de malária em esquemas específicos",
        "estoque": True
    },

    "artemeter_lumefantrina": {
        "preco": "R$ 39,90 (comprimidos) - necessário receita",
        "indicacao": "Tratamento de malária em esquemas específicos",
        "estoque": True
    },

    "benznidazol": {
        "preco": "R$ 49,90 (comprimidos) - necessário receita",
        "indicacao": "Tratamento da doença de Chagas",
        "estoque": False
    },

    "anfotericina_b": {
        "preco": "R$ 99,90 (frasco-ampola) - uso hospitalar",
        "indicacao": "Tratamento de infecções fúngicas graves específicas",
        "estoque": False
    },

    "aciclovir_injetavel": {
        "preco": "R$ 49,90 (frasco-ampola) - uso hospitalar",
        "indicacao": "Tratamento de infecções virais específicas",
        "estoque": True
    },

    "valaciclovir": {
        "preco": "R$ 39,90 (500mg - 10 comprimidos)",
        "indicacao": "Tratamento de algumas infecções por herpes",
        "estoque": True
    },

    "famciclovir": {
        "preco": "R$ 49,90 (500mg - 10 comprimidos)",
        "indicacao": "Tratamento de algumas infecções por herpes",
        "estoque": False
    },

    "oseltamivir_75": {
        "preco": "R$ 79,90 (75mg - 10 cápsulas) - necessário receita",
        "indicacao": "Tratamento da influenza conforme orientação médica",
        "estoque": True
    },

    "zanamivir": {
        "preco": "R$ 69,90 (inalador) - necessário receita",
        "indicacao": "Tratamento da influenza em situações específicas",
        "estoque": False
    },

    "ribavirina": {
        "preco": "R$ 89,90 (cápsulas) - necessário receita",
        "indicacao": "Tratamento antiviral específico",
        "estoque": False
    },

    "interferon": {
        "preco": "R$ 199,90 (apresentação injetável) - necessário receita",
        "indicacao": "Tratamentos específicos sob acompanhamento especializado",
        "estoque": False
    },

    "insulina_detemir": {
        "preco": "R$ 109,90 (caneta) - necessário receita",
        "indicacao": "Controle do diabetes",
        "estoque": True
    },

    "insulina_lispro": {
        "preco": "R$ 109,90 (caneta) - necessário receita",
        "indicacao": "Controle do diabetes",
        "estoque": False
    },

    "insulina_degludeca": {
        "preco": "R$ 119,90 (caneta) - necessário receita",
        "indicacao": "Controle do diabetes",
        "estoque": True
    },

    "liraglutida_6mg": {
        "preco": "R$ 279,90 (caneta) - necessário receita",
        "indicacao": "Tratamento de diabetes e outras indicações específicas",
        "estoque": False
    },

    "exenatida": {
        "preco": "R$ 189,90 (caneta) - necessário receita",
        "indicacao": "Tratamento do diabetes tipo 2",
        "estoque": True
    },

    "acarbose": {
        "preco": "R$ 29,90 (50mg - 30 comprimidos)",
        "indicacao": "Controle do diabetes tipo 2",
        "estoque": True
    },

    "linagliptina": {
        "preco": "R$ 69,90 (5mg - 30 comprimidos)",
        "indicacao": "Controle do diabetes tipo 2",
        "estoque": False
    },

    "alogliptina": {
        "preco": "R$ 59,90 (25mg - 30 comprimidos)",
        "indicacao": "Controle do diabetes tipo 2",
        "estoque": True
    },

    "dapagliflozina_metformina": {
        "preco": "R$ 99,90 (comprimidos) - necessário receita",
        "indicacao": "Controle do diabetes tipo 2",
        "estoque": False
    },

    "empagliflozina_metformina": {
        "preco": "R$ 109,90 (comprimidos) - necessário receita",
        "indicacao": "Controle do diabetes tipo 2",
        "estoque": True
    },

    "colesevelam": {
        "preco": "R$ 69,90 (comprimidos)",
        "indicacao": "Controle do colesterol",
        "estoque": False
    },

    "ezetimiba": {
        "preco": "R$ 29,90 (10mg - 30 comprimidos)",
        "indicacao": "Controle do colesterol",
        "estoque": True
    },

    "ezetimiba_sinvastatina": {
        "preco": "R$ 39,90 (comprimidos)",
        "indicacao": "Controle do colesterol",
        "estoque": True
    },

    "fenofibrato_200": {
        "preco": "R$ 34,90 (200mg - 30 cápsulas)",
        "indicacao": "Controle de triglicerídeos",
        "estoque": False
    },

    "bezafibrato": {
        "preco": "R$ 29,90 (200mg - 30 comprimidos)",
        "indicacao": "Controle de triglicerídeos e colesterol",
        "estoque": True
    },

    "ciprofibrato": {
        "preco": "R$ 24,90 (100mg - 30 comprimidos)",
        "indicacao": "Controle de triglicerídeos e colesterol",
        "estoque": True
    },

    "pitavastatina": {
        "preco": "R$ 39,90 (2mg - 30 comprimidos)",
        "indicacao": "Controle do colesterol",
        "estoque": False
    },

    "fluvastatina": {
        "preco": "R$ 29,90 (80mg - 30 comprimidos)",
        "indicacao": "Controle do colesterol",
        "estoque": True
    },

    "amlodipino_valsartana": {
        "preco": "R$ 39,90 (comprimidos)",
        "indicacao": "Tratamento da pressão alta",
        "estoque": True
    },

    "losartana_hidroclorotiazida": {
        "preco": "R$ 24,90 (comprimidos)",
        "indicacao": "Tratamento da pressão alta",
        "estoque": True
    },

    "enalapril_hidroclorotiazida": {
        "preco": "R$ 19,90 (comprimidos)",
        "indicacao": "Tratamento da pressão alta",
        "estoque": False
    },

    "valsartana_hidroclorotiazida": {
        "preco": "R$ 34,90 (comprimidos)",
        "indicacao": "Tratamento da pressão alta",
        "estoque": True
    },

    "bisoprolol": {
        "preco": "R$ 29,90 (5mg - 30 comprimidos)",
        "indicacao": "Controle da pressão e frequência cardíaca",
        "estoque": True
    },

    "nebivolol": {
        "preco": "R$ 39,90 (5mg - 30 comprimidos)",
        "indicacao": "Tratamento da pressão alta",
        "estoque": False
    },

    "olmesartana": {
        "preco": "R$ 29,90 (20mg - 30 comprimidos)",
        "indicacao": "Tratamento da pressão alta",
        "estoque": True
    },

    "azilsartana": {
        "preco": "R$ 44,90 (40mg - 30 comprimidos)",
        "indicacao": "Tratamento da pressão alta",
        "estoque": False
    },

    "perindopril": {
        "preco": "R$ 29,90 (5mg - 30 comprimidos)",
        "indicacao": "Tratamento da pressão alta e doenças cardiovasculares",
        "estoque": True
    },

    "indapamida": {
        "preco": "R$ 16,90 (1,5mg - 30 comprimidos)",
        "indicacao": "Tratamento da pressão alta",
        "estoque": True
    },

    "torasemida": {
        "preco": "R$ 24,90 (10mg - 20 comprimidos)",
        "indicacao": "Retenção de líquidos em condições específicas",
        "estoque": False
    },

    "eplerenona": {
        "preco": "R$ 69,90 (25mg - 30 comprimidos)",
        "indicacao": "Algumas condições cardiovasculares",
        "estoque": True
    },

    "ivermectina_creme": {
        "preco": "R$ 29,90 (creme dermatológico)",
        "indicacao": "Tratamento tópico de condições específicas",
        "estoque": False
    },

    "permetrina": {
        "preco": "R$ 19,90 (loção 60mL)",
        "indicacao": "Tratamento de escabiose e pediculose",
        "estoque": True
    },

    "benzoato_benzila": {
        "preco": "R$ 17,90 (loção)",
        "indicacao": "Tratamento de escabiose e pediculose",
        "estoque": True
    },

    "enxofre_precipitado": {
        "preco": "R$ 15,90 (pomada)",
        "indicacao": "Tratamento tópico de condições específicas da pele",
        "estoque": False
    },

    "cetoconazol_shampoo": {
        "preco": "R$ 24,90 (shampoo 100mL)",
        "indicacao": "Caspa e algumas infecções fúngicas do couro cabeludo",
        "estoque": True
    },

    "selenio_sulfeto": {
        "preco": "R$ 19,90 (shampoo)",
        "indicacao": "Controle de caspa e dermatite seborreica",
        "estoque": True
    },

    "piritionato_zinco": {
        "preco": "R$ 21,90 (shampoo)",
        "indicacao": "Controle de caspa",
        "estoque": True
    },

    "minoxidil": {
        "preco": "R$ 39,90 (solução tópica)",
        "indicacao": "Tratamento tópico de queda de cabelo",
        "estoque": True
    },

    "espironolactona_100": {
        "preco": "R$ 29,90 (100mg - 30 comprimidos)",
        "indicacao": "Uso em condições específicas conforme orientação médica",
        "estoque": False
    },

    "dutasterida": {
        "preco": "R$ 49,90 (0,5mg - 30 cápsulas)",
        "indicacao": "Tratamento de sintomas relacionados à próstata",
        "estoque": True
    },

    "finasterida_dutasterida": {
        "preco": "R$ 59,90 (cápsulas)",
        "indicacao": "Tratamento específico da próstata",
        "estoque": False
    },

    "terazosina": {
        "preco": "R$ 29,90 (2mg - 30 comprimidos)",
        "indicacao": "Sintomas urinários relacionados à próstata",
        "estoque": True
    },

    "alfuzosina": {
        "preco": "R$ 39,90 (10mg - 30 comprimidos)",
        "indicacao": "Sintomas urinários relacionados à próstata",
        "estoque": True
    },

    "desmopressina": {
        "preco": "R$ 49,90 (comprimidos) - necessário receita",
        "indicacao": "Condições específicas relacionadas ao controle de líquidos",
        "estoque": False
    },

    "tolterodina": {
        "preco": "R$ 49,90 (2mg - 30 comprimidos)",
        "indicacao": "Sintomas de bexiga hiperativa",
        "estoque": True
    },

    "mirabegrona": {
        "preco": "R$ 69,90 (50mg - 30 comprimidos)",
        "indicacao": "Sintomas de bexiga hiperativa",
        "estoque": False
    },

    "fenazopiridina": {
        "preco": "R$ 17,90 (comprimidos)",
        "indicacao": "Alívio sintomático de desconforto urinário",
        "estoque": True
    },

    "fosfomicina": {
        "preco": "R$ 29,90 (sachê) - necessário receita",
        "indicacao": "Tratamento de algumas infecções urinárias",
        "estoque": True
    },

    "cefixima": {
        "preco": "R$ 39,90 (400mg - comprimidos) - necessário receita",
        "indicacao": "Tratamento de infecções bacterianas específicas",
        "estoque": False
    },

    "claritromicina": {
        "preco": "R$ 29,90 (500mg - 14 comprimidos) - necessário receita",
        "indicacao": "Tratamento de algumas infecções bacterianas",
        "estoque": True
    },

    "minociclina": {
        "preco": "R$ 39,90 (100mg - cápsulas) - necessário receita",
        "indicacao": "Tratamento de algumas infecções bacterianas",
        "estoque": False
    },

    "linezolida": {
        "preco": "R$ 149,90 (600mg - comprimidos) - necessário receita",
        "indicacao": "Tratamento de infecções bacterianas específicas",
        "estoque": False
    },

    "daptomicina": {
        "preco": "R$ 199,90 (frasco-ampola) - uso hospitalar",
        "indicacao": "Tratamento hospitalar de infecções bacterianas específicas",
        "estoque": False
    },

    "piperacilina_tazobactam": {
        "preco": "R$ 119,90 (frasco-ampola) - uso hospitalar",
        "indicacao": "Tratamento hospitalar de infecções bacterianas graves",
        "estoque": False
    },

    "clavulanato_potassio": {
        "preco": "R$ 39,90 (comprimidos) - necessário receita",
        "indicacao": "Componente de combinações antibióticas",
        "estoque": True
    },

    "ceftazidima": {
        "preco": "R$ 59,90 (frasco-ampola) - uso hospitalar",
        "indicacao": "Tratamento de infecções bacterianas específicas",
        "estoque": False
    },

    "cefepima": {
        "preco": "R$ 69,90 (frasco-ampola) - uso hospitalar",
        "indicacao": "Tratamento hospitalar de infecções bacterianas",
        "estoque": True
    },

    "meropenem_vaborbactam": {
        "preco": "R$ 299,90 (frasco-ampola) - uso hospitalar",
        "indicacao": "Tratamento hospitalar de infecções específicas",
        "estoque": False
    },

    "voriconazol": {
        "preco": "R$ 199,90 (comprimidos) - necessário receita",
        "indicacao": "Tratamento de infecções fúngicas específicas",
        "estoque": False
    },

    "posaconazol": {
        "preco": "R$ 249,90 (comprimidos) - necessário receita",
        "indicacao": "Tratamento e prevenção de infecções fúngicas específicas",
        "estoque": False
    },

    "terbinafina_comprimido": {
        "preco": "R$ 49,90 (250mg - 14 comprimidos)",
        "indicacao": "Tratamento de infecções fúngicas específicas",
        "estoque": True
    },

    "griseofulvina": {
        "preco": "R$ 39,90 (comprimidos)",
        "indicacao": "Tratamento de algumas micoses",
        "estoque": False
    },

    "fluocinolona": {
        "preco": "R$ 18,90 (creme dermatológico)",
        "indicacao": "Inflamações específicas da pele",
        "estoque": True
    },

    "clobetasol": {
        "preco": "R$ 24,90 (creme dermatológico)",
        "indicacao": "Inflamações específicas da pele",
        "estoque": False
    },

    "mometasona": {
        "preco": "R$ 22,90 (creme dermatológico)",
        "indicacao": "Inflamações e alergias da pele",
        "estoque": True
    },

    "desonida": {
        "preco": "R$ 19,90 (creme dermatológico)",
        "indicacao": "Inflamações leves da pele",
        "estoque": True
    },

    "tacrolimo_topico": {
        "preco": "R$ 49,90 (pomada) - necessário receita",
        "indicacao": "Tratamento tópico de condições dermatológicas específicas",
        "estoque": False
    },

    "pimecrolimo": {
        "preco": "R$ 59,90 (creme) - necessário receita",
        "indicacao": "Tratamento tópico de condições dermatológicas específicas",
        "estoque": True
    },

    "metronidazol_creme": {
        "preco": "R$ 19,90 (creme dermatológico)",
        "indicacao": "Tratamento de algumas condições dermatológicas",
        "estoque": True
    },

    "peroxido_ureia": {
        "preco": "R$ 21,90 (creme)",
        "indicacao": "Hidratação e cuidado de pele ressecada",
        "estoque": False
    },

    "vaselina": {
        "preco": "R$ 9,90 (pomada)",
        "indicacao": "Proteção e hidratação da pele",
        "estoque": True
    },

    "lanolina": {
        "preco": "R$ 19,90 (pomada)",
        "indicacao": "Hidratação e proteção da pele",
        "estoque": True
    },

    "oxido_zinco": {
        "preco": "R$ 12,90 (pomada)",
        "indicacao": "Proteção e cuidado da pele",
        "estoque": True
    },

    "calendula": {
        "preco": "R$ 16,90 (pomada)",
        "indicacao": "Hidratação e cuidado da pele",
        "estoque": False
    },

    "dexpantenol_creme": {
        "preco": "R$ 18,90 (creme)",
        "indicacao": "Hidratação e proteção da pele",
        "estoque": True
    },

    "carboximetilcelulose_gel": {
        "preco": "R$ 29,90 (gel oftálmico)",
        "indicacao": "Lubrificação ocular",
        "estoque": True
    },

    "hialuronato_sodio_colirio": {
        "preco": "R$ 39,90 (colírio)",
        "indicacao": "Lubrificação ocular",
        "estoque": False
    },

    "olopatadina": {
        "preco": "R$ 34,90 (colírio) - necessário receita",
        "indicacao": "Alergia ocular",
        "estoque": True
    },

    "cetotifeno": {
        "preco": "R$ 29,90 (colírio)",
        "indicacao": "Alergia ocular",
        "estoque": True
    },

    "nafazolina": {
        "preco": "R$ 11,90 (solução nasal)",
        "indicacao": "Congestão nasal",
        "estoque": False
    },

    "ipratropio": {
        "preco": "R$ 29,90 (spray nasal)",
        "indicacao": "Sintomas respiratórios específicos",
        "estoque": True
    },

    "tiotropio": {
        "preco": "R$ 89,90 (inalador) - necessário receita",
        "indicacao": "Controle de doenças respiratórias específicas",
        "estoque": False
    },

    "salmeterol": {
        "preco": "R$ 59,90 (inalador) - necessário receita",
        "indicacao": "Controle de doenças respiratórias específicas",
        "estoque": True
    },

    "fluticasona": {
        "preco": "R$ 49,90 (spray nasal)",
        "indicacao": "Tratamento de rinite e inflamação nasal",
        "estoque": True
    },

    "fluticasona_inalatoria": {
        "preco": "R$ 69,90 (inalador) - necessário receita",
        "indicacao": "Controle de doenças respiratórias específicas",
        "estoque": False
    },

    "beclometasona_inalatoria": {
        "preco": "R$ 39,90 (inalador)",
        "indicacao": "Controle de inflamação respiratória",
        "estoque": True
    },

    "budesonida_formoterol": {
        "preco": "R$ 89,90 (inalador) - necessário receita",
        "indicacao": "Controle de doenças respiratórias específicas",
        "estoque": True
    },

    "ipratropio_salbutamol": {
        "preco": "R$ 49,90 (solução inalatória) - necessário receita",
        "indicacao": "Controle de broncoespasmo",
        "estoque": False
    },

    "teofilina": {
        "preco": "R$ 24,90 (200mg - 30 comprimidos)",
        "indicacao": "Tratamento respiratório específico",
        "estoque": True
    },

    "montelucaste_4mg": {
        "preco": "R$ 24,90 (4mg - 30 comprimidos)",
        "indicacao": "Controle de asma e rinite",
        "estoque": True
    },

    "montelucaste_5mg": {
        "preco": "R$ 27,90 (5mg - 30 comprimidos)",
        "indicacao": "Controle de asma e rinite",
        "estoque": False
    },

    "salbutamol_xarope": {
        "preco": "R$ 17,90 (xarope)",
        "indicacao": "Alívio de broncoespasmo em situações específicas",
        "estoque": True
    },

    "prednisona_5mg": {
        "preco": "R$ 9,90 (5mg - 20 comprimidos) - necessário receita",
        "indicacao": "Processos inflamatórios e alérgicos",
        "estoque": True
    },

    "prednisona_20mg": {
        "preco": "R$ 14,90 (20mg - 20 comprimidos) - necessário receita",
        "indicacao": "Processos inflamatórios e alérgicos",
        "estoque": False
    },

    "prednisolona_3mg_ml": {
        "preco": "R$ 22,90 (solução oral) - necessário receita",
        "indicacao": "Processos inflamatórios e alérgicos",
        "estoque": True
    },

    "hidroxicloroquina_200": {
        "preco": "R$ 29,90 (200mg - 30 comprimidos) - necessário receita",
        "indicacao": "Tratamento de doenças específicas",
        "estoque": False
    },

    "sulfato_hidroxicloroquina": {
        "preco": "R$ 39,90 (400mg - 30 comprimidos) - necessário receita",
        "indicacao": "Tratamento de doenças autoimunes específicas",
        "estoque": True
    },

    "ciclosporina": {
        "preco": "R$ 149,90 (cápsulas) - necessário receita",
        "indicacao": "Imunossupressão em condições específicas",
        "estoque": False
    },

    "micofenolato": {
        "preco": "R$ 99,90 (500mg - 50 comprimidos) - necessário receita",
        "indicacao": "Imunossupressão em condições específicas",
        "estoque": True
    },

    "prednisolona_5": {
        "preco": "R$ 12,90 (5mg - comprimidos) - necessário receita",
        "indicacao": "Processos inflamatórios específicos",
        "estoque": True
    },

    "adalimumabe": {
        "preco": "R$ 499,90 (seringa) - necessário receita",
        "indicacao": "Tratamento de doenças inflamatórias específicas",
        "estoque": False
    },

    "etanercepte": {
        "preco": "R$ 599,90 (seringa) - necessário receita",
        "indicacao": "Tratamento de doenças inflamatórias específicas",
        "estoque": False
    },

    "infliximabe": {
        "preco": "R$ 799,90 (frasco) - uso hospitalar",
        "indicacao": "Tratamento de doenças inflamatórias específicas",
        "estoque": False
    },

    "omeprazol_suspensao": {
        "preco": "R$ 19,90 (suspensão oral)",
        "indicacao": "Redução da acidez gástrica",
        "estoque": True
    },

    "pantoprazol_20": {
        "preco": "R$ 16,90 (20mg - 28 comprimidos)",
        "indicacao": "Azia e refluxo",
        "estoque": True
    },

    "esomeprazol_20": {
        "preco": "R$ 24,90 (20mg - 28 cápsulas)",
        "indicacao": "Azia e refluxo",
        "estoque": False
    },

    "lansoprazol_30": {
        "preco": "R$ 22,90 (30mg - 28 cápsulas)",
        "indicacao": "Azia e refluxo",
        "estoque": True
    },

    "rabeprazol": {
        "preco": "R$ 29,90 (20mg - 28 comprimidos)",
        "indicacao": "Refluxo e redução da acidez gástrica",
        "estoque": True
    },

    "misoprostol": {
        "preco": "Uso hospitalar e controlado - necessário receita especial",
        "indicacao": "Uso médico específico",
        "estoque": False
    },

    "mesalazina_800": {
        "preco": "R$ 89,90 (800mg - 30 comprimidos) - necessário receita",
        "indicacao": "Doenças inflamatórias intestinais",
        "estoque": True
    },

    "balsalazida": {
        "preco": "R$ 99,90 (cápsulas) - necessário receita",
        "indicacao": "Doenças inflamatórias intestinais específicas",
        "estoque": False
    },

    "sulfassalazina_500": {
        "preco": "R$ 39,90 (500mg - 30 comprimidos) - necessário receita",
        "indicacao": "Doenças inflamatórias específicas",
        "estoque": True
    },

    "domperidona_suspensao": {
        "preco": "R$ 19,90 (suspensão oral)",
        "indicacao": "Náuseas e distúrbios gastrointestinais",
        "estoque": True
    },

    "metoclopramida_gotas": {
        "preco": "R$ 12,90 (gotas)",
        "indicacao": "Náuseas e vômitos",
        "estoque": True
    },

    "ondansetrona_odt": {
        "preco": "R$ 29,90 (8mg - comprimidos orodispersíveis)",
        "indicacao": "Náuseas e vômitos",
        "estoque": False
    },

    "racecadotrila": {
        "preco": "R$ 24,90 (100mg - cápsulas)",
        "indicacao": "Tratamento sintomático da diarreia aguda",
        "estoque": True
    },

    "mesalazina_1200": {
        "preco": "R$ 119,90 (1200mg - comprimidos) - necessário receita",
        "indicacao": "Doenças inflamatórias intestinais",
        "estoque": False
    },

    "sulfato_atropina": {
        "preco": "R$ 19,90 (ampola) - uso profissional",
        "indicacao": "Uso médico em situações específicas",
        "estoque": True
    },

    "adrenalina": {
        "preco": "R$ 19,90 (ampola) - uso profissional",
        "indicacao": "Tratamento emergencial de reações graves",
        "estoque": True
    },

    "noradrenalina": {
        "preco": "R$ 29,90 (ampola) - uso hospitalar",
        "indicacao": "Uso hospitalar em situações de emergência",
        "estoque": False
    },

    "dopamina": {
        "preco": "R$ 29,90 (ampola) - uso hospitalar",
        "indicacao": "Uso hospitalar em condições cardiovasculares específicas",
        "estoque": True
    },

    "dobutamina": {
        "preco": "R$ 39,90 (ampola) - uso hospitalar",
        "indicacao": "Uso hospitalar em condições cardíacas específicas",
        "estoque": False
    },

    "naloxona": {
        "preco": "R$ 49,90 (ampola) - uso profissional",
        "indicacao": "Reversão de efeitos de opioides em situações específicas",
        "estoque": True
    },

    "flumazenil": {
        "preco": "R$ 59,90 (ampola) - uso hospitalar",
        "indicacao": "Reversão de efeitos de benzodiazepínicos em situações específicas",
        "estoque": False
    },

    "heparina": {
        "preco": "R$ 39,90 (frasco) - necessário receita",
        "indicacao": "Anticoagulação em situações específicas",
        "estoque": True
    },

    "enoxaparina": {
        "preco": "R$ 89,90 (seringa) - necessário receita",
        "indicacao": "Prevenção e tratamento de eventos trombóticos",
        "estoque": False
    },

    "protamina": {
        "preco": "R$ 69,90 (ampola) - uso hospitalar",
        "indicacao": "Reversão de efeito da heparina em situações específicas",
        "estoque": False
    },

    "fitomenadiona": {
        "preco": "R$ 39,90 (ampola) - uso profissional",
        "indicacao": "Reposição de vitamina K em situações específicas",
        "estoque": True
    },

    "albumina_humana": {
        "preco": "R$ 299,90 (frasco) - uso hospitalar",
        "indicacao": "Reposição de albumina em situações específicas",
        "estoque": False
    },

    "imunoglobulina_humana": {
        "preco": "R$ 499,90 (frasco) - uso hospitalar",
        "indicacao": "Tratamentos imunológicos específicos",
        "estoque": False
    },

    "hidroxido_aluminio": {
        "preco": "R$ 13,90 (suspensão oral)",
        "indicacao": "Azia e excesso de acidez",
        "estoque": True
    },

    "carbonato_calcio": {
        "preco": "R$ 18,90 (500mg - 60 comprimidos)",
        "indicacao": "Suplementação de cálcio",
        "estoque": True
    },

    "simeticona_125": {
        "preco": "R$ 13,90 (125mg - cápsulas)",
        "indicacao": "Alívio de gases",
        "estoque": True
    },

    "sacarose_ferro": {
        "preco": "R$ 49,90 (solução) - uso profissional",
        "indicacao": "Reposição de ferro em situações específicas",
        "estoque": False
    },

    "hidroxocobalamina": {
        "preco": "R$ 29,90 (ampola) - uso profissional",
        "indicacao": "Reposição de vitamina B12",
        "estoque": True
    },

    "eritropoetina": {
        "preco": "R$ 299,90 (seringa) - necessário receita",
        "indicacao": "Tratamento de anemias específicas",
        "estoque": False
    },

    "filgrastim": {
        "preco": "R$ 399,90 (seringa) - necessário receita",
        "indicacao": "Tratamentos hematológicos específicos",
        "estoque": False
    },

    "tamoxifeno": {
        "preco": "R$ 39,90 (20mg - 30 comprimidos) - necessário receita",
        "indicacao": "Tratamento hormonal específico",
        "estoque": True
    },

    "anastrozol": {
        "preco": "R$ 49,90 (1mg - 30 comprimidos) - necessário receita",
        "indicacao": "Tratamento hormonal específico",
        "estoque": False
    },

    "letrozol": {
        "preco": "R$ 59,90 (2,5mg - 30 comprimidos) - necessário receita",
        "indicacao": "Tratamento hormonal específico",
        "estoque": True
    },

    "goserrelina": {
        "preco": "R$ 399,90 (implante) - necessário receita",
        "indicacao": "Tratamento hormonal específico",
        "estoque": False
    },

    "leuprorelina": {
        "preco": "R$ 499,90 (injeção) - necessário receita",
        "indicacao": "Tratamento hormonal específico",
        "estoque": False
    },

    "sildenafil": {
        "preco": "R$ 39,90 (50mg - 4 comprimidos) - necessário receita",
        "indicacao": "Tratamento de disfunção erétil",
        "estoque": True
    },

    "tadalafil": {
        "preco": "R$ 49,90 (20mg - 4 comprimidos) - necessário receita",
        "indicacao": "Tratamento de disfunção erétil",
        "estoque": True
    },

    "vardenafil": {
        "preco": "R$ 59,90 (20mg - 4 comprimidos) - necessário receita",
        "indicacao": "Tratamento de disfunção erétil",
        "estoque": False
    },

    "finasterida_1mg": {
        "preco": "R$ 29,90 (1mg - 30 comprimidos)",
        "indicacao": "Tratamento da queda de cabelo masculina",
        "estoque": True
    },

    "dutasterida_tansulosina": {
        "preco": "R$ 69,90 (cápsulas) - necessário receita",
        "indicacao": "Tratamento de sintomas relacionados à próstata",
        "estoque": False
    },

    "clomipramina": {
        "preco": "R$ 29,90 (25mg - 30 comprimidos) - necessário receita",
        "indicacao": "Tratamento de transtornos psiquiátricos específicos",
        "estoque": True
    },

    "imipramina": {
        "preco": "R$ 24,90 (25mg - 30 comprimidos) - necessário receita",
        "indicacao": "Tratamento de transtornos específicos",
        "estoque": False
    },

    "mirtazapina": {
        "preco": "R$ 39,90 (30mg - 30 comprimidos) - necessário receita",
        "indicacao": "Tratamento de depressão",
        "estoque": True
    },

    "agomelatina": {
        "preco": "R$ 69,90 (25mg - 30 comprimidos) - necessário receita",
        "indicacao": "Tratamento de depressão",
        "estoque": False
    },

    "bupropiona": {
        "preco": "R$ 39,90 (150mg - 30 comprimidos) - necessário receita",
        "indicacao": "Tratamento de depressão e outras indicações específicas",
        "estoque": True
    },

    "lisdexanfetamina": {
        "preco": "Medicamento controlado - necessário receita especial",
        "indicacao": "Tratamento de condições específicas sob acompanhamento médico",
        "estoque": False
    },

    "metilfenidato": {
        "preco": "Medicamento controlado - necessário receita especial",
        "indicacao": "Tratamento de TDAH e outras condições específicas",
        "estoque": True
    },

    "clorpromazina": {
        "preco": "R$ 24,90 (100mg - 20 comprimidos) - necessário receita",
        "indicacao": "Transtornos psiquiátricos específicos",
        "estoque": False
    },

    "levomepromazina": {
        "preco": "R$ 29,90 (100mg - 20 comprimidos) - necessário receita",
        "indicacao": "Transtornos psiquiátricos específicos",
        "estoque": True
    },

    "aripiprazol": {
        "preco": "R$ 69,90 (10mg - 30 comprimidos) - necessário receita",
        "indicacao": "Transtornos psiquiátricos específicos",
        "estoque": False
    },

    "ziprasidona": {
        "preco": "R$ 59,90 (40mg - 30 cápsulas) - necessário receita",
        "indicacao": "Transtornos psiquiátricos específicos",
        "estoque": True
    },

    "lacosamida": {
        "preco": "R$ 89,90 (100mg - 30 comprimidos) - necessário receita",
        "indicacao": "Tratamento de epilepsia",
        "estoque": False
    },

    "clonazepam_gotas": {
        "preco": "Medicamento controlado - necessário receita especial",
        "indicacao": "Algumas condições neurológicas e de ansiedade",
        "estoque": True
    },

    "diazepam_gotas": {
        "preco": "Medicamento controlado - necessário receita especial",
        "indicacao": "Algumas condições de ansiedade e espasmos",
        "estoque": False
    },

    "fenobarbital": {
        "preco": "Medicamento controlado - necessário receita especial",
        "indicacao": "Tratamento de epilepsia em situações específicas",
        "estoque": True
    },

    "fenitoina": {
        "preco": "R$ 29,90 (100mg - 30 comprimidos) - necessário receita",
        "indicacao": "Tratamento de epilepsia",
        "estoque": False
    },

    "clonidina": {
        "preco": "R$ 16,90 (0,1mg - 30 comprimidos)",
        "indicacao": "Pressão alta e outras condições específicas",
        "estoque": True
    },

    "metildopa": {
        "preco": "R$ 18,90 (250mg - 30 comprimidos)",
        "indicacao": "Tratamento da pressão alta",
        "estoque": True
    },

    "hidralazina": {
        "preco": "R$ 19,90 (25mg - 30 comprimidos)",
        "indicacao": "Tratamento da pressão alta em situações específicas",
        "estoque": False
    },

    "minoxidil_oral": {
        "preco": "R$ 24,90 (comprimidos) - necessário receita",
        "indicacao": "Tratamento de condições específicas sob acompanhamento médico",
        "estoque": True
    },

    "reserpina": {
        "preco": "R$ 19,90 (comprimidos)",
        "indicacao": "Tratamento cardiovascular específico",
        "estoque": False
    },

    "prazosina": {
        "preco": "R$ 29,90 (1mg - 30 comprimidos)",
        "indicacao": "Pressão alta e condições específicas",
        "estoque": True
    },

    "doxazosina_4mg": {
        "preco": "R$ 29,90 (4mg - 30 comprimidos)",
        "indicacao": "Pressão alta e sintomas urinários",
        "estoque": True
    },

    "amlodipino_10mg": {
        "preco": "R$ 15,90 (10mg - 30 comprimidos)",
        "indicacao": "Pressão alta e angina",
        "estoque": True
    },

    "losartana_100mg": {
        "preco": "R$ 21,90 (100mg - 30 comprimidos)",
        "indicacao": "Pressão alta",
        "estoque": True
    },

    "valsartana_160mg": {
        "preco": "R$ 29,90 (160mg - 30 comprimidos)",
        "indicacao": "Pressão alta",
        "estoque": False
    },

    "telmisartana_80mg": {
        "preco": "R$ 34,90 (80mg - 30 comprimidos)",
        "indicacao": "Pressão alta",
        "estoque": True
    },

    "candesartana_16mg": {
        "preco": "R$ 29,90 (16mg - 30 comprimidos)",
        "indicacao": "Pressão alta",
        "estoque": True
    },

    "irbesartana_300mg": {
        "preco": "R$ 29,90 (300mg - 30 comprimidos)",
        "indicacao": "Pressão alta",
        "estoque": False
    },

    "enalapril_20mg": {
        "preco": "R$ 16,90 (20mg - 30 comprimidos)",
        "indicacao": "Pressão alta e insuficiência cardíaca",
        "estoque": True
    },

    "captopril_50mg": {
        "preco": "R$ 13,90 (50mg - 30 comprimidos)",
        "indicacao": "Pressão alta e insuficiência cardíaca",
        "estoque": True
    },

    "carvedilol_25mg": {
        "preco": "R$ 29,90 (25mg - 30 comprimidos)",
        "indicacao": "Insuficiência cardíaca e pressão alta",
        "estoque": False
    },

    "metoprolol_100mg": {
        "preco": "R$ 24,90 (100mg - 30 comprimidos)",
        "indicacao": "Controle da pressão e frequência cardíaca",
        "estoque": True
    },

    "atenolol_100mg": {
        "preco": "R$ 14,90 (100mg - 30 comprimidos)",
        "indicacao": "Pressão alta e controle cardíaco",
        "estoque": True
    },

    "furosemida_40mg": {
        "preco": "R$ 9,90 (40mg - 30 comprimidos)",
        "indicacao": "Retenção de líquidos e edema",
        "estoque": True
    },

    "espironolactona_25mg": {
        "preco": "R$ 16,90 (25mg - 30 comprimidos)",
        "indicacao": "Retenção de líquidos e condições cardíacas específicas",
        "estoque": True
    },

    "hidroclorotiazida_25mg": {
        "preco": "R$ 9,90 (25mg - 30 comprimidos)",
        "indicacao": "Pressão alta",
        "estoque": True
    },

    "clortalidona_25mg": {
        "preco": "R$ 17,90 (25mg - 30 comprimidos)",
        "indicacao": "Pressão alta",
        "estoque": False
    },

    "indapamida_15mg": {
        "preco": "R$ 18,90 (1,5mg - 30 comprimidos)",
        "indicacao": "Pressão alta",
        "estoque": True
    },

    "torasemida_10mg": {
        "preco": "R$ 26,90 (10mg - 20 comprimidos)",
        "indicacao": "Retenção de líquidos em condições específicas",
        "estoque": False
    },

    "manitol_solucao": {
        "preco": "R$ 29,90 (solução intravenosa) - uso hospitalar",
        "indicacao": "Uso hospitalar em condições específicas",
        "estoque": False
    },

    "acetazolamida": {
        "preco": "R$ 29,90 (250mg - 20 comprimidos)",
        "indicacao": "Tratamento de condições específicas",
        "estoque": True
    },

    "betaxolol": {
        "preco": "R$ 39,90 (colírio) - necessário receita",
        "indicacao": "Tratamento de glaucoma",
        "estoque": False
    },

    "travoprosta": {
        "preco": "R$ 49,90 (colírio) - necessário receita",
        "indicacao": "Tratamento de glaucoma",
        "estoque": True
    },

    "bimatoprosta": {
        "preco": "R$ 59,90 (colírio) - necessário receita",
        "indicacao": "Tratamento de glaucoma",
        "estoque": False
    },

    "brinzolamida": {
        "preco": "R$ 49,90 (colírio) - necessário receita",
        "indicacao": "Tratamento de glaucoma",
        "estoque": True
    },

    "pilocarpina": {
        "preco": "R$ 29,90 (colírio) - necessário receita",
        "indicacao": "Tratamento oftalmológico específico",
        "estoque": False
    },

    "atropina_colirio": {
        "preco": "R$ 19,90 (colírio) - necessário receita",
        "indicacao": "Uso oftalmológico específico",
        "estoque": True
    },

    "tropicamida": {
        "preco": "R$ 24,90 (colírio) - uso profissional",
        "indicacao": "Dilatação da pupila para exames oftalmológicos",
        "estoque": True
    },

    "fenilefrina_colirio": {
        "preco": "R$ 19,90 (colírio) - uso profissional",
        "indicacao": "Uso oftalmológico específico",
        "estoque": False
    },

    "diclofenaco_colirio": {
        "preco": "R$ 24,90 (colírio) - necessário receita",
        "indicacao": "Inflamação ocular específica",
        "estoque": True
    },

    "prednisolona_colirio": {
        "preco": "R$ 29,90 (colírio) - necessário receita",
        "indicacao": "Inflamação ocular específica",
        "estoque": False
    },

    "dexametasona_colirio": {
        "preco": "R$ 29,90 (colírio) - necessário receita",
        "indicacao": "Inflamação ocular específica",
        "estoque": True
    },

    "ciprofloxacino_colirio": {
        "preco": "R$ 24,90 (colírio) - necessário receita",
        "indicacao": "Infecções bacterianas oculares",
        "estoque": True
    },

    "ofloxacino_colirio": {
        "preco": "R$ 29,90 (colírio) - necessário receita",
        "indicacao": "Infecções bacterianas oculares",
        "estoque": False
    },

    "moxifloxacino": {
        "preco": "R$ 39,90 (400mg - comprimidos) - necessário receita",
        "indicacao": "Infecções bacterianas específicas",
        "estoque": True
    },

    "levofloxacino_750": {
        "preco": "R$ 34,90 (750mg - 5 comprimidos) - necessário receita",
        "indicacao": "Infecções bacterianas específicas",
        "estoque": False
    },

    "azitromicina_500": {
        "preco": "R$ 24,90 (500mg - 5 comprimidos) - necessário receita",
        "indicacao": "Infecções bacterianas específicas",
        "estoque": True
    },

    "claritromicina_500": {
        "preco": "R$ 29,90 (500mg - 14 comprimidos) - necessário receita",
        "indicacao": "Infecções bacterianas específicas",
        "estoque": True
    },

    "amoxicilina_875": {
        "preco": "R$ 29,90 (875mg - 14 comprimidos) - necessário receita",
        "indicacao": "Infecções bacterianas",
        "estoque": True
    },

    "amoxicilina_500": {
        "preco": "R$ 26,90 (500mg - 21 cápsulas) - necessário receita",
        "indicacao": "Infecções bacterianas",
        "estoque": True
    },

    "cefalexina_500": {
        "preco": "R$ 27,90 (500mg - 28 cápsulas) - necessário receita",
        "indicacao": "Infecções bacterianas",
        "estoque": False
    },

    "cefixima_400": {
        "preco": "R$ 39,90 (400mg - 10 comprimidos) - necessário receita",
        "indicacao": "Infecções bacterianas específicas",
        "estoque": True
    },

    "cefuroxima_500": {
        "preco": "R$ 44,90 (500mg - 14 comprimidos) - necessário receita",
        "indicacao": "Infecções bacterianas específicas",
        "estoque": False
    },

    "doxiciclina_100": {
        "preco": "R$ 21,90 (100mg - 15 cápsulas) - necessário receita",
        "indicacao": "Infecções bacterianas específicas",
        "estoque": True
    },

    "metronidazol_400": {
        "preco": "R$ 16,90 (400mg - comprimidos) - necessário receita",
        "indicacao": "Infecções bacterianas e parasitárias",
        "estoque": True
    },

    "tinidazol_500": {
        "preco": "R$ 22,90 (500mg - comprimidos) - necessário receita",
        "indicacao": "Infecções bacterianas e parasitárias",
        "estoque": False
    },

    "fluconazol_150": {
        "preco": "R$ 14,90 (150mg - 1 cápsula)",
        "indicacao": "Infecções fúngicas",
        "estoque": True
    },

    "itraconazol_100": {
        "preco": "R$ 39,90 (100mg - 15 cápsulas)",
        "indicacao": "Infecções fúngicas",
        "estoque": False
    },

    "terbinafina_250": {
        "preco": "R$ 49,90 (250mg - 14 comprimidos)",
        "indicacao": "Infecções fúngicas específicas",
        "estoque": True
    },

    "aciclovir_200": {
        "preco": "R$ 24,90 (200mg - 25 comprimidos)",
        "indicacao": "Infecções por herpes",
        "estoque": True
    },

    "valaciclovir_500": {
        "preco": "R$ 39,90 (500mg - 10 comprimidos)",
        "indicacao": "Infecções por herpes",
        "estoque": False
    },

    "oseltamivir_75mg": {
        "preco": "R$ 79,90 (75mg - 10 cápsulas) - necessário receita",
        "indicacao": "Tratamento da influenza conforme orientação médica",
        "estoque": True
    },

    "soro_fisiologico_500": {
        "preco": "R$ 12,90 (500mL)",
        "indicacao": "Higienização e hidratação",
        "estoque": True
    },

    "agua_oxigenada": {
        "preco": "R$ 8,90 (solução 10 volumes)",
        "indicacao": "Uso externo específico",
        "estoque": True
    },

    "solucao_antisseptica": {
        "preco": "R$ 14,90 (250mL)",
        "indicacao": "Higienização e antissepsia",
        "estoque": True
    },

    "glicerina": {
        "preco": "R$ 9,90 (solução)",
        "indicacao": "Uso específico conforme orientação",
        "estoque": False
    },

    "oleo_mineral": {
        "preco": "R$ 16,90 (frasco)",
        "indicacao": "Uso específico conforme orientação",
        "estoque": True
    },

    "vaselina_liquida": {
        "preco": "R$ 13,90 (frasco)",
        "indicacao": "Uso tópico e hidratação",
        "estoque": True
    },

    "pomada_assaduras": {
        "preco": "R$ 18,90 (pomada)",
        "indicacao": "Proteção da pele contra assaduras",
        "estoque": True
    },

    "protetor_labial": {
        "preco": "R$ 12,90 (bastão)",
        "indicacao": "Proteção e hidratação dos lábios",
        "estoque": True
    },

    "gel_aloe_vera": {
        "preco": "R$ 19,90 (gel tópico)",
        "indicacao": "Hidratação e cuidado da pele",
        "estoque": False
    },

    "creme_hidratante": {
        "preco": "R$ 24,90 (creme 200g)",
        "indicacao": "Hidratação da pele",
        "estoque": True
    },

    "protetor_solar": {
        "preco": "R$ 49,90 (FPS 50 - 120g)",
        "indicacao": "Proteção da pele contra radiação solar",
        "estoque": True
    },

    "repelente": {
        "preco": "R$ 24,90 (loção 100mL)",
        "indicacao": "Proteção contra picadas de insetos",
        "estoque": True
    },

    "gel_antisseptico": {
        "preco": "R$ 9,90 (álcool em gel 70%)",
        "indicacao": "Higienização das mãos",
        "estoque": True
    },

    "algodao": {
        "preco": "R$ 8,90 (pacote)",
        "indicacao": "Material para higiene e cuidados",
        "estoque": True
    },

    "gaze_estéril": {
        "preco": "R$ 12,90 (pacote)",
        "indicacao": "Curativos e cuidados de feridas",
        "estoque": True
    },

    "atadura": {
        "preco": "R$ 9,90 (unidade)",
        "indicacao": "Curativos e suporte",
        "estoque": True
    },

    "esparadrapo": {
        "preco": "R$ 8,90 (rolo)",
        "indicacao": "Fixação de curativos",
        "estoque": True
    },

    "fita_microporosa": {
        "preco": "R$ 9,90 (rolo)",
        "indicacao": "Fixação de curativos",
        "estoque": False
    },

    "termometro_digital": {
        "preco": "R$ 29,90 (unidade)",
        "indicacao": "Medição da temperatura corporal",
        "estoque": True
    },

    "soro_reidratacao_oral": {
        "preco": "R$ 5,90 (envelope)",
        "indicacao": "Reposição de líquidos e eletrólitos",
        "estoque": True
    },

    "carvao_vegetal": {
        "preco": "R$ 14,90 (cápsulas)",
        "indicacao": "Uso específico conforme orientação profissional",
        "estoque": False
    },

    "loperamida_2mg": {
        "preco": "R$ 10,90 (2mg - 12 cápsulas)",
        "indicacao": "Diarreia aguda",
        "estoque": True
    },

    "racecadotrila_100": {
        "preco": "R$ 24,90 (100mg - 9 cápsulas)",
        "indicacao": "Tratamento sintomático da diarreia aguda",
        "estoque": True
    },

    "lactulose_667": {
        "preco": "R$ 18,90 (667mg/mL - 120mL)",
        "indicacao": "Constipação intestinal",
        "estoque": True
    },

    "macrogol": {
        "preco": "R$ 29,90 (sachês)",
        "indicacao": "Constipação intestinal",
        "estoque": False
    },

    "senosideos": {
        "preco": "R$ 11,90 (comprimidos)",
        "indicacao": "Constipação ocasional",
        "estoque": True
    },

    "psyllium_capsulas": {
        "preco": "R$ 29,90 (cápsulas)",
        "indicacao": "Fonte de fibras e auxílio intestinal",
        "estoque": True
    },

    "bromoprida": {
        "preco": "R$ 13,90 (10mg - 20 cápsulas)",
        "indicacao": "Náuseas e distúrbios gastrointestinais",
        "estoque": True
    },

    "dimenidrinato_b6": {
        "preco": "R$ 16,90 (comprimidos)",
        "indicacao": "Náuseas e enjoo",
        "estoque": True
    },

    "meclizina_25": {
        "preco": "R$ 18,90 (25mg - 10 comprimidos)",
        "indicacao": "Tontura e náuseas",
        "estoque": False
    },

    "cinarizina": {
        "preco": "R$ 19,90 (75mg - 30 comprimidos)",
        "indicacao": "Tontura e distúrbios vestibulares",
        "estoque": True
    },

    "flunarizina": {
        "preco": "R$ 24,90 (10mg - 30 comprimidos)",
        "indicacao": "Prevenção de enxaqueca e condições vestibulares",
        "estoque": True
    },

    "betahistina": {
        "preco": "R$ 29,90 (24mg - 30 comprimidos)",
        "indicacao": "Tratamento de vertigem em condições específicas",
        "estoque": False
    },

    "piracetam": {
        "preco": "R$ 39,90 (800mg - 30 comprimidos)",
        "indicacao": "Uso em condições neurológicas específicas",
        "estoque": False
    },

    "citicolina": {
        "preco": "R$ 49,90 (500mg - 20 comprimidos)",
        "indicacao": "Uso em condições neurológicas específicas",
        "estoque": True
    },

    "donepezila_5mg": {
        "preco": "R$ 39,90 (5mg - 30 comprimidos) - necessário receita",
        "indicacao": "Tratamento sintomático da doença de Alzheimer",
        "estoque": True
    },

    "memantina_10mg": {
        "preco": "R$ 59,90 (10mg - 30 comprimidos) - necessário receita",
        "indicacao": "Tratamento sintomático da doença de Alzheimer",
        "estoque": False
    },

    "rivastigmina": {
        "preco": "R$ 79,90 (cápsulas) - necessário receita",
        "indicacao": "Tratamento sintomático de demências específicas",
        "estoque": True
    },

    "galantamina": {
        "preco": "R$ 69,90 (cápsulas) - necessário receita",
        "indicacao": "Tratamento sintomático de demências específicas",
        "estoque": False
    },

    "selegilina": {
        "preco": "R$ 39,90 (5mg - 20 comprimidos) - necessário receita",
        "indicacao": "Tratamento de condições neurológicas específicas",
        "estoque": True
    },

    "amantadina": {
        "preco": "R$ 29,90 (100mg - 20 cápsulas) - necessário receita",
        "indicacao": "Tratamento de condições neurológicas específicas",
        "estoque": False
    },

    "entacapona": {
        "preco": "R$ 69,90 (200mg - 30 comprimidos) - necessário receita",
        "indicacao": "Tratamento complementar da doença de Parkinson",
        "estoque": True
    },

    "ropinirol": {
        "preco": "R$ 49,90 (2mg - 30 comprimidos) - necessário receita",
        "indicacao": "Tratamento de doença de Parkinson e condições específicas",
        "estoque": False
    },

    "rotigotina": {
        "preco": "R$ 149,90 (adesivos) - necessário receita",
        "indicacao": "Tratamento de doença de Parkinson",
        "estoque": False
    },

    "apomorfina": {
        "preco": "R$ 199,90 (injeção) - necessário receita",
        "indicacao": "Tratamento especializado da doença de Parkinson",
        "estoque": False
    },

    "baclofeno_10mg": {
        "preco": "R$ 24,90 (10mg - 20 comprimidos) - necessário receita",
        "indicacao": "Tratamento de espasticidade muscular",
        "estoque": True
    },

    "gabapentina_600mg": {
        "preco": "R$ 49,90 (600mg - 30 comprimidos) - necessário receita",
        "indicacao": "Algumas condições neurológicas e dores",
        "estoque": False
    },

    "pregabalina_150mg": {
        "preco": "R$ 59,90 (150mg - 30 cápsulas) - necessário receita",
        "indicacao": "Algumas dores neuropáticas e condições específicas",
        "estoque": True
    },

    "carbamazepina_400mg": {
        "preco": "R$ 29,90 (400mg - 20 comprimidos) - necessário receita",
        "indicacao": "Epilepsia e condições neurológicas específicas",
        "estoque": True
    },

    "lamotrigina_100mg": {
        "preco": "R$ 39,90 (100mg - 30 comprimidos) - necessário receita",
        "indicacao": "Epilepsia e condições específicas",
        "estoque": False
    },

    "topiramato_100mg": {
        "preco": "R$ 39,90 (100mg - 30 comprimidos) - necessário receita",
        "indicacao": "Epilepsia e prevenção de enxaqueca",
        "estoque": True
    },

    "levetiracetam_1000mg": {
        "preco": "R$ 69,90 (1000mg - 30 comprimidos) - necessário receita",
        "indicacao": "Tratamento de epilepsia",
        "estoque": False
    },

    "valproato_500mg": {
        "preco": "R$ 29,90 (500mg - 30 comprimidos) - necessário receita",
        "indicacao": "Tratamento de epilepsia e condições específicas",
        "estoque": True
    },

    "fenitoina_100mg": {
        "preco": "R$ 29,90 (100mg - 30 comprimidos) - necessário receita",
        "indicacao": "Tratamento de epilepsia",
        "estoque": False
    },

    "fenobarbital_100mg": {
        "preco": "Medicamento controlado - necessário receita especial",
        "indicacao": "Tratamento de epilepsia",
        "estoque": True
    },

    "clobazam": {
        "preco": "Medicamento controlado - necessário receita especial",
        "indicacao": "Tratamento de epilepsia e condições específicas",
        "estoque": False
    },

    "clonazepam_2mg": {
        "preco": "Medicamento controlado - necessário receita especial",
        "indicacao": "Transtornos de ansiedade e condições neurológicas específicas",
        "estoque": True
    },

    "alprazolam_1mg": {
        "preco": "Medicamento controlado - necessário receita especial",
        "indicacao": "Transtornos de ansiedade",
        "estoque": False
    },

    "lorazepam_2mg": {
        "preco": "Medicamento controlado - necessário receita especial",
        "indicacao": "Transtornos de ansiedade",
        "estoque": True
    },

    "diazepam_10mg": {
        "preco": "Medicamento controlado - necessário receita especial",
        "indicacao": "Ansiedade e espasmos musculares",
        "estoque": False
    },

    "zolpidem_10mg": {
        "preco": "Medicamento controlado - necessário receita especial",
        "indicacao": "Tratamento de insônia",
        "estoque": True
    },

    "trazodona_100mg": {
        "preco": "R$ 39,90 (100mg - 30 comprimidos) - necessário receita",
        "indicacao": "Depressão e condições relacionadas ao sono",
        "estoque": False
    },

    "sertralina_100mg": {
        "preco": "R$ 39,90 (100mg - 30 comprimidos) - necessário receita",
        "indicacao": "Depressão e transtornos de ansiedade",
        "estoque": True
    },

    "escitalopram_20mg": {
        "preco": "R$ 39,90 (20mg - 30 comprimidos) - necessário receita",
        "indicacao": "Depressão e ansiedade",
        "estoque": True
    },

    "fluoxetina_20mg": {
        "preco": "R$ 18,90 (20mg - 30 cápsulas) - necessário receita",
        "indicacao": "Depressão e transtornos específicos",
        "estoque": False
    },

    "paroxetina_20mg": {
        "preco": "R$ 27,90 (20mg - 30 comprimidos) - necessário receita",
        "indicacao": "Depressão e ansiedade",
        "estoque": True
    },

    "venlafaxina_150mg": {
        "preco": "R$ 49,90 (150mg - 30 cápsulas) - necessário receita",
        "indicacao": "Depressão e ansiedade",
        "estoque": False
    },

    "duloxetina_60mg": {
        "preco": "R$ 59,90 (60mg - 30 cápsulas) - necessário receita",
        "indicacao": "Depressão e algumas condições de dor",
        "estoque": True
    },

    "bupropiona_150mg": {
        "preco": "R$ 39,90 (150mg - 30 comprimidos) - necessário receita",
        "indicacao": "Depressão e outras indicações específicas",
        "estoque": True
    },

    "mirtazapina_30mg": {
        "preco": "R$ 39,90 (30mg - 30 comprimidos) - necessário receita",
        "indicacao": "Tratamento de depressão",
        "estoque": False
    },

    "quetiapina_100mg": {
        "preco": "R$ 49,90 (100mg - 30 comprimidos) - necessário receita",
        "indicacao": "Transtornos psiquiátricos específicos",
        "estoque": True
    },

    "risperidona_2mg": {
        "preco": "R$ 29,90 (2mg - 30 comprimidos) - necessário receita",
        "indicacao": "Transtornos psiquiátricos específicos",
        "estoque": False
    },

    "olanzapina_10mg": {
        "preco": "R$ 49,90 (10mg - 30 comprimidos) - necessário receita",
        "indicacao": "Transtornos psiquiátricos específicos",
        "estoque": True
    },

    "aripiprazol_10mg": {
        "preco": "R$ 69,90 (10mg - 30 comprimidos) - necessário receita",
        "indicacao": "Transtornos psiquiátricos específicos",
        "estoque": False
    },

    "haloperidol_5mg": {
        "preco": "R$ 15,90 (5mg - 20 comprimidos) - necessário receita",
        "indicacao": "Transtornos psiquiátricos específicos",
        "estoque": True
    },

    "clorpromazina_100mg": {
        "preco": "R$ 24,90 (100mg - 20 comprimidos) - necessário receita",
        "indicacao": "Transtornos psiquiátricos específicos",
        "estoque": False
    },

    "metilfenidato_10mg": {
        "preco": "Medicamento controlado - necessário receita especial",
        "indicacao": "Tratamento de TDAH e outras condições específicas",
        "estoque": True
    },

    "lisdexanfetamina_30mg": {
        "preco": "Medicamento controlado - necessário receita especial",
        "indicacao": "Tratamento de TDAH e outras condições específicas",
        "estoque": False
    },

    "atomoxetina": {
        "preco": "R$ 79,90 (40mg - 30 cápsulas) - necessário receita",
        "indicacao": "Tratamento de TDAH",
        "estoque": True
    },

    "clonidina_01": {
        "preco": "R$ 16,90 (0,1mg - 30 comprimidos)",
        "indicacao": "Pressão alta e condições específicas",
        "estoque": True
    },

    "melatonina_3mg": {
        "preco": "R$ 24,90 (3mg - 30 comprimidos)",
        "indicacao": "Auxílio na regulação do sono",
        "estoque": True
    },

    "melatonina_5mg": {
        "preco": "R$ 29,90 (5mg - 30 comprimidos)",
        "indicacao": "Auxílio na regulação do sono",
        "estoque": False
    },

    "valeriana": {
        "preco": "R$ 19,90 (cápsulas)",
        "indicacao": "Auxílio tradicional para relaxamento e sono",
        "estoque": True
    },

    "passiflora": {
        "preco": "R$ 18,90 (cápsulas)",
        "indicacao": "Auxílio tradicional para relaxamento",
        "estoque": True
    },

    "maracuja_extrato": {
        "preco": "R$ 21,90 (cápsulas)",
        "indicacao": "Auxílio tradicional para relaxamento",
        "estoque": False
    },

    "ginkgo_biloba": {
        "preco": "R$ 29,90 (cápsulas)",
        "indicacao": "Suplementação nutricional",
        "estoque": True
    },

    "guarana": {
        "preco": "R$ 19,90 (cápsulas)",
        "indicacao": "Suplementação nutricional",
        "estoque": True
    },

    "creatina_monohidratada": {
        "preco": "R$ 79,90 (pó - 300g)",
        "indicacao": "Suplementação esportiva",
        "estoque": True
    },

    "multivitaminico_mineral": {
        "preco": "R$ 34,90 (60 comprimidos)",
        "indicacao": "Suplementação de vitaminas e minerais",
        "estoque": True
    },

    "vitamina_c_500": {
        "preco": "R$ 16,90 (500mg - 30 comprimidos)",
        "indicacao": "Suplementação de vitamina C",
        "estoque": True
    },

    "vitamina_c_1g": {
        "preco": "R$ 22,90 (1g - 30 comprimidos)",
        "indicacao": "Suplementação de vitamina C",
        "estoque": False
    },

    "vitamina_d_7000": {
        "preco": "R$ 39,90 (cápsulas)",
        "indicacao": "Suplementação de vitamina D sob orientação",
        "estoque": True
    },

    "zinco_quelato": {
        "preco": "R$ 24,90 (cápsulas)",
        "indicacao": "Suplementação de zinco",
        "estoque": True
    },

    "magnesio_quelato": {
        "preco": "R$ 29,90 (cápsulas)",
        "indicacao": "Suplementação de magnésio",
        "estoque": False
    },

    "ferro_quelato": {
        "preco": "R$ 29,90 (cápsulas)",
        "indicacao": "Suplementação de ferro",
        "estoque": True
    },

    "acido_folico_5mg": {
        "preco": "R$ 9,90 (5mg - 30 comprimidos)",
        "indicacao": "Suplementação de ácido fólico",
        "estoque": True
    },

    "cianocobalamina_1000": {
        "preco": "R$ 22,90 (1000mcg - 30 comprimidos)",
        "indicacao": "Suplementação de vitamina B12",
        "estoque": True
    },

    "vitamina_b_complexo": {
        "preco": "R$ 18,90 (30 comprimidos)",
        "indicacao": "Suplementação de vitaminas do complexo B",
        "estoque": False
    },

    "omega_3_1000": {
        "preco": "R$ 39,90 (1000mg - 60 cápsulas)",
        "indicacao": "Suplementação nutricional",
        "estoque": True
    },

    "coenzima_q10_100": {
        "preco": "R$ 49,90 (100mg - 30 cápsulas)",
        "indicacao": "Suplementação nutricional",
        "estoque": False
    },

    "colageno_hidrolisado": {
        "preco": "R$ 59,90 (pó - 300g)",
        "indicacao": "Suplementação nutricional",
        "estoque": True
    },

    "glucosamina_condroitina": {
        "preco": "R$ 69,90 (cápsulas)",
        "indicacao": "Suplementação para articulações",
        "estoque": True
    },

    "probiótico": {
        "preco": "R$ 29,90 (cápsulas)",
        "indicacao": "Suplementação da flora intestinal",
        "estoque": True
    },

    "soro_fisiologico_100": {
        "preco": "R$ 6,90 (100mL)",
        "indicacao": "Higiene e hidratação",
        "estoque": True
    },

    "soro_fisiologico_250": {
        "preco": "R$ 9,90 (250mL)",
        "indicacao": "Higiene e hidratação",
        "estoque": False
    },

    "soro_fisiologico_1000": {
        "preco": "R$ 18,90 (1L)",
        "indicacao": "Uso em cuidados de saúde",
        "estoque": True
    },

    "agua_boricada": {
        "preco": "R$ 12,90 (solução)",
        "indicacao": "Uso externo específico",
        "estoque": False
    },

    "clorexidina_2": {
        "preco": "R$ 19,90 (solução antisséptica)",
        "indicacao": "Antissepsia da pele",
        "estoque": True
    },

    "clorexidina_bucal": {
        "preco": "R$ 17,90 (solução bucal)",
        "indicacao": "Higiene bucal e controle de placa conforme orientação",
        "estoque": True
    },

    "fluoreto_sodio": {
        "preco": "R$ 14,90 (solução bucal)",
        "indicacao": "Prevenção de cáries",
        "estoque": True
    },

    "benzidamina": {
        "preco": "R$ 19,90 (spray/solução oral)",
        "indicacao": "Alívio de inflamações e desconfortos da boca e garganta",
        "estoque": True
    },

    "lidocaina_oral": {
        "preco": "R$ 19,90 (gel)",
        "indicacao": "Anestesia local superficial",
        "estoque": False
    },

    "hexomedina": {
        "preco": "R$ 18,90 (pastilhas)",
        "indicacao": "Alívio de irritações da garganta",
        "estoque": True
    },

    "benzidamina_pastilhas": {
        "preco": "R$ 17,90 (pastilhas)",
        "indicacao": "Alívio de dor e irritação da garganta",
        "estoque": True
    },

    "ambroxol_pastilhas": {
        "preco": "R$ 16,90 (pastilhas)",
        "indicacao": "Alívio de sintomas respiratórios",
        "estoque": True
    },

    "mentol_pastilhas": {
        "preco": "R$ 9,90 (pastilhas)",
        "indicacao": "Alívio de irritação da garganta",
        "estoque": True
    },

    "guaifenesina_xarope": {
        "preco": "R$ 17,90 (xarope)",
        "indicacao": "Auxílio na eliminação de secreções respiratórias",
        "estoque": False
    },

    "acetilcisteina_600": {
        "preco": "R$ 18,90 (600mg - sachês)",
        "indicacao": "Fluidificação de secreções respiratórias",
        "estoque": True
    },

    "acetilcisteina_200": {
        "preco": "R$ 14,90 (200mg - sachês)",
        "indicacao": "Fluidificação de secreções respiratórias",
        "estoque": True
    },

    "bromexina_xarope": {
        "preco": "R$ 13,90 (xarope)",
        "indicacao": "Fluidificação de secreções respiratórias",
        "estoque": True
    },

    "ambroxol_30mg": {
        "preco": "R$ 16,90 (30mg - comprimidos)",
        "indicacao": "Auxílio na eliminação de secreções",
        "estoque": False
    },

    "oximetazolina_05": {
        "preco": "R$ 16,90 (spray nasal)",
        "indicacao": "Congestão nasal",
        "estoque": True
    },

    "nafazolina_nasal": {
        "preco": "R$ 11,90 (solução nasal)",
        "indicacao": "Congestão nasal",
        "estoque": False
    },

    "budesonida_32": {
        "preco": "R$ 31,90 (spray nasal)",
        "indicacao": "Rinite alérgica",
        "estoque": True
    },

    "fluticasona_50": {
        "preco": "R$ 49,90 (spray nasal)",
        "indicacao": "Rinite alérgica",
        "estoque": True
    },

    "mometasona_nasal": {
        "preco": "R$ 39,90 (spray nasal)",
        "indicacao": "Rinite alérgica",
        "estoque": False
    },

    "cetirizina_10": {
        "preco": "R$ 10,90 (10mg - 12 comprimidos)",
        "indicacao": "Alergias e rinite",
        "estoque": True
    },

    "desloratadina_5": {
        "preco": "R$ 16,90 (5mg - 10 comprimidos)",
        "indicacao": "Alergias e rinite",
        "estoque": True
    },

    "fexofenadina_120": {
        "preco": "R$ 21,90 (120mg - 10 comprimidos)",
        "indicacao": "Alergias e rinite",
        "estoque": False
    },

    "levocetirizina_5": {
        "preco": "R$ 18,90 (5mg - 10 comprimidos)",
        "indicacao": "Alergias e rinite",
        "estoque": True
    },

    "loratadina_10": {
        "preco": "R$ 11,90 (10mg - 12 comprimidos)",
        "indicacao": "Alergias e coriza",
        "estoque": True
    },

    "hidroxizina_25": {
        "preco": "R$ 15,90 (25mg - 30 comprimidos) - necessário receita",
        "indicacao": "Alergias e coceira",
        "estoque": False
    },

    "prometazina_25": {
        "preco": "R$ 13,90 (25mg - 20 comprimidos)",
        "indicacao": "Alergias e náuseas",
        "estoque": True
    },

    "ebastina": {
        "preco": "R$ 24,90 (10mg - 10 comprimidos)",
        "indicacao": "Alergias e rinite",
        "estoque": True
    },

    "bilastina": {
        "preco": "R$ 29,90 (20mg - 10 comprimidos)",
        "indicacao": "Alergias e rinite",
        "estoque": False
    },

    "rupatadina": {
        "preco": "R$ 27,90 (10mg - 10 comprimidos)",
        "indicacao": "Alergias e rinite",
        "estoque": True
    },

    "azelastina": {
        "preco": "R$ 29,90 (spray nasal)",
        "indicacao": "Rinite alérgica",
        "estoque": True
    },

    "ipratropio_nasal": {
        "preco": "R$ 29,90 (spray nasal)",
        "indicacao": "Sintomas nasais específicos",
        "estoque": False
    },

    "salbutamol_100": {
        "preco": "R$ 22,90 (100mcg - inalador)",
        "indicacao": "Alívio do broncoespasmo",
        "estoque": True
    },

    "budesonida_200": {
        "preco": "R$ 39,90 (200mcg - inalador)",
        "indicacao": "Controle de inflamação respiratória",
        "estoque": True
    },

    "beclometasona_200": {
        "preco": "R$ 34,90 (200mcg - inalador)",
        "indicacao": "Controle de inflamação respiratória",
        "estoque": False
    },

    "formoterol_12": {
        "preco": "R$ 49,90 (12mcg - inalador)",
        "indicacao": "Controle de sintomas respiratórios",
        "estoque": True
    },

    "salmeterol_25": {
        "preco": "R$ 59,90 (25mcg - inalador)",
        "indicacao": "Controle de doenças respiratórias específicas",
        "estoque": False
    },

    "tiotropio_18": {
        "preco": "R$ 89,90 (18mcg - inalador) - necessário receita",
        "indicacao": "Controle de doenças respiratórias específicas",
        "estoque": True
    },

    "montelucaste_10": {
        "preco": "R$ 29,90 (10mg - 30 comprimidos)",
        "indicacao": "Controle de asma e rinite",
        "estoque": True
    },

    "teofilina_200": {
        "preco": "R$ 24,90 (200mg - 30 comprimidos)",
        "indicacao": "Tratamento respiratório específico",
        "estoque": False
    },

    "prednisona_20": {
        "preco": "R$ 14,90 (20mg - 20 comprimidos) - necessário receita",
        "indicacao": "Processos inflamatórios e alérgicos",
        "estoque": True
    },

    "dexametasona_4": {
        "preco": "R$ 12,90 (4mg - 10 comprimidos) - necessário receita",
        "indicacao": "Processos inflamatórios específicos",
        "estoque": False
    },

    "prednisolona_20": {
        "preco": "R$ 19,90 (20mg - 20 comprimidos) - necessário receita",
        "indicacao": "Processos inflamatórios e alérgicos",
        "estoque": True
    },

    "ibuprofeno_600": {
        "preco": "R$ 16,90 (600mg - 20 comprimidos)",
        "indicacao": "Dor, inflamação e febre",
        "estoque": True
    },

    "naproxeno_550": {
        "preco": "R$ 16,90 (550mg - 10 comprimidos)",
        "indicacao": "Dor e inflamação",
        "estoque": False
    },

    "diclofenaco_50": {
        "preco": "R$ 13,90 (50mg - 20 comprimidos)",
        "indicacao": "Dor e inflamação",
        "estoque": True
    },

    "meloxicam_15": {
        "preco": "R$ 18,90 (15mg - 10 comprimidos)",
        "indicacao": "Dor e inflamação",
        "estoque": True
    },

    "nimesulida_100": {
        "preco": "R$ 12,90 (100mg - 12 comprimidos)",
        "indicacao": "Dor e inflamação",
        "estoque": False
    },

    "cetoprofeno_100": {
        "preco": "R$ 17,90 (100mg - 20 comprimidos)",
        "indicacao": "Dor e inflamação",
        "estoque": True
    },

    "celecoxibe_200": {
        "preco": "R$ 29,90 (200mg - 10 cápsulas)",
        "indicacao": "Dor e inflamação",
        "estoque": False
    },

    "etoricoxibe": {
        "preco": "R$ 39,90 (90mg - 7 comprimidos)",
        "indicacao": "Dor e inflamação",
        "estoque": True
    },

    "piroxicam": {
        "preco": "R$ 19,90 (20mg - 10 comprimidos)",
        "indicacao": "Dor e inflamação",
        "estoque": False
    },

    "tenoxicam": {
        "preco": "R$ 29,90 (20mg - 10 comprimidos)",
        "indicacao": "Dor e inflamação",
        "estoque": True
    },

    "meloxicam_7_5": {
        "preco": "R$ 16,90 (7,5mg - 10 comprimidos)",
        "indicacao": "Dor e inflamação",
        "estoque": True
    },

    "paracetamol_500": {
        "preco": "R$ 7,50 (500mg - 20 comprimidos)",
        "indicacao": "Dor e febre",
        "estoque": True
    },

    "paracetamol_750": {
        "preco": "R$ 8,90 (750mg - 20 comprimidos)",
        "indicacao": "Dor e febre",
        "estoque": False
    },

    "dipirona_500": {
        "preco": "R$ 8,90 (500mg - 20 comprimidos)",
        "indicacao": "Dor e febre",
        "estoque": True
    },

    "dipirona_1g": {
        "preco": "R$ 11,90 (1g - 10 comprimidos)",
        "indicacao": "Dor e febre",
        "estoque": True
    },

    "aspirina_100": {
        "preco": "R$ 9,90 (100mg - 30 comprimidos)",
        "indicacao": "Uso cardiovascular específico conforme orientação médica",
        "estoque": True
    },

    "aspirina_500": {
        "preco": "R$ 12,90 (500mg - 20 comprimidos)",
        "indicacao": "Dor e febre",
        "estoque": False
    },

    "metamizol_500": {
        "preco": "R$ 9,90 (500mg - 20 comprimidos)",
        "indicacao": "Dor e febre",
        "estoque": True
    },

    "tramadol_50": {
        "preco": "Medicamento controlado - necessário receita especial",
        "indicacao": "Tratamento de dores moderadas a intensas",
        "estoque": False
    },

    "codeina_30": {
        "preco": "Medicamento controlado - necessário receita especial",
        "indicacao": "Tratamento de algumas dores",
        "estoque": True
    },

    "morfina_10": {
        "preco": "Medicamento controlado - necessário receita especial",
        "indicacao": "Tratamento de dores intensas",
        "estoque": False
    },

    "oxicodona_10": {
        "preco": "Medicamento controlado - necessário receita especial",
        "indicacao": "Tratamento de dores intensas",
        "estoque": False
    },

    "fentanil": {
        "preco": "Medicamento controlado - uso hospitalar",
        "indicacao": "Analgesia intensa em ambiente médico",
        "estoque": False
    },

    "buprenorfina": {
        "preco": "Medicamento controlado - necessário receita especial",
        "indicacao": "Tratamento de dores específicas",
        "estoque": False
    },

    "naloxona_nasal": {
        "preco": "R$ 149,90 (spray nasal)",
        "indicacao": "Reversão emergencial de efeitos de opioides",
        "estoque": False
    },

    "lidocaina_2": {
        "preco": "R$ 19,90 (gel/solução)",
        "indicacao": "Anestesia local",
        "estoque": True
    },

    "benzocaina_gel": {
        "preco": "R$ 14,90 (gel)",
        "indicacao": "Anestesia local superficial",
        "estoque": True
    },

    "procaína": {
        "preco": "R$ 29,90 (solução) - uso profissional",
        "indicacao": "Anestesia local em situações específicas",
        "estoque": False
    },

    "bupivacaina": {
        "preco": "R$ 39,90 (solução) - uso profissional",
        "indicacao": "Anestesia local em ambiente médico",
        "estoque": True
    },

    "ropivacaina": {
        "preco": "R$ 49,90 (solução) - uso profissional",
        "indicacao": "Anestesia local em ambiente médico",
        "estoque": False
    },

    "cetamina": {
        "preco": "Medicamento controlado - uso hospitalar",
        "indicacao": "Uso anestésico e médico especializado",
        "estoque": False
    },

    "propofol": {
        "preco": "R$ 99,90 (ampola) - uso hospitalar",
        "indicacao": "Anestesia geral em ambiente médico",
        "estoque": False
    },

    "midazolam": {
        "preco": "Medicamento controlado - uso hospitalar",
        "indicacao": "Sedação e anestesia em ambiente médico",
        "estoque": True
    },

    "etomidato": {
        "preco": "R$ 79,90 (ampola) - uso hospitalar",
        "indicacao": "Indução anestésica em ambiente médico",
        "estoque": False
    },

    "sevoflurano": {
        "preco": "R$ 299,90 (frasco) - uso hospitalar",
        "indicacao": "Anestesia geral em ambiente hospitalar",
        "estoque": False
    },

    "isoflurano": {
        "preco": "R$ 249,90 (frasco) - uso hospitalar",
        "indicacao": "Anestesia geral em ambiente hospitalar",
        "estoque": False
    },

    "succinilcolina": {
        "preco": "R$ 49,90 (frasco-ampola) - uso hospitalar",
        "indicacao": "Relaxamento muscular em procedimentos médicos",
        "estoque": False
    },

    "rocuronio": {
        "preco": "R$ 89,90 (frasco-ampola) - uso hospitalar",
        "indicacao": "Bloqueio neuromuscular em procedimentos médicos",
        "estoque": False
    },

    "neostigmina": {
        "preco": "R$ 29,90 (ampola) - uso profissional",
        "indicacao": "Reversão de bloqueio neuromuscular em situações específicas",
        "estoque": True
    },

    "atropina": {
        "preco": "R$ 19,90 (ampola) - uso profissional",
        "indicacao": "Uso médico em situações específicas",
        "estoque": True
    },

    "adrenalina_autoinjetavel": {
        "preco": "R$ 299,90 (autoinjetor) - necessário receita",
        "indicacao": "Tratamento emergencial de anafilaxia",
        "estoque": False
    },

    "hidroxocobalamina_1mg": {
        "preco": "R$ 29,90 (ampola)",
        "indicacao": "Reposição de vitamina B12",
        "estoque": True
    },

    "tiamina": {
        "preco": "R$ 19,90 (comprimidos)",
        "indicacao": "Suplementação de vitamina B1",
        "estoque": True
    },

    "piridoxina": {
        "preco": "R$ 17,90 (comprimidos)",
        "indicacao": "Suplementação de vitamina B6",
        "estoque": False
    },

    "riboflavina": {
        "preco": "R$ 15,90 (comprimidos)",
        "indicacao": "Suplementação de vitamina B2",
        "estoque": True
    },

    "niacina": {
        "preco": "R$ 18,90 (comprimidos)",
        "indicacao": "Suplementação de vitamina B3",
        "estoque": True
    },

    "biotina_5mg": {
        "preco": "R$ 24,90 (5mg - cápsulas)",
        "indicacao": "Suplementação nutricional",
        "estoque": False
    },

    "vitamina_e_400": {
        "preco": "R$ 21,90 (400UI - cápsulas)",
        "indicacao": "Suplementação de vitamina E",
        "estoque": True
    },

    "vitamina_a_10000": {
        "preco": "R$ 19,90 (cápsulas)",
        "indicacao": "Suplementação de vitamina A",
        "estoque": False
    },

    "vitamina_k2": {
        "preco": "R$ 39,90 (cápsulas)",
        "indicacao": "Suplementação nutricional",
        "estoque": True
    },

    "selênio_100": {
        "preco": "R$ 19,90 (100mcg - comprimidos)",
        "indicacao": "Suplementação de selênio",
        "estoque": True
    },

    "zinco_30": {
        "preco": "R$ 18,90 (30mg - comprimidos)",
        "indicacao": "Suplementação de zinco",
        "estoque": True
    },

    "magnesio_400": {
        "preco": "R$ 24,90 (400mg - comprimidos)",
        "indicacao": "Suplementação de magnésio",
        "estoque": False
    },

    "calcio_600": {
        "preco": "R$ 24,90 (600mg - comprimidos)",
        "indicacao": "Suplementação de cálcio",
        "estoque": True
    },

    "ferro_40": {
        "preco": "R$ 17,90 (40mg - comprimidos)",
        "indicacao": "Reposição de ferro",
        "estoque": True
    },

    "acido_folico_400": {
        "preco": "R$ 9,90 (400mcg - comprimidos)",
        "indicacao": "Suplementação de ácido fólico",
        "estoque": True
    },

    "omega_3_500": {
        "preco": "R$ 29,90 (500mg - 60 cápsulas)",
        "indicacao": "Suplementação nutricional",
        "estoque": False
    },

    "vitamina_d_1000": {
        "preco": "R$ 19,90 (1000UI - 30 cápsulas)",
        "indicacao": "Suplementação de vitamina D",
        "estoque": True
    },

    "vitamina_d_2000": {
        "preco": "R$ 24,90 (2000UI - 30 cápsulas)",
        "indicacao": "Suplementação de vitamina D",
        "estoque": True
    },

    "vitamina_d_5000": {
        "preco": "R$ 34,90 (5000UI - 30 cápsulas)",
        "indicacao": "Suplementação de vitamina D sob orientação",
        "estoque": False
    },

    "polivitaminico_adulto": {
        "preco": "R$ 29,90 (30 comprimidos)",
        "indicacao": "Suplementação de vitaminas e minerais",
        "estoque": True
    },

    "polivitaminico_senior": {
        "preco": "R$ 39,90 (30 comprimidos)",
        "indicacao": "Suplementação de vitaminas e minerais",
        "estoque": True
    },

    "polivitaminico_infantil": {
        "preco": "R$ 29,90 (solução oral)",
        "indicacao": "Suplementação nutricional infantil",
        "estoque": False
    },

    "sais_reidratacao": {
        "preco": "R$ 5,90 (envelope)",
        "indicacao": "Reposição de líquidos e eletrólitos",
        "estoque": True
    },

    "zinco_soro": {
        "preco": "R$ 19,90 (solução oral)",
        "indicacao": "Suplementação de zinco em situações específicas",
        "estoque": True
    },

    "glicose_oral": {
        "preco": "R$ 9,90 (gel oral)",
        "indicacao": "Fonte rápida de glicose em situações específicas",
        "estoque": True
    },

    "gel_reidratacao": {
        "preco": "R$ 7,90 (sachê)",
        "indicacao": "Reposição de líquidos e eletrólitos",
        "estoque": False
    },

    "barreira_protetora": {
        "preco": "R$ 19,90 (creme)",
        "indicacao": "Proteção da pele",
        "estoque": True
    },

    "creme_ureia_20": {
        "preco": "R$ 22,90 (creme 20%)",
        "indicacao": "Hidratação de pele muito ressecada",
        "estoque": True
    },

    "creme_ureia_10": {
        "preco": "R$ 19,90 (creme 10%)",
        "indicacao": "Hidratação da pele",
        "estoque": True
    },

    "acido_salicilico": {
        "preco": "R$ 18,90 (solução tópica)",
        "indicacao": "Tratamento tópico de condições específicas da pele",
        "estoque": False
    },

    "peroxido_benzoila_5": {
        "preco": "R$ 27,90 (gel 5%)",
        "indicacao": "Tratamento tópico da acne",
        "estoque": True
    },

    "adapaleno_01": {
        "preco": "R$ 32,90 (gel 0,1%)",
        "indicacao": "Tratamento tópico da acne",
        "estoque": True
    },

    "clindamicina_topica": {
        "preco": "R$ 29,90 (gel dermatológico)",
        "indicacao": "Tratamento tópico da acne",
        "estoque": False
    },

    "isoconazol": {
        "preco": "R$ 24,90 (creme)",
        "indicacao": "Tratamento de infecções fúngicas da pele",
        "estoque": True
    },

    "econazol": {
        "preco": "R$ 27,90 (creme)",
        "indicacao": "Tratamento de infecções fúngicas da pele",
        "estoque": False
    },

    "oxiconazol": {
        "preco": "R$ 29,90 (creme)",
        "indicacao": "Tratamento de infecções fúngicas da pele",
        "estoque": True
    },

    "ciclopirox": {
        "preco": "R$ 34,90 (creme)",
        "indicacao": "Tratamento de infecções fúngicas da pele",
        "estoque": True
    },

    "naftifina": {
        "preco": "R$ 29,90 (creme)",
        "indicacao": "Tratamento de algumas micoses",
        "estoque": False
    },

    "amorolfina": {
        "preco": "R$ 59,90 (esmalte para unhas)",
        "indicacao": "Tratamento tópico de algumas micoses das unhas",
        "estoque": True
    },

    "terbinafina_spray": {
        "preco": "R$ 29,90 (spray tópico)",
        "indicacao": "Tratamento de micoses da pele",
        "estoque": True
    },

    "permanganato": {
        "preco": "R$ 8,90 (comprimidos para solução)",
        "indicacao": "Uso externo específico",
        "estoque": False
    },

    "sulfato_de_prata": {
        "preco": "R$ 29,90 (creme)",
        "indicacao": "Cuidados específicos de feridas",
        "estoque": True
    },

    "colagenase_cloranfenicol": {
        "preco": "R$ 44,90 (pomada)",
        "indicacao": "Tratamento de feridas específicas",
        "estoque": False
    },

    "papaína": {
        "preco": "R$ 39,90 (pomada)",
        "indicacao": "Cuidados específicos de feridas",
        "estoque": True
    },

    "hidrogel": {
        "preco": "R$ 39,90 (gel para curativo)",
        "indicacao": "Cuidados de feridas",
        "estoque": True
    },

    "alginato_calcio": {
        "preco": "R$ 49,90 (curativo)",
        "indicacao": "Curativos de feridas específicas",
        "estoque": False
    },

    "carvao_ativado_curativo": {
        "preco": "R$ 59,90 (curativo)",
        "indicacao": "Curativos de feridas específicas",
        "estoque": True
    },

    "pomada_dexpantenol": {
        "preco": "R$ 17,90 (pomada)",
        "indicacao": "Hidratação e proteção da pele",
        "estoque": True
    },

    "pomada_oxido_zinco": {
        "preco": "R$ 12,90 (pomada)",
        "indicacao": "Proteção da pele",
        "estoque": True
    },

    "creme_calendula": {
        "preco": "R$ 16,90 (creme)",
        "indicacao": "Hidratação e cuidado da pele",
        "estoque": False
    },

    "creme_aloe_vera": {
        "preco": "R$ 19,90 (creme)",
        "indicacao": "Hidratação da pele",
        "estoque": True
    },

    "gel_arnique": {
        "preco": "R$ 18,90 (gel)",
        "indicacao": "Alívio de desconforto muscular",
        "estoque": True
    },

    "mentol": {
        "preco": "R$ 12,90 (gel tópico)",
        "indicacao": "Sensação refrescante e alívio temporário",
        "estoque": True
    },

    "canfora": {
        "preco": "R$ 13,90 (pomada tópica)",
        "indicacao": "Alívio temporário de desconforto muscular",
        "estoque": False
    },

    "salicilato_metila": {
        "preco": "R$ 16,90 (pomada tópica)",
        "indicacao": "Alívio temporário de dores musculares",
        "estoque": True
    },

    "capsaicina": {
        "preco": "R$ 29,90 (creme tópico)",
        "indicacao": "Alívio de algumas dores neuropáticas",
        "estoque": False
    },

    "lidocaina_patch": {
        "preco": "R$ 69,90 (adesivos) - necessário receita",
        "indicacao": "Alívio de algumas dores localizadas",
        "estoque": False
    },

    "diclofenaco_adesivo": {
        "preco": "R$ 39,90 (adesivos)",
        "indicacao": "Alívio de dor e inflamação localizada",
        "estoque": True
    },

    "ibuprofeno_gel": {
        "preco": "R$ 19,90 (gel tópico)",
        "indicacao": "Dor e inflamação localizada",
        "estoque": True
    },

    "diclofenaco_emulgel": {
        "preco": "R$ 19,90 (gel tópico)",
        "indicacao": "Dor e inflamação localizada",
        "estoque": True
    },

    "cetoprofeno_gel": {
        "preco": "R$ 22,90 (gel tópico)",
        "indicacao": "Dor e inflamação localizada",
        "estoque": False
    },

    "naproxeno_sodio": {
        "preco": "R$ 17,90 (550mg - 10 comprimidos)",
        "indicacao": "Dor e inflamação",
        "estoque": True
    },

    "etodolaco": {
        "preco": "R$ 29,90 (400mg - 20 comprimidos)",
        "indicacao": "Dor e inflamação",
        "estoque": False
    },

    "lornoxicam": {
        "preco": "R$ 39,90 (8mg - 10 comprimidos)",
        "indicacao": "Dor e inflamação",
        "estoque": True
    },

    "dexketoprofeno": {
        "preco": "R$ 24,90 (25mg - 10 comprimidos)",
        "indicacao": "Dor e inflamação",
        "estoque": True
    },

    "dipirona_composta": {
        "preco": "R$ 14,90 (comprimidos)",
        "indicacao": "Dor e sintomas associados",
        "estoque": False
    },

    "paracetamol_cafeina": {
        "preco": "R$ 16,90 (comprimidos)",
        "indicacao": "Dor de cabeça e outras dores",
        "estoque": True
    },

    "ibuprofeno_400": {
        "preco": "R$ 12,90 (400mg - 20 comprimidos)",
        "indicacao": "Dor, inflamação e febre",
        "estoque": True
    },

    "naproxeno_250": {
        "preco": "R$ 14,90 (250mg - 20 comprimidos)",
        "indicacao": "Dor e inflamação",
        "estoque": False
    },

    "diclofenaco_75": {
        "preco": "R$ 18,90 (75mg - comprimidos)",
        "indicacao": "Dor e inflamação",
        "estoque": True
    },

    "meloxicam_15mg": {
        "preco": "R$ 18,90 (15mg - 10 comprimidos)",
        "indicacao": "Dor e inflamação",
        "estoque": True
    },

    "piroxicam_20": {
        "preco": "R$ 19,90 (20mg - 10 comprimidos)",
        "indicacao": "Dor e inflamação",
        "estoque": False
    },

    "tenoxicam_20": {
        "preco": "R$ 29,90 (20mg - 10 comprimidos)",
        "indicacao": "Dor e inflamação",
        "estoque": True
    },

    "etoricoxibe_90": {
        "preco": "R$ 39,90 (90mg - 7 comprimidos)",
        "indicacao": "Dor e inflamação",
        "estoque": False
    },

    "celecoxibe_100": {
        "preco": "R$ 24,90 (100mg - 10 cápsulas)",
        "indicacao": "Dor e inflamação",
        "estoque": True
    },

    "nimesulida_gel": {
        "preco": "R$ 19,90 (gel tópico)",
        "indicacao": "Dor e inflamação localizada",
        "estoque": True
    },

    "diclofenaco_pomada": {
        "preco": "R$ 19,90 (pomada)",
        "indicacao": "Dor e inflamação localizada",
        "estoque": True
    },

    "ketorolaco": {
        "preco": "R$ 29,90 (10mg - comprimidos)",
        "indicacao": "Tratamento de dor aguda conforme orientação",
        "estoque": False
    },

    "ketorolaco_colirio": {
        "preco": "R$ 29,90 (colírio) - necessário receita",
        "indicacao": "Dor e inflamação ocular específica",
        "estoque": True
    },

    "sumatriptana_50": {
        "preco": "R$ 29,90 (50mg - 4 comprimidos)",
        "indicacao": "Tratamento de crises de enxaqueca",
        "estoque": True
    },

    "sumatriptana_100": {
        "preco": "R$ 39,90 (100mg - 4 comprimidos)",
        "indicacao": "Tratamento de crises de enxaqueca",
        "estoque": False
    },

    "rizatriptana": {
        "preco": "R$ 34,90 (10mg - 4 comprimidos)",
        "indicacao": "Tratamento de crises de enxaqueca",
        "estoque": True
    },

    "naratriptana_25": {
        "preco": "R$ 34,90 (2,5mg - 4 comprimidos)",
        "indicacao": "Tratamento de crises de enxaqueca",
        "estoque": False
    },

    "zolmitriptana": {
        "preco": "R$ 39,90 (2,5mg - 4 comprimidos)",
        "indicacao": "Tratamento de crises de enxaqueca",
        "estoque": True
    },

    "topiramato_50": {
        "preco": "R$ 34,90 (50mg - 30 comprimidos) - necessário receita",
        "indicacao": "Prevenção de enxaqueca e epilepsia",
        "estoque": True
    },

    "amitriptilina_25": {
        "preco": "R$ 16,90 (25mg - 30 comprimidos) - necessário receita",
        "indicacao": "Algumas condições de dor e depressão",
        "estoque": False
    },

    "nortriptilina_25": {
        "preco": "R$ 24,90 (25mg - 30 cápsulas) - necessário receita",
        "indicacao": "Algumas condições de dor e depressão",
        "estoque": True
    },

    "propranolol_40": {
        "preco": "R$ 10,90 (40mg - 30 comprimidos)",
        "indicacao": "Controle cardiovascular e prevenção de algumas enxaquecas",
        "estoque": True
    },

    "flunarizina_10": {
        "preco": "R$ 24,90 (10mg - 30 comprimidos)",
        "indicacao": "Prevenção de enxaqueca e vertigem",
        "estoque": False
    },

    "betahistina_24": {
        "preco": "R$ 29,90 (24mg - 30 comprimidos)",
        "indicacao": "Tratamento de vertigem em condições específicas",
        "estoque": True
    },

    "cinarizina_75": {
        "preco": "R$ 19,90 (75mg - 30 comprimidos)",
        "indicacao": "Tontura e condições vestibulares",
        "estoque": True
    },

    "dimenidrinato_50": {
        "preco": "R$ 14,90 (50mg - 20 comprimidos)",
        "indicacao": "Enjoo, náuseas e vômitos",
        "estoque": False
    },

    "meclizina_25mg": {
        "preco": "R$ 18,90 (25mg - 10 comprimidos)",
        "indicacao": "Náuseas e tontura",
        "estoque": True
    },

    "bromoprida_10": {
        "preco": "R$ 13,90 (10mg - 20 cápsulas)",
        "indicacao": "Náuseas e distúrbios gastrointestinais",
        "estoque": True
    },

    "domperidona_10": {
        "preco": "R$ 14,90 (10mg - 30 comprimidos)",
        "indicacao": "Náuseas e distúrbios gastrointestinais",
        "estoque": False
    },

    "metoclopramida_10": {
        "preco": "R$ 9,90 (10mg - 20 comprimidos)",
        "indicacao": "Náuseas e vômitos",
        "estoque": True
    },

    "ondansetrona_8": {
        "preco": "R$ 24,90 (8mg - 10 comprimidos)",
        "indicacao": "Náuseas e vômitos",
        "estoque": True
    },

    "ondansetrona_4": {
        "preco": "R$ 19,90 (4mg - 10 comprimidos)",
        "indicacao": "Náuseas e vômitos",
        "estoque": False
    },

    "simeticona_40": {
        "preco": "R$ 11,90 (40mg - 20 comprimidos)",
        "indicacao": "Gases e desconforto abdominal",
        "estoque": True
    },

    "simeticona_125": {
        "preco": "R$ 13,90 (125mg - cápsulas)",
        "indicacao": "Gases e desconforto abdominal",
        "estoque": True
    },

    "loperamida_2": {
        "preco": "R$ 10,90 (2mg - 12 cápsulas)",
        "indicacao": "Diarreia aguda",
        "estoque": False
    },

    "racecadotrila_100mg": {
        "preco": "R$ 24,90 (100mg - cápsulas)",
        "indicacao": "Tratamento sintomático da diarreia aguda",
        "estoque": True
    },

    "lactulose_667mg": {
        "preco": "R$ 18,90 (667mg/mL - 120mL)",
        "indicacao": "Constipação intestinal",
        "estoque": True
    },

    "bisacodil_5": {
        "preco": "R$ 12,90 (5mg - 20 comprimidos)",
        "indicacao": "Constipação intestinal",
        "estoque": False
    },

    "sene_12": {
        "preco": "R$ 9,90 (comprimidos)",
        "indicacao": "Constipação ocasional",
        "estoque": True
    },

    "psyllium_po": {
        "preco": "R$ 24,90 (pó para solução oral)",
        "indicacao": "Fonte de fibras e auxílio intestinal",
        "estoque": True
    },

    "polietilenoglicol_sache": {
        "preco": "R$ 29,90 (sachês)",
        "indicacao": "Constipação intestinal",
        "estoque": False
    },

    "sucralfato_1g": {
        "preco": "R$ 27,90 (1g - 30 comprimidos)",
        "indicacao": "Proteção da mucosa gástrica",
        "estoque": True
    },

    "famotidina_20": {
        "preco": "R$ 18,90 (20mg - 20 comprimidos)",
        "indicacao": "Azia e excesso de acidez",
        "estoque": False
    },

    "pantoprazol_40": {
        "preco": "R$ 19,90 (40mg - 28 comprimidos)",
        "indicacao": "Refluxo e redução da acidez gástrica",
        "estoque": True
    },

    "esomeprazol_40": {
        "preco": "R$ 27,90 (40mg - 28 cápsulas)",
        "indicacao": "Refluxo e redução da acidez gástrica",
        "estoque": True
    },

    "rabeprazol_20": {
        "preco": "R$ 29,90 (20mg - 28 comprimidos)",
        "indicacao": "Refluxo e redução da acidez gástrica",
        "estoque": False
    },

    "hidroxido_magnesio": {
        "preco": "R$ 12,90 (suspensão oral)",
        "indicacao": "Azia e constipação",
        "estoque": True
    },

    "hidroxido_aluminio": {
        "preco": "R$ 13,90 (suspensão oral)",
        "indicacao": "Azia e excesso de acidez",
        "estoque": True
    },

    "carbonato_calcio_500": {
        "preco": "R$ 18,90 (500mg - 60 comprimidos)",
        "indicacao": "Suplementação de cálcio",
        "estoque": False
    },

    "alendronato_70": {
        "preco": "R$ 24,90 (70mg - 4 comprimidos)",
        "indicacao": "Tratamento e prevenção da osteoporose",
        "estoque": True
    },

    "risedronato_35": {
        "preco": "R$ 39,90 (35mg - 4 comprimidos)",
        "indicacao": "Tratamento da osteoporose",
        "estoque": False
    },

    "calcitonina": {
        "preco": "R$ 89,90 (spray/ampola) - necessário receita",
        "indicacao": "Tratamento de condições ósseas específicas",
        "estoque": False
    },

    "colecalciferol": {
        "preco": "R$ 24,90 (2000UI - cápsulas)",
        "indicacao": "Suplementação de vitamina D",
        "estoque": True
    },

    "calcitriol": {
        "preco": "R$ 29,90 (0,25mcg - 30 cápsulas) - necessário receita",
        "indicacao": "Tratamento de condições específicas relacionadas ao cálcio",
        "estoque": False
    },

    "teriparatida": {
        "preco": "R$ 999,90 (caneta) - necessário receita",
        "indicacao": "Tratamento especializado da osteoporose",
        "estoque": False
    },

    "denosumabe": {
        "preco": "R$ 799,90 (seringa) - necessário receita",
        "indicacao": "Tratamento de condições ósseas específicas",
        "estoque": False
    },

    "estradiol_terapia": {
        "preco": "R$ 39,90 (comprimidos) - necessário receita",
        "indicacao": "Terapia hormonal específica",
        "estoque": True
    },

    "progesterona_100": {
        "preco": "R$ 49,90 (100mg - cápsulas) - necessário receita",
        "indicacao": "Terapia hormonal específica",
        "estoque": False
    },

    "testosterona": {
        "preco": "Medicamento controlado - necessário receita especial",
        "indicacao": "Reposição hormonal em situações específicas",
        "estoque": False
    },

    "levotiroxina_25": {
        "preco": "R$ 12,90 (25mcg - 30 comprimidos)",
        "indicacao": "Reposição de hormônio da tireoide",
        "estoque": True
    },

    "levotiroxina_50": {
        "preco": "R$ 15,90 (50mcg - 30 comprimidos)",
        "indicacao": "Reposição de hormônio da tireoide",
        "estoque": True
    },

    "levotiroxina_75": {
        "preco": "R$ 17,90 (75mcg - 30 comprimidos)",
        "indicacao": "Reposição de hormônio da tireoide",
        "estoque": False
    },

    "levotiroxina_100": {
        "preco": "R$ 18,90 (100mcg - 30 comprimidos)",
        "indicacao": "Reposição de hormônio da tireoide",
        "estoque": True
    },

    "levotiroxina_125": {
        "preco": "R$ 21,90 (125mcg - 30 comprimidos)",
        "indicacao": "Reposição de hormônio da tireoide",
        "estoque": True
    },

    "levotiroxina_150": {
        "preco": "R$ 24,90 (150mcg - 30 comprimidos)",
        "indicacao": "Reposição de hormônio da tireoide",
        "estoque": False
    },

    "metimazol_10": {
        "preco": "R$ 25,90 (10mg - 30 comprimidos)",
        "indicacao": "Tratamento do hipertireoidismo",
        "estoque": True
    },

    "propiltiouracil": {
        "preco": "R$ 29,90 (100mg - 30 comprimidos)",
        "indicacao": "Tratamento de hipertireoidismo em situações específicas",
        "estoque": False
    },

    "iodo_radioativo": {
        "preco": "Uso hospitalar especializado",
        "indicacao": "Tratamentos específicos da tireoide",
        "estoque": False
    },

    "desmopressina_01": {
        "preco": "R$ 49,90 (comprimidos) - necessário receita",
        "indicacao": "Tratamento de condições hormonais específicas",
        "estoque": True
    },

    "hidrocortisona": {
        "preco": "R$ 11,90 (creme dermatológico)",
        "indicacao": "Inflamações e irritações da pele",
        "estoque": True
    },

    "hidrocortisona_comprimido": {
        "preco": "R$ 29,90 (comprimidos) - necessário receita",
        "indicacao": "Reposição hormonal em situações específicas",
        "estoque": False
    },

    "fludrocortisona": {
        "preco": "R$ 49,90 (comprimidos) - necessário receita",
        "indicacao": "Reposição hormonal específica",
        "estoque": False
    },

    "prednisona_5": {
        "preco": "R$ 9,90 (5mg - 20 comprimidos) - necessário receita",
        "indicacao": "Processos inflamatórios e alérgicos",
        "estoque": True
    },

    "metotrexato_2_5": {
        "preco": "R$ 39,90 (2,5mg - comprimidos) - necessário receita",
        "indicacao": "Doenças autoimunes específicas",
        "estoque": False
    },

    "azatioprina_50": {
        "preco": "R$ 44,90 (50mg - 30 comprimidos) - necessário receita",
        "indicacao": "Imunossupressão em condições específicas",
        "estoque": True
    },

    "micofenolato_500": {
        "preco": "R$ 99,90 (500mg - comprimidos) - necessário receita",
        "indicacao": "Imunossupressão em condições específicas",
        "estoque": False
    },

    "ciclosporina_100": {
        "preco": "R$ 149,90 (100mg - cápsulas) - necessário receita",
        "indicacao": "Imunossupressão em condições específicas",
        "estoque": False
    },

    "tacrolimo": {
        "preco": "R$ 129,90 (cápsulas) - necessário receita",
        "indicacao": "Imunossupressão em condições específicas",
        "estoque": True
    },

    "colchicina_05": {
        "preco": "R$ 19,90 (0,5mg - 20 comprimidos)",
        "indicacao": "Tratamento e prevenção de crises de gota",
        "estoque": True
    },

    "alopurinol_100": {
        "preco": "R$ 17,90 (100mg - 30 comprimidos)",
        "indicacao": "Controle do ácido úrico",
        "estoque": True
    },

    "alopurinol_300": {
        "preco": "R$ 24,90 (300mg - 30 comprimidos)",
        "indicacao": "Controle do ácido úrico",
        "estoque": False
    },

    "febuxostate": {
        "preco": "R$ 59,90 (80mg - 30 comprimidos)",
        "indicacao": "Controle do ácido úrico",
        "estoque": True
    },

    "probenecida_500": {
        "preco": "R$ 29,90 (500mg - 30 comprimidos)",
        "indicacao": "Tratamento específico da gota",
        "estoque": False
    },

    "sulfassalazina": {
        "preco": "R$ 39,90 (500mg - 30 comprimidos) - necessário receita",
        "indicacao": "Doenças inflamatórias específicas",
        "estoque": True
    },

    "mesalazina": {
        "preco": "R$ 69,90 (500mg - 30 comprimidos) - necessário receita",
        "indicacao": "Doenças inflamatórias intestinais",
        "estoque": True
    },

    "leflunomida": {
        "preco": "R$ 59,90 (20mg - 30 comprimidos) - necessário receita",
        "indicacao": "Doenças autoimunes específicas",
        "estoque": False
    },

    "etanercepte": {
        "preco": "R$ 599,90 (seringa) - necessário receita",
        "indicacao": "Tratamento de doenças inflamatórias específicas",
        "estoque": False
    },

    "adalimumabe": {
        "preco": "R$ 499,90 (seringa) - necessário receita",
        "indicacao": "Tratamento de doenças inflamatórias específicas",
        "estoque": False
    },

    "infliximabe": {
        "preco": "R$ 799,90 (frasco) - uso hospitalar",
        "indicacao": "Tratamento de doenças inflamatórias específicas",
        "estoque": False
    },

    "tocilizumabe": {
        "preco": "R$ 999,90 (frasco) - uso hospitalar",
        "indicacao": "Tratamento de doenças inflamatórias específicas",
        "estoque": False
    },

    "rituximabe": {
        "preco": "R$ 1999,90 (frasco) - uso hospitalar",
        "indicacao": "Tratamento de doenças específicas",
        "estoque": False
    },

    "clopidogrel_75": {
        "preco": "R$ 24,90 (75mg - 30 comprimidos)",
        "indicacao": "Prevenção de eventos cardiovasculares conforme orientação",
        "estoque": True
    },

    "ticagrelor": {
        "preco": "R$ 69,90 (90mg - 60 comprimidos) - necessário receita",
        "indicacao": "Prevenção de eventos cardiovasculares específicos",
        "estoque": False
    },

    "prasugrel": {
        "preco": "R$ 89,90 (10mg - 30 comprimidos) - necessário receita",
        "indicacao": "Prevenção de eventos cardiovasculares específicos",
        "estoque": True
    },

    "rivaroxabana_20": {
        "preco": "R$ 79,90 (20mg - 28 comprimidos) - necessário receita",
        "indicacao": "Anticoagulação",
        "estoque": False
    },

    "apixabana_5": {
        "preco": "R$ 89,90 (5mg - 60 comprimidos) - necessário receita",
        "indicacao": "Anticoagulação",
        "estoque": True
    },

    "dabigatrana_150": {
        "preco": "R$ 99,90 (150mg - 60 cápsulas) - necessário receita",
        "indicacao": "Anticoagulação",
        "estoque": False
    },

    "varfarina_5": {
        "preco": "R$ 19,90 (5mg - 30 comprimidos) - necessário receita",
        "indicacao": "Anticoagulação",
        "estoque": True
    },

    "heparina_5000": {
        "preco": "R$ 39,90 (frasco) - necessário receita",
        "indicacao": "Anticoagulação em situações específicas",
        "estoque": False
    },

    "enoxaparina_40": {
        "preco": "R$ 89,90 (seringa) - necessário receita",
        "indicacao": "Prevenção de eventos trombóticos",
        "estoque": True
    },

    "digoxina_025": {
        "preco": "R$ 16,90 (0,25mg - 30 comprimidos) - necessário receita",
        "indicacao": "Condições cardíacas específicas",
        "estoque": False
    },

    "amiodarona_200": {
        "preco": "R$ 29,90 (200mg - 30 comprimidos) - necessário receita",
        "indicacao": "Distúrbios específicos do ritmo cardíaco",
        "estoque": True
    },

    "sotalol": {
        "preco": "R$ 29,90 (160mg - 30 comprimidos) - necessário receita",
        "indicacao": "Distúrbios específicos do ritmo cardíaco",
        "estoque": False
    },

    "propafenona": {
        "preco": "R$ 39,90 (150mg - 30 comprimidos) - necessário receita",
        "indicacao": "Distúrbios específicos do ritmo cardíaco",
        "estoque": True
    },

    "diltiazem_60": {
        "preco": "R$ 24,90 (60mg - 30 comprimidos)",
        "indicacao": "Pressão alta e algumas condições cardíacas",
        "estoque": True
    },

    "verapamil_80": {
        "preco": "R$ 19,90 (80mg - 30 comprimidos)",
        "indicacao": "Pressão alta e algumas condições cardíacas",
        "estoque": False
    },

    "isosorbida": {
        "preco": "R$ 18,90 (20mg - 30 comprimidos)",
        "indicacao": "Prevenção de sintomas de angina",
        "estoque": True
    },

    "nitroglicerina": {
        "preco": "R$ 29,90 (comprimidos sublinguais)",
        "indicacao": "Alívio de crises de angina conforme orientação",
        "estoque": False
    },

    "amlodipino_5": {
        "preco": "R$ 12,90 (5mg - 30 comprimidos)",
        "indicacao": "Pressão alta e angina",
        "estoque": True
    },

    "atenolol_50": {
        "preco": "R$ 11,90 (50mg - 30 comprimidos)",
        "indicacao": "Pressão alta e controle cardíaco",
        "estoque": True
    },

    "bisoprolol_5": {
        "preco": "R$ 29,90 (5mg - 30 comprimidos)",
        "indicacao": "Controle cardiovascular",
        "estoque": False
    },

    "nebivolol_5": {
        "preco": "R$ 39,90 (5mg - 30 comprimidos)",
        "indicacao": "Tratamento da pressão alta",
        "estoque": True
    },

    "carvedilol_625": {
        "preco": "R$ 21,90 (6,25mg - 30 comprimidos)",
        "indicacao": "Insuficiência cardíaca e pressão alta",
        "estoque": True
    },

    "carvedilol_125": {
        "preco": "R$ 24,90 (12,5mg - 30 comprimidos)",
        "indicacao": "Insuficiência cardíaca e pressão alta",
        "estoque": False
    },

    "sacubitril_valsartana_50": {
        "preco": "R$ 89,90 (comprimidos) - necessário receita",
        "indicacao": "Tratamento de insuficiência cardíaca",
        "estoque": True
    },

    "empagliflozina_10": {
        "preco": "R$ 89,90 (10mg - 30 comprimidos)",
        "indicacao": "Diabetes tipo 2 e condições cardiovasculares",
        "estoque": False
    },

    "dapagliflozina_10": {
        "preco": "R$ 79,90 (10mg - 30 comprimidos)",
        "indicacao": "Diabetes tipo 2 e condições cardíacas específicas",
        "estoque": True
    },

    "metformina_500": {
        "preco": "R$ 12,90 (500mg - 30 comprimidos)",
        "indicacao": "Controle do diabetes tipo 2",
        "estoque": True
    },

    "metformina_850": {
        "preco": "R$ 16,90 (850mg - 30 comprimidos)",
        "indicacao": "Controle do diabetes tipo 2",
        "estoque": False
    },

    "gliclazida_30": {
        "preco": "R$ 18,90 (30mg - 30 comprimidos)",
        "indicacao": "Controle do diabetes tipo 2",
        "estoque": True
    },

    "glimepirida_2": {
        "preco": "R$ 17,90 (2mg - 30 comprimidos)",
        "indicacao": "Controle do diabetes tipo 2",
        "estoque": True
    },

    "sitagliptina_100": {
        "preco": "R$ 69,90 (100mg - 28 comprimidos)",
        "indicacao": "Controle do diabetes tipo 2",
        "estoque": False
    },

    "pioglitazona_30": {
        "preco": "R$ 24,90 (30mg - 30 comprimidos)",
        "indicacao": "Controle do diabetes tipo 2",
        "estoque": True
    },

    "acarbose_50": {
        "preco": "R$ 29,90 (50mg - 30 comprimidos)",
        "indicacao": "Controle do diabetes tipo 2",
        "estoque": True
    },

    "insulina_nph": {
        "preco": "R$ 39,90 (frasco) - necessário receita",
        "indicacao": "Controle do diabetes",
        "estoque": True
    },

    "insulina_regular": {
        "preco": "R$ 39,90 (frasco) - necessário receita",
        "indicacao": "Controle do diabetes",
        "estoque": False
    },

    "insulina_glargina": {
        "preco": "R$ 99,90 (caneta) - necessário receita",
        "indicacao": "Controle do diabetes",
        "estoque": True
    },

    "insulina_detemir": {
        "preco": "R$ 109,90 (caneta) - necessário receita",
        "indicacao": "Controle do diabetes",
        "estoque": False
    },

    "insulina_asparte": {
        "preco": "R$ 109,90 (caneta) - necessário receita",
        "indicacao": "Controle do diabetes",
        "estoque": True
    },

    "semaglutida": {
        "preco": "R$ 299,90 (caneta) - necessário receita",
        "indicacao": "Tratamento de diabetes e outras indicações específicas",
        "estoque": False
    },

    "liraglutida": {
        "preco": "R$ 279,90 (caneta) - necessário receita",
        "indicacao": "Tratamento de diabetes e outras indicações específicas",
        "estoque": True
    },

    "orlistat_120": {
        "preco": "R$ 49,90 (120mg - cápsulas)",
        "indicacao": "Tratamento da obesidade associado a medidas de estilo de vida",
        "estoque": True
    },

    "sibutramina": {
        "preco": "Medicamento controlado - necessário receita especial",
        "indicacao": "Tratamento específico da obesidade sob acompanhamento médico",
        "estoque": False
    }

}
# ============================================================
# 4. Dados de treinamento
# ============================================================
treinamentos = [
    ("oi", "saudacao"), ("olá", "saudacao"), ("ola", "saudacao"),
    ("bom dia", "saudacao"), ("boa tarde", "saudacao"), ("boa noite", "saudacao"),
    ("tudo bem", "saudacao"), ("e aí", "saudacao"), ("fala", "saudacao"),
    ("oi tudo bem", "saudacao"), ("olá bom dia", "saudacao"),

    ("tchau", "despedida"), ("até logo", "despedida"), ("até mais", "despedida"),
    ("obrigado", "despedida"), ("obrigada", "despedida"), ("valeu", "despedida"),
    ("falou", "despedida"), ("até breve", "despedida"), ("flw", "despedida"),

    ("qual o horário", "horario"), ("que horas abre", "horario"),
    ("que horas fecha", "horario"), ("horário de funcionamento", "horario"),
    ("vocês abrem aos domingos", "horario"), ("está aberto agora", "horario"),
    ("funciona no sábado", "horario"), ("abre domingo", "horario"),
    ("horário de hoje", "horario"), ("quando fecha", "horario"),

    ("vocês tem dipirona", "medicamento"), ("tem paracetamol", "medicamento"),
    ("tem ibuprofeno", "medicamento"), ("tem remédio para dor de cabeça", "medicamento"),
    ("tem antibiótico", "medicamento"), ("tem omeprazol", "medicamento"),
    ("tem losartana", "medicamento"), ("tem rivotril", "medicamento"),
    ("vocês vendem vitamina c", "medicamento"), ("tem pomada", "medicamento"),
    ("tem remédio para gripe", "medicamento"), ("tem algo para dor de garganta", "medicamento"),
    ("tem buscopan", "medicamento"), ("tem loratadina", "medicamento"),
    ("tem amoxicilina", "medicamento"), ("vocês tem esse remédio", "medicamento"),
    ("tem em estoque", "medicamento"),

    ("quanto custa a dipirona", "preco"), ("qual o preço do paracetamol", "preco"),
    ("quanto é o ibuprofeno", "preco"), ("preço da vitamina c", "preco"),
    ("quanto custa o omeprazol", "preco"), ("tem promoção", "preco"),
    ("está barato", "preco"), ("qual o valor", "preco"), ("quanto fica", "preco"),
    ("preço do losartana", "preco"), ("quanto custa", "preco"),

    ("estou com dor de cabeça", "sintoma"), ("tenho febre", "sintoma"),
    ("estou gripado", "sintoma"), ("dor de garganta", "sintoma"),
    ("estou com azia", "sintoma"), ("tenho pressão alta", "sintoma"),
    ("o que tomar para dor", "sintoma"), ("recomenda algo para gripe", "sintoma"),
    ("estou com tosse", "sintoma"), ("tenho alergia", "sintoma"),
    ("dor de barriga", "sintoma"), ("estou com cólica", "sintoma"),
    ("o que é bom para dor", "sintoma"), ("remédio para febre", "sintoma"),

    ("onde fica a farmácia", "localizacao"), ("qual o endereço", "localizacao"),
    ("vocês fazem entrega", "localizacao"), ("tem delivery", "localizacao"),
    ("como chegar", "localizacao"), ("fazem entrega", "localizacao"),
    ("onde fica", "localizacao"),

    ("aceita cartão", "pagamento"), ("pode pagar no pix", "pagamento"),
    ("aceita dinheiro", "pagamento"), ("parcelam", "pagamento"),
    ("tem máquina de cartão", "pagamento"), ("aceita pix", "pagamento"),
    ("forma de pagamento", "pagamento"),
]

# ============================================================
# 5. Pré-processamento
# ============================================================
def preprocessar(texto):
    texto = texto.lower()
    tokens = word_tokenize(texto, language='portuguese')
    tokens_limpos = [t for t in tokens if t.isalnum() and t not in stop_pt]
    return " ".join(tokens_limpos)

# ============================================================
# 6. Treinamento do modelo
# ============================================================
X_treino = [preprocessar(frase) for frase, _ in treinamentos]
y_treino = [intencao for _, intencao in treinamentos]

modelo = Pipeline([
    ('tfidf', TfidfVectorizer(ngram_range=(1, 2))),
    ('clf', MultinomialNB(alpha=0.1))
])
modelo.fit(X_treino, y_treino)

print("--> Modelo treinado com sucesso!")

# ============================================================
# 7. Funções de resposta
# ============================================================
def encontrar_medicamento(texto):
    texto = texto.lower()
    for nome in medicamentos.keys():
        if nome in texto:
            return nome
    return None

def gerar_resposta(mensagem):
    mensagem_lower = mensagem.lower()
    texto_limpo = preprocessar(mensagem)

    med = encontrar_medicamento(mensagem_lower)

    if texto_limpo:
        intencao = modelo.predict([texto_limpo])[0]
        proba = max(modelo.predict_proba([texto_limpo])[0])
        if proba < 0.28:
            intencao = "fallback"
    else:
        intencao = "fallback"

    if med:
        info = medicamentos[med]
        if intencao == "preco" or "quanto" in mensagem_lower or "preço" in mensagem_lower or "custa" in mensagem_lower:
            return f"O {med.title()} custa {info['preco']}. Indicação: {info['indicacao']}."
        elif intencao == "medicamento" or "tem" in mensagem_lower:
            status = "Sim, temos em estoque!" if info['estoque'] else "No momento está em falta."
            return f"{status} {med.title()} → {info['preco']}. Indicação: {info['indicacao']}."
        else:
            return f"Sobre o {med.title()}: {info['preco']}. Indicação: {info['indicacao']}. Deseja mais informações?"

    if intencao == "saudacao":
        return random.choice([
            "Olá! Seja bem-vindo(a) à Farmácia Saúde Total. Como posso te ajudar hoje?",
            "Oi! É um prazer atender você. Em que posso ser útil?",
            "Olá! Sou o assistente da farmácia. Pode me contar o que você precisa?"
        ])

    elif intencao == "despedida":
        return random.choice([
            "Foi um prazer atender você! Cuide-se e volte sempre.",
            "Até logo! Qualquer coisa é só chamar. Tenha um ótimo dia!",
            "Obrigado pela preferência. Até a próxima e se cuide!"
        ])

    elif intencao == "horario":
        return "Nosso horário: Segunda a Sábado das 08h às 20h | Domingos e feriados das 09h às 13h."

    elif intencao == "medicamento":
        return ("Temos vários medicamentos disponíveis. "
                "Me diga o nome do produto (ex: dipirona, paracetamol, omeprazol...) "
                "que eu verifico o preço e a disponibilidade.")

    elif intencao == "preco":
        return ("Posso te informar o preço sim! Me diga o nome do medicamento "
                "(ex: dipirona, ibuprofeno, vitamina c...)")

    elif intencao == "sintoma":
        if "cabeça" in mensagem_lower or "dor de cabeça" in mensagem_lower:
            return "Para dor de cabeça, costumamos indicar Dipirona (R$ 8,90) ou Paracetamol (R$ 7,50). Ambos são de venda livre."
        elif "febre" in mensagem_lower:
            return "Para febre, Dipirona ou Paracetamol são as opções mais comuns. Dipirona: R$ 8,90 | Paracetamol: R$ 7,50."
        elif "azia" in mensagem_lower or "estômago" in mensagem_lower:
            return "Para azia e proteção do estômago, o Omeprazol (R$ 15,90) é muito indicado."
        elif "gripe" in mensagem_lower or "resfriado" in mensagem_lower:
            return "Para gripe e resfriado, além de repouso, Vitamina C (R$ 22,00) e Paracetamol ajudam bastante."
        elif "alergia" in mensagem_lower or "coriza" in mensagem_lower:
            return "Para alergia e coriza, a Loratadina (R$ 11,90) é uma boa opção de venda livre."
        elif "cólica" in mensagem_lower or "barriga" in mensagem_lower:
            return "Para cólica e dor abdominal, o Buscopan (R$ 14,50) costuma aliviar bem."
        else:
            return ("Entendi o seu sintoma. Temos opções de venda livre, mas o ideal é sempre "
                    "consultar um farmacêutico ou médico. Me diga o sintoma com mais detalhes "
                    "que eu tento te indicar algo adequado.")

    elif intencao == "localizacao":
        return ("Estamos na Rua das Flores, 123 - Centro. "
                "Fazemos delivery na região (taxa a partir de R$ 5,00). "
                "Quer o número do WhatsApp para pedir?")

    elif intencao == "pagamento":
        return "Aceitamos Dinheiro, Pix, Débito e Crédito (até 3x sem juros)."

    else:
        return random.choice([
            "Desculpe, não entendi bem. Pode me explicar de outra forma? Posso ajudar com medicamentos, preços, horários, sintomas ou localização.",
            "Não consegui identificar exatamente o que você precisa. Pode reformular? Estou aqui para ajudar com remédios, preços e orientações.",
            "Pode me dar mais detalhes? Posso informar sobre estoque, preços, horários de funcionamento e recomendações de venda livre."
        ])

# ============================================================
# 8. Handlers do Telegram
# ============================================================
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "💊 Olá! Sou o assistente virtual da Farmácia Saúde Total.\n"
        "Como posso te ajudar hoje?\n\n"
        "Posso ajudar com mais alguma coisa?"
    )

async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    mensagem = update.message.text
    resposta = gerar_resposta(mensagem)
    await update.message.reply_text(f"{resposta}\n\nPosso ajudar com mais alguma coisa?")

# ============================================================
# 9. Inicialização do bot
# ============================================================
def main():
    if not TOKEN:
        raise RuntimeError(
            "A variável TELEGRAM_BOT_TOKEN não foi configurada. "
            "Cadastre o token em Environment no Render."
        )
    print("--> Iniciando bot do Telegram...")
    app = Application.builder().token(TOKEN).build()

    app.add_handler(CommandHandler("start", start))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))

    print("--> Bot está online! Pressione Ctrl+C para parar.")
    app.run_polling()

if __name__ == "__main__":
    main()