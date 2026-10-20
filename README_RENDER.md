# Chatbot de Farmácia no Render (Telegram)

## Arquivos
- `bot.py`: seu bot original, adaptado para ler o token por variável de ambiente.
- `requirements.txt`: bibliotecas Python.
- `render.yaml`: configuração de Background Worker do Render.
- `.gitignore`: evita enviar ambientes virtuais e arquivos `.env`.

## Publicar no Render
1. Extraia este ZIP e envie os arquivos para um repositório GitHub.
2. No Render, escolha **New +** > **Blueprint** e conecte o repositório que contém `render.yaml`.
   Alternativamente, crie um **Background Worker** manualmente com:
   - Runtime: Python
   - Build Command: `pip install -r requirements.txt`
   - Start Command: `python bot.py`
3. Nas variáveis de ambiente do serviço, adicione `TELEGRAM_BOT_TOKEN` com o token novo obtido no `@BotFather`.
4. Faça o deploy e confira os logs. Deve aparecer `Bot está online`.
5. No Telegram, envie `/start` ao bot.

## Importante sobre o token
O código que você enviou continha um token de Telegram diretamente no arquivo. Considere-o exposto:
revogue-o/ gere um novo com o `@BotFather` e cadastre o novo token apenas em Environment no Render.
Não coloque o token no código nem no GitHub.

## Observações
- Este é um **Background Worker**, não um site: o bot usa polling do Telegram e não precisa de uma URL pública.
- Os recursos NLTK são baixados na inicialização pelo próprio código. A primeira inicialização pode demorar.
- Os preços, estoques, endereço e informações comerciais do código são dados de protótipo; revise-os antes de uso real.
- Como o bot fala sobre medicamentos, não use as respostas sobre sintomas como diagnóstico ou prescrição. Recomende avaliação de farmacêutico/médico e confirme as exigências regulatórias antes de colocar em produção.
