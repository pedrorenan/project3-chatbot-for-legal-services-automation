from langchain.agents import AgentType, initialize_agent, load_tools
from langchain.callbacks import StreamlitCallbackHandler
import streamlit as st
from pinecone import Pinecone
import os
import io
import re
import fitz
import pytesseract
from PIL import Image
import pandas as pd
from langchain_experimental.tools.python.tool import PythonAstREPLTool
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
from langchain.vectorstores import Pinecone as LangChainPinecone
from langchain.chains import RetrievalQA
from langchain.agents import Tool
from langchain.prompts import PromptTemplate

from dotenv import load_dotenv
import uuid

load_dotenv()

os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_ENDPOINT"]="https://api.smith.langchain.com"
os.environ["LANGCHAIN_PROJECT"]="project3"


pc = Pinecone(api_key=os.environ.get("PINECONE_API_KEY"))
index_name = "deliberations-services"
index = pc.Index(index_name)


embeddings = HuggingFaceEmbeddings(model_name='neuralmind/bert-base-portuguese-cased')

vectorstore = LangChainPinecone(
    index=index,
    embedding=embeddings,
    text_key="description"
)

llm = ChatOpenAI(
    openai_api_key=os.environ.get("OPENAI_API_KEY"),
    model_name='gpt-3.5-turbo',
    temperature=0.0
)

prompt = PromptTemplate.from_template(
    """
    Use as seguintes informações de contexto para responder quais são as deliberações e serviços necessários para atender ao ato societário.
    Devolva uma lista de deliberações e serviços que devem ser realizados para atender ao ato societário.
    Devolva a cidade onde os serão prestados. 
    Se você não souber a resposta, apenas diga que não sabe, não tente inventar uma resposta.
    
    {context}
    
    Questão: {question}
    Resposta útil:
    """
)

deliberations = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=vectorstore.as_retriever(),
    chain_type_kwargs={"prompt": prompt}
)

def clean_text(text):
    """
    Clean the extracted text by removing unnecessary characters and formatting.
    
    Args:
    - text (str): The extracted text.
    
    Returns:
    - str: The cleaned text.
    """
    # Remove multiple spaces and replace with a single space
    text = re.sub(r'\s+', ' ', text)
    # Remove line breaks and tabs
    text = text.replace('\n', ' ').replace('\t', ' ')
    # Remove non-alphanumeric characters and basic punctuation
    text = re.sub(r'[^a-zA-Z0-9áàâãéèêíïóôõöúçñÁÀÂÃÉÈÍÏÓÔÕÖÚÇÑ.,;!?()\-\'" ]', '', text)
    return text

def extract_text_from_pdf(pdf_path):
    """
    Extract text from a PDF file using PyMuPDF (fitz).
    Use OCR for scanned pages.
    
    Args:
    - pdf_path (str): Path to the PDF file.
    
    Returns:
    - str: Extracted text.
    """
    text = ""
    # test if pdf_path is a valid path
    if not os.path.exists(pdf_path):
        return "Não foi possível extrair texto do arquivo PDF. Por favor, verifique se o arquivo está correto."
    document = fitz.open(pdf_path)
    for page_num in range(len(document)):
        page = document.load_page(page_num)
        text += page.get_text()
        if not text.strip():  # If no text is extracted, use OCR
            pix = page.get_pixmap()
            img = Image.open(io.BytesIO(pix.tobytes()))
            text += pytesseract.image_to_string(img)
    cleaned_text = clean_text(text)
    return cleaned_text
 
        

class PDFExtractionTool:
    def __init__(self, extraction_function):
        self.extraction_function = extraction_function
    
    def run(self, pdf_path):
        return self.extraction_function(pdf_path)
    
pdf_tool = PDFExtractionTool(extract_text_from_pdf)

df = pd.read_csv("data/prices.csv")
python = PythonAstREPLTool(locals={"df": df}, verbose=True) 


tools = [
    Tool(
        name = "Deliberations",
        func=deliberations.run,
        description="""
        Use para qualquer todas as questões que envolverem assuntos relacionados a empresas e atos societários.
        """
    ),
    Tool(
        name="PDF Extraction",
        func=pdf_tool.run,
        description="""
        Use esta ferramenta para extrair texto de arquivos PDF.
        Só execute essa ferramenta se você tiver um arquivo PDF para extrair texto.
        Nunca use mais de uma vez.
        """
    ),
    Tool(
        name = "Services prices",
        func=python.run,
        description = f"""
        Essa ferramenta só dever ser utilizada depois que a ferramenta 'Deliberations' for utilizada.
        Use essa ferramenta para responder perguntas sobre os preços dos serviços que estão armazenados em um dataframe pandas 'df'.
        Execute python pandas operations em 'df' para ajudá-lo a obter a resposta certa.
        Nunca passe o input diretamente para a função 'run', sempre faça a manipulação necessária antes de passar para a função 'run'.
        'df' tem as seguintes colunas: {df.columns}
        Os serviços podem ser consultados por seus nomes na coluna 'Serviços'.
        Você pode pesquisar os serviços utilizando df[df['Serviços'].str.contains('exemplo de serviços')]
        A coluna 'Valor' contém o valor dos serviços. Caso esteja em branco, considere o valor como 100.
        A coluna 'Atributo' contém o nome da cidade onde o serviço será prestado.
        Sempre apresente os valores no formato R$ 0,00.
        """
    ),
]

agent_kwargs = {'prefix': f'''
                Você é um especialista em atos societários e trabalha para a empresa A2 Soluções inteligentes.
                Qualquer recomendação que você fizer deverá ser para profissionais da sua empresa que é a A2 Soluções inteligentes.
                Não mande consultar qualquer outro profissional que não seja da A2 Soluções inteligentes.
                Sempre responsa em Português do Brasil.
                '''}


agent = initialize_agent(
    tools, 
    llm, 
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, 
    verbose=True,
    agent_kwargs=agent_kwargs,
    handle_parsing_errors=True,
    max_iterations=3,
    early_stopping_method='generate',
)

st.info("Quais são os serviços necessários para abrir uma empresa em Goiânia e quanto eles custam?")
st.info("O que é necessário para abrir uma empresa ?")
uploaded_file = st.file_uploader("Envie seu ato societário", type="pdf")
if prompt := st.chat_input() or uploaded_file:
    if uploaded_file:
        filename = f"{str(uuid.uuid4())}.pdf"
        with open(filename, "wb") as f:
            f.write(uploaded_file.read())
        prompt = filename
    
    st.chat_message("user").write(prompt)
    with st.chat_message("assistant"):
        st_callback = StreamlitCallbackHandler(st.container())
        response = agent.run(prompt, callbacks=[st_callback])
        st.write(response)