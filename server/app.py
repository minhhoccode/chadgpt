import os
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import TextLoader, UnstructuredPDFLoader, YoutubeLoader, PyPDFLoader
from langchain.embeddings import HuggingFaceEmbeddings, OpenAIEmbeddings
from langchain.indexes import VectorstoreIndexCreator
from langchain.llms import OpenAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.llms import GPT4All
from pdf2image import convert_from_path

from flask import Flask, request, jsonify
from flask_cors import CORS, cross_origin
from langchain.agents import create_sql_agent
from langchain.agents.agent_toolkits import SQLDatabaseToolkit
from langchain.agents.agent_types import AgentType
from langchain.llms.openai import OpenAI
from langchain.sql_database import SQLDatabase
import os
from langchain.memory import ChatMessageHistory
from langchain.chat_models import ChatOpenAI
from langchain.prompts import (
    ChatPromptTemplate,
    MessagesPlaceholder,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain.chains import LLMChain
from langchain.memory import ConversationBufferMemory
from langchain import PromptTemplate


os.environ["OPENAI_API_KEY"] = "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"
app = Flask(__name__)
cors = CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'

db = SQLDatabase.from_uri("mysql+pymysql://root:@localhost/Hackathon")
toolkit = SQLDatabaseToolkit(db=db, llm=OpenAI(temperature=0))

agent_executor = create_sql_agent(
    llm=OpenAI(temperature=0),
    toolkit=toolkit,
    verbose=True,
    agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
)

@app.route('/davinci', methods=['POST'])
def get_davinci_response():
    try:
        data = request.get_json()
        prompt = data['prompt']
        print("prompt")
        print(prompt)
        Chatprompt = ChatPromptTemplate(
            messages=[
                SystemMessagePromptTemplate.from_template(
                    "You are a nice chatbot having a conversation with a human."
                ),
                MessagesPlaceholder(variable_name="history"),
                HumanMessagePromptTemplate.from_template("{input}")
            ]
        )
        memory = ConversationBufferMemory(memory_key="history", return_messages=True)
        llm = ChatOpenAI()
        conversation = LLMChain(
            llm=llm,
            prompt=Chatprompt,
            verbose=True,
            memory=memory
        )
        response = conversation({"input": str(prompt)})
        print("response")
        print(response)
        return jsonify({"response": response['text']})

    except Exception as e:
        return jsonify({'error': str(e)})
    
@app.route('/run_agent', methods=['POST'])
def run_agent():
    try:
        data = request.get_json()
        prompt = data.get('prompt', '')
        result = agent_executor.run(prompt)
        return jsonify({"result": result})
    except Exception as e:
        return jsonify({"error": str(e)})

from langchain.document_loaders.csv_loader import CSVLoader
@app.route('/vectorize', methods=['POST'])
def vectorize():
    try:
        query = request.get_json()['query']
        loader = CSVLoader(file_path="/Users/vungocminh/Desktop/laptop_tgdd (1) - laptop_tgdd (1).csv")
        data = loader.load()
        MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
        hf_embeddings = HuggingFaceEmbeddings(model_name=MODEL_NAME)
        db1 = Chroma.from_documents(data, hf_embeddings, persist_directory="db2")
        custom_prompt_template = """
        Chỉ sử dụng các thông tin trên
        Hãy đóng vai bạn là 1 tư vấn viên bán máy tính xách tay.
        Nhưng hãy cố gắng thuyết phục khách hàng mua sản phẩm phù hợp nhất với nhu cầu của họ.

        Nếu tôi đã đề cập đến một dòng sản phẩm nào, hãy tư vấn.
        Nếu như tôi chưa biết lựa chọn, hãy hỏi tôi về các nhu cầu của tôi, thường làm việc, giải trí trên phần mềm nào. 

        Lưu ý khi hỏi không được quá vồ vập và nhiều câu hỏi cùng lúc. gợi ý từ 2-3 máy phù hợp nhất và có thể không gợi ý máy nào

        Nếu có thì yêu cầu cấu hình ra sao, ngoại hình, màu sắc , giá cả ra sao, thương hiệu nào,...
        Đặc biệt lưu ý, bạn hãy hỏi tôi về số vốn tôi có và bất kì thắc mắc nào, hãy hỏi lại tôi ngay lập tức.
        Tương ứng với mỗi sản phẩm mà người dùng phân vân, bạn hãy so sánh các tính năng chúng và đưa vào trong một bảng.

        Nếu bạn không biết câu trả lời, chỉ cần nói rằng bạn không biết, đừng cố bịa ra câu trả lời.
        Hoặc không có sản phẩm đủ tiêu chí thì hãy nói rằng không có sản phẩm nào phù hợp với nhu cầu của tôi.

        Context: {context}
        Question: {question}
        """
        model = OpenAI(temperature=0, model_name="gpt-3.5-turbo")

        def set_custom_prompt():
            prompt = PromptTemplate(template=custom_prompt_template,
                                    input_variables=['context', 'question'])
            return prompt
        prompt = set_custom_prompt()
        chain = RetrievalQA.from_chain_type(
            llm=model,
            chain_type="stuff",
            retriever=db1.as_retriever(search_kwargs={"k": 2}),
            chain_type_kwargs={'prompt': prompt}
        )
        chain.run(query)
        return jsonify({"result": chain.result})
if __name__ == '__main__':
    app.run(port=5000)
