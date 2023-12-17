import { ConversationChain } from 'langchain/chains';
import { ChatOpenAI } from 'langchain/chat_models/openai';
import {
  ChatPromptTemplate,
  SystemMessagePromptTemplate,
  HumanMessagePromptTemplate,
  MessagesPlaceholder,
} from 'langchain/prompts';
import { BufferMemory } from 'langchain/memory';
import axios from 'axios';
import { OpenAI } from "langchain/llms/openai";
import { SqlDatabase } from "langchain/sql_db";
import { createSqlAgent, SqlToolkit } from "langchain/agents/toolkits/sql";
import { DataSource } from "typeorm";
/** This example uses Chinook database, which is a sample database available for SQL Server, Oracle, MySQL, etc.
 * To set it up follow the instructions on https://database.guide/2-sample-databases-sqlite/, placing the .db file
 * in the examples folder.
 */
export const run = async (prompt) => {
  const url = 'http://127.0.0.1:5000/run_agent'
  axios.post(url, { prompt })
    .then(function (response) {
      console.log(response);
      return response.data.response;
    })
    .catch(function (error) {
      console.log(error);
    });
};
const memory = new BufferMemory({
  returnMessages: true,
  memoryKey: 'history',
});
export const davinci = async (prompt, key, gptVersion) => {
  const chatPrompt = ChatPromptTemplate.fromMessages([
    SystemMessagePromptTemplate.fromTemplate(
      'Phía dưới là cuộc trò chuyện thân thiện giữa người và một nhân viên tư vấn máy tính. Nhân viên tư vấn máy tính rất nhiệt tình và cung cấp rất nhiều thông tin cụ thể từ ngữ cảnh của nó. Nếu nhân viên không biết câu trả lời cho một câu hỏi nào đó, nó sẽ trung thực nói rằng nó không biết. Hãy chú ý hỏi tôi những câu liên quan đến số tiền tôi có, nhu cầu sử dụng, phần mềm tôi làm và không hỏi quá vồ vập.'
    ),
    new MessagesPlaceholder('history'),
    HumanMessagePromptTemplate.fromTemplate('{input}'),
  ]);
  const model = new ChatOpenAI({
    openAIApiKey: key,
    model: gptVersion,
    temperature: 0.,
  });
  const chain = new ConversationChain({
    memory: memory,
    prompt: chatPrompt,
    llm: model,
    verbose: true,
  });
  const response = await chain.call({ input: prompt });
  console.log(response);
  var trigger = true;
  return [response.response, trigger];
};