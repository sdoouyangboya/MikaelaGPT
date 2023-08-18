import streamlit as st
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings
from langchain.document_loaders.csv_loader import CSVLoader
from langchain.prompts.example_selector import SemanticSimilarityExampleSelector
from langchain.chat_models import ChatOpenAI
from langchain.chains.conversation.memory import ConversationBufferWindowMemory
from langchain.chains import RetrievalQA
from langchain.memory import ConversationBufferMemory
from langchain import FewShotPromptTemplate
from langchain import PromptTemplate
from langchain.agents import ConversationalChatAgent, Tool, AgentExecutor
from langchain.schema import (
      SystemMessage,
      HumanMessage,
      AIMessage
  )
import pandas as pd
import pickle

# Streamlit app code

# with open('key.txt', 'r') as file:
#     # Read the contents of the file
#     openai_api_key = file.read()
openai_api_key = st.secrets["openai_api_key"]

@st.cache_resource
def load_data():
    loader = CSVLoader("experiences_update_4.csv", encoding="utf-8", csv_args={
                'delimiter': ','})
    documents= loader.load()
    # Split the documents into smaller chunks
    text_splitter = CharacterTextSplitter(
        separator=",",
        chunk_size=6000,
        chunk_overlap=100,
        length_function=len,
    )
    texts = text_splitter.split_documents(documents)

    # Embed the texts
    embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
    docsearch = FAISS.from_documents(texts, embeddings)
    return docsearch 
docsearch = load_data()
# with open("faiss_store.pkl", "rb") as f:
#     docsearch  = pickle.load(f)
example = [{'query':'''I want experiences suitable for teamsize of 10''', 'answer': 'Beyond what you’ve shared, are there any other outcomes you’d like to achieve for your team'},
           {'query':''' I want experience with price below 100 per person''', 'answer': 'Beyond what you’ve shared, are there any other outcomes you’d like to achieve for your team'},
           {'query':'''I want  experience for global team?''', 'answer': 'Beyond what you’ve shared, are there any other outcomes you’d like to achieve for your team'},
    {'query': '''I want experiences that can improve the connection of my team''',
  'answer': ''' I recommend the experience "Uncover the Power of Asking Better Questions for Connection" with Taylor Buonocore-Guthrie as the host. This experience focuses on building team trust and developing crucial communication skills. By learning how to ask better questions, your team can improve their connection and understanding of each other. You can find more information about this experience [here](https://www.teamraderie.com/experiences/uncover-the-power-of-asking-better-questions-for-connection/).

Another recommendation is the experience "Building Empathy and Listening Skills through Improv" with Pam Mcleod as the host. This experience uses improv techniques to enhance empathy and improve communication within the team. It's a fun and interactive way to strengthen team connections. You can learn more about this experience [here](https://www.teamraderie.com/experiences/team-improv-experience-for-connection-and-empathy/).

Lastly, I suggest the experience "Purposeful Painting Exercise for Team Unity" with a professional artist as the host. This creative experience allows your team to relax, bond, and paint something meaningful together. It's a great way to foster connection and appreciation among team members. You can find more details about this experience [here](https://www.teamraderie.com/experiences/virtual-team-building-with-art/).

I hope these recommendations help improve your team's connection! Let me know if you have any other questions or if there's anything else I can assist you with.'''},
  {'query': 'I have teammaneber located outside US, are these experiences available remotely?', 
  'answer' : """yes all the experiencess can be participated remotely,here are examples of team photos: In Office:
https://www.teamraderie.com/wp-content/uploads/2023/07/InOffice2.jpg
https://www.teamraderie.com/wp-content/uploads/2023/07/InOffice1.jpg

Hybrid:
https://www.teamraderie.com/wp-content/uploads/2023/07/Hybrid2.jpg
https://www.teamraderie.com/wp-content/uploads/2023/07/Hybrid1.jpg

Remote:
https://www.teamraderie.com/wp-content/uploads/2023/07/Remote1.jpg
https://www.teamraderie.com/wp-content/uploads/2023/07/Remote2.jpg"""},
           {'query': '''I want experiences that can improve the connection of my team with size of 20''',
  'answer': ''' I recommend the experience "Uncover the Power of Asking Better Questions for Connection" with Taylor Buonocore-Guthrie as the host. This experience focuses on building team trust and developing crucial communication skills. By learning how to ask better questions, your team can improve their connection and understanding of each other. You can find more information about this experience [here](https://www.teamraderie.com/experiences/uncover-the-power-of-asking-better-questions-for-connection/).

Another recommendation is the experience "Building Empathy and Listening Skills through Improv" with Pam Mcleod as the host. This experience uses improv techniques to enhance empathy and improve communication within the team. It's a fun and interactive way to strengthen team connections. You can learn more about this experience [here](https://www.teamraderie.com/experiences/team-improv-experience-for-connection-and-empathy/).

Lastly, I suggest the experience "Purposeful Painting Exercise for Team Unity" with a professional artist as the host. This creative experience allows your team to relax, bond, and paint something meaningful together. It's a great way to foster connection and appreciation among team members. You can find more details about this experience [here](https://www.teamraderie.com/experiences/virtual-team-building-with-art/).

I hope these recommendations help improve your team's connection! Let me know if you have any other questions or if there's anything else I can assist you with.'''},
  {'query': 'I have teammaneber located outside US, are these experiences available remotely?', 
  'answer' : """yes all the experiencess can be participated remotely,here are examples of team photos: In Office:
https://www.teamraderie.com/wp-content/uploads/2023/07/InOffice2.jpg
https://www.teamraderie.com/wp-content/uploads/2023/07/InOffice1.jpg

Hybrid:
https://www.teamraderie.com/wp-content/uploads/2023/07/Hybrid2.jpg
https://www.teamraderie.com/wp-content/uploads/2023/07/Hybrid1.jpg

Remote:
https://www.teamraderie.com/wp-content/uploads/2023/07/Remote1.jpg
https://www.teamraderie.com/wp-content/uploads/2023/07/Remote2.jpg"""},
{'query': '''I want experiences that can improve the connection of for a global team ''',
  'answer': ''' I recommend the experience "Uncover the Power of Asking Better Questions for Connection" with Taylor Buonocore-Guthrie as the host. This experience focuses on building team trust and developing crucial communication skills. By learning how to ask better questions, your team can improve their connection and understanding of each other. You can find more information about this experience [here](https://www.teamraderie.com/experiences/uncover-the-power-of-asking-better-questions-for-connection/).

Another recommendation is the experience "Building Empathy and Listening Skills through Improv" with Pam Mcleod as the host. This experience uses improv techniques to enhance empathy and improve communication within the team. It's a fun and interactive way to strengthen team connections. You can learn more about this experience [here](https://www.teamraderie.com/experiences/team-improv-experience-for-connection-and-empathy/).

Lastly, I suggest the experience "Purposeful Painting Exercise for Team Unity" with a professional artist as the host. This creative experience allows your team to relax, bond, and paint something meaningful together. It's a great way to foster connection and appreciation among team members. You can find more details about this experience [here](https://www.teamraderie.com/experiences/virtual-team-building-with-art/).

I hope these recommendations help improve your team's connection! Let me know if you have any other questions or if there's anything else I can assist you with.'''},
  {'query': 'I have teammaneber located outside US, are these experiences available remotely?', 
  'answer' : """yes all the experiencess can be participated remotely,here are examples of team photos: In Office:
https://www.teamraderie.com/wp-content/uploads/2023/07/InOffice2.jpg
https://www.teamraderie.com/wp-content/uploads/2023/07/InOffice1.jpg

Hybrid:
https://www.teamraderie.com/wp-content/uploads/2023/07/Hybrid2.jpg
https://www.teamraderie.com/wp-content/uploads/2023/07/Hybrid1.jpg

Remote:
https://www.teamraderie.com/wp-content/uploads/2023/07/Remote1.jpg
https://www.teamraderie.com/wp-content/uploads/2023/07/Remote2.jpg"""}]


# create a example template
example_template = """
User: {query}
AI: {answer}
"""

# create a prompt example from above template
example_prompt = PromptTemplate(
    input_variables=["query", "answer"],
    template=example_template
)

# now break our previous prompt into a prefix and suffix
# the prefix is our instructions
prefix = """The following are example conversation bewteen teamraderie
assistant and customer :
"""
# and the suffix our user input and output indicator

suffix = """
here is the history coversation: {history}
here is the context 
------
<ctx>
{context}
</ctx>
**You are the TeamRaderie Assistant. Your primary role is to provide reponse to customers questions and address their inquiries based on provided example and  context (<ctx></ctx>). Here are your guidelines:**
1. **Recommendations**: If prompted for recommendations, offer up to three that align closely with the customer's needs. 
   * ask the customer about their desired outcome or goal they hope to achieve from the experience if they don't specify any areas for improvement
   * Present the title of each experience.
   * Include direct links to the experiences.
   * State the reasons for each recommendation.

2. **Ranking**: Aim to recommend experiences with a ranking of 8 or higher. However, do not disclose the ranking in your response.

3. **Pricing**: Only provide the cost details of experiences when specifically requested. 
   * The total cost of an experience is calculated by combining the flat_fee_price with the base_price per person.
   * Display cost details in a bulleted format.
   * Structure the cost according to tiered pricing.

4. **Location Details**:
   * If a user asks about the location of an experience (in-office, hybrid, remote) or whether the experience available globally, provide appropriate links according to links to photos of In-office team ,Hybrid team, Remote team:
     * For 'In-office' experiences not available worldwide, give links to photos of the In-office team.
     * If the experience is global, provide links to photos of the Hybrid or Remote teams.
   * Limit the links to a maximum of two.

5. **Communication**:
   * If you're uncertain about your response, feel free to request more information from the customer.
   * Always refer to TeamRaderie using she/her pronouns.

6. **Empathy**: While offering answers, ensure that your responses are thoughtful, logical, and empathetic.
here is the Customer's Question you need to respond according to above guideline: {question}
"""

# now create the few shot prompt template
few_shot_prompt_template = FewShotPromptTemplate(
    examples= example
    example_prompt=example_prompt,
    prefix=prefix,
    suffix=suffix,
    input_variables=["history", "context", "question"],
    example_separator="\n\n"
)

# chat completion llm
llm = ChatOpenAI(
    openai_api_key= openai_api_key,
    model_name='gpt-3.5-turbo-16k',
    temperature=0.5
)


example_selector = SemanticSimilarityExampleSelector.from_examples(
    # This is the list of examples available to select from.
    example
    # This is the embedding class used to produce embeddings which are used to measure semantic similarity.
    OpenAIEmbeddings(openai_api_key= openai_api_key),
    # This is the VectorStore class that is used to store the embeddings and do a similarity search over.
    FAISS,
    # This is the number of examples to produce.
    k= 2
)
similar_prompt = FewShotPromptTemplate(
    # We provide an ExampleSelector instead of examples.
    example_selector=example_selector,
    example_prompt=example_prompt,
    prefix=prefix,
    suffix=suffix,
    input_variables=["history", "context", "question"],
    example_separator="\n"
)


qa = RetrievalQA.from_chain_type(
llm=llm,
chain_type="stuff",
retriever=docsearch.as_retriever(search_type="mmr", search_kwargs={"k":8}),
chain_type_kwargs={
    "verbose": True,
    "prompt": similar_prompt ,
    "memory": ConversationBufferMemory(
        memory_key="history",
        input_key="question"
        )
        })
        

@st.cache_resource
def typing(message):
    history = qa.combine_documents_chain.memory.chat_memory.messages
    history =''.join([x.content + x.content for x in history ])
    question = message["content"]
    if history:
        h= ' based on chat history '+ history +'here is the question:' + question + ', list a new question specifying what is the customer asking about specifically, +"incorporating specific titles of recommended experienes into the questions if you can'

        messages = [

            HumanMessage(content = h)
        ]
        summary= llm(messages)
        return qa.run({"query": summary.content})
    else:
        return qa.run({"query": question})

    # def extract_ai_message(conversation):
    #     split_text = conversation.split('\n')
    #     for i in range(len(split_text)):
    #         if split_text[i].startswith('AI:'):
    #             return split_text[i].replace('AI: ', '')
    #     return "No AI message found."
    # return extract_ai_message(answer)

def main():
    st.title("MikaelaGPT")
    st.image("https://i.imgur.com/qnE2MRG.png",width= 200)
    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("What is up?"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            full_response = ""
            for message in st.session_state.messages:
                response = typing(message)
            full_response += response
            message_placeholder.markdown(full_response)
        st.session_state.messages.append({"role": "assistant", "content": full_response})
if __name__ == "__main__":
    main()
