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
import pickle
# Streamlit app code

# with open('key.txt', 'r') as file:
#     # Read the contents of the file
#     openai_api_key = file.read()
openai_api_key = st.secrets["openai_api_key"]

@st.cache_resource
def load_data():
    loader = CSVLoader("experiences_update.csv", encoding="utf-8", csv_args={
                'delimiter': ','})
    documents= loader.load()
    # Split the documents into smaller chunks
    text_splitter = CharacterTextSplitter(
        separator=",",
        chunk_size=3000,
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
examples1 = [{'query': 'Exchange on chat, thank you!',
  'answer': 'Sure! We can help!\nWould you like to speak live to someone or exchange on chat?'},
 {'query': 'This is not a new team, we have quarterly team-building events. In the past we have done the terrarium event, a mixology class, etc. I do want something light and fun to get us connected',
  'answer': 'Sure! Tell me what the experience is about -- a kickoff, a new team gathering for first time, etc.\nAnd tell me what you want the event to accomplish (e.g., "people feel bonded/connected", "people feel inspired")'},
 {'query': 'Yes we have!',
  'answer': "Got it. Sounds like you've used Teamraderie before then?\nHere are a few ideas that are popular:\nCoffee/Tea Tasting with French Press"}]
examples2 = [{'query': 'hi,I am looking form something that involves food but belowe $50 per person',
  'answer': '''Hi!
Let me give you two options
https://www.teamraderie.com/experiences/international-snack-box-and-experience/
https://www.teamraderie.com/experiences/brew-break-for-busy-teams/
Both options are $50 per person (including shipping) for the team members in the U.S. (Kit and experience is $45 pp; $5pp is domestic shipping)
'''},
 {'query': 'oh great! thanks so much. is there an option for tea instead of coffee in the snack box?',
  'answer': '''It's  a a special Vietnamese coffee pour over that is used as an example of a regional beverage... Great question. Let me check with a team member
I just confirmed that the standard processes don't offer a tea option for this experience. However, if your team member needs a replacement, let us know and we can replace it.
'''}]
examples3 = [{'query': "let's continue via chat please",
  'answer': 'Sure. Our format is explicitly designed for hybrid.\nI can help you down-select.\nWould you like to speak live or via chat?'},
 {'query': 'Product Managers',
  'answer': 'Great\nFirst, tell me about the team. Engineers? Sales people?'},
 {'query': 'No, more just the team needs something fun for the quarter\nprobably not celebrating but more trying to skill build',
  'answer': 'Next, tell me anything about the context for the gathering. QBR?\nAnd third, tell me if we are celebrating or trying to get them to see problems in a new way.'},
 {'query': 'Larger tech company',
  'answer': 'What kind of company? Or what’s your company name?'},
 {'query': "Hm, maybe skill building was too strict of a parameter. We still want it to have a fun feel, we just thinking booking an activity for purse fun isn't a great use of budget right now, so trying to incorporate some type of teambuilding into it will add impact value",
  'answer': "Ok. Skill building.\nHere are a few options:\n1 - Learning to tell your story better\n2 - Learning to become GREAT at visually representing ideas (makes it easier to collaborate)\n3 - Learning to get better aligned (how and why)\n4 - Learning to become better at 'flexing' how you lead teams\n(Each of the above has experiences that support this goal)\nI'm trying to get a sense of what kinds of skills people are excited to build\nAnother direction is to get people to rethink the 'how' of their goals.\nHere are a few:"},
 {'query': 'how much is the Hostage Rescue Mission activity? This sounds interesting\nHonestly they both sound interesting actually, could we get pricing for both please?',
  'answer': "PERFECT. That really helps.\nAll of our experiences are outcome-based (lead teams to become better at X). And all of them are research-based (so there's science to 'how' we lead them).\nLet me give you some more that will feel 'fun'.\nOh, quick question...what is the total budget you have (or want to stay under)?\nHostage Rescue Mission\nInnovate on a Deadline: Lessons from Hip Hop\nThose are lots of fun -- but they also are clearly purpose-driven"},
 {'query': 'Would any of these activities be able to accommodate a group of 40 though?',
  'answer': "It's $2K for up to 20 or $3K for up to 30.\nI can give you more ideas like that. I think i have a better sense of what you're looking for.\nWould you like me to email you a couple of other ideas and prices?\nThe other one Product Managers like is the NASCAR experience. You do a timed trial on changign wheels/tires (we send you toy NASCARs) and the sport's first pit crew chief leads you in rethinking how you create/define roles.\nThe Hostage Rescue is $2K for 20 and $3K for 30."}]
examples4 = [{'query': '''hi - i am looking for a topic on time management and/or recognition.  thoughts?''',
  'answer': "Sure. I have a few ideas. Is the team global (distributed in APAC and EMEA and NAMER) or just one region?(i'll drop a few ideas and rationale below)"},
  {'query': 'just one region - us and nyc',
  'answer': "Sure. I have a few ideas.\nIs the team global (distributed in APAC and EMEA and NAMER) or just one region?\n(i'll drop a few ideas and rationale below)"},
 {'query': 'thanks - on idea 2 - i did the holiday gifting last year.  looks very similar as i got a book too.  is it about the same?',
  'answer': "Great. Here are some ideas. I'll share three. You can look at them and I'm happy to answer questions.\nIdea 1 - Digital Card Game to Recognize Team Strengths\nIdea 2 - Activate Your 2023 Plan with Gifting\nIdea 3 - The Simplicity Principle: A Team-Based Agreement to Improve Time Management\nSo, those are three ideas. Read through them. I'm here to answer questions you may have on each one. There ought to be a video for each experience (except the third one)."},
 {'query': '👍',
  'answer': 'Yes. The "Holiday Gifting" experience was co-developed with IBM. It was higly-rated by teams -- and IBM asked for a version that could be available specifically to help teams mid-year.\nI\'d guide you towards Idea 1 or Idea 3 -- just so you have more variety.'}]
examples5 = [{'query': '''Related to the options for more than 45 persons the answer are:
1. Will you have any attendees located outside of the US
R=US, Canada, Mexico, Brazil
2. Are you looking for a kit - based experience or an idea - centric?
R=The experience could be with or without kit. We are looking fexamples4 =or an experience that help us to strengthen the teamwork, collaboration and innovation.
3. Do you have a budget per person?
R=No, I don't have any additional budget, I understand the experiences were paid by WW. Is it correct?''',
  'answer': 'Sure, just one more quick question. Are you with IBM?'},
  {'query': 'Yes, this experience is For IBM',
  'answer': 'Perfect. Yes, this experience will be covered by the PO that is open centrally by marketing organization\nHere are some suggestions for your goals:\n1. https://www.teamraderie.com/experiences/virtual-team-building-experience-with-drew-dixon/\n(Drew Dixon is a legendary music producer who consistently created hit records. Drew teaches you to spark and scale innovation, as she guides your team to write a song with hit potential. You will get your song after the experience. It is great for teamwork and innovation)\n2. https://www.teamraderie.com/experiences/let-your-creativity-crush-your-expertise-a-team-exercise/\n(Stanford Professor Kathryn Segovia shows your team how to unlock its creativity thru exercises. Practice on a real work problem and make progress – fast. Also perfect for teamwork and innovation)\n3. https://www.teamraderie.com/experiences/virtual-team-building-the-nascar-experience/\n(Led by a legendary Nascar pit crew coach, you will learn how to collaborate better and optimize for team vs. individual outcome. This experience includes a kit (Nascar kit) )\nLet me know your thoughts about these three suggestions. Do any of these look good?'},
 {'query': 'Thank you very much. I will check with my boss and team and I will be back whit you as soon as possible',
  'answer': 'Perfect! Sounds great, thank you'}]


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
prefix = """The following are exerpts from conversations with a teamraderie
assistant. The assistant is professional, producing answers and recommendations to the users query. Here are some
examples:
"""
# and the suffix our user input and output indicator

suffix = """
you are a member of teamraderie and your job is to Use the above examples ,the chat history (delimited by <hs></hs>) and following context (delimited by <ctx></ctx>) to answer the question ,if you are asked to make recommendations, pls make no more than three recommendations with experience title, corresponding links  and reasons. 
Try to reommend experiecne with ranking larger than 8, but don't mention ranking in your response.
You only give the cost of the experiences when you are asked, the cost of experiences is calucualted by addding flat_fee_price and base_price per person together. pls provedie cost information in full detail including the price for different number of people according to tiering price structure.
if you are asked about whether the experience is in-office, hybrid, or remote (available in other country), pls anser the questions and provide link to photos of corrsponding team.
pls pay attention to historic conversation, if the customer asks information about recommended experience, just answer the questions regrading recommended experienc , don't recommend new experience.
------
<ctx>
{context}
</ctx>
------
<hs>
{history}
</hs>
------
User:{question}
teamraderie assistant:
"""

# now create the few shot prompt template
few_shot_prompt_template = FewShotPromptTemplate(
    examples= examples1 + examples2 + examples3 + examples4 + examples5,
    example_prompt=example_prompt,
    prefix=prefix,
    suffix=suffix,
    input_variables=["history", "context", "question"],
    example_separator="\n\n"
)

# chat completion llm
llm = ChatOpenAI(
    openai_api_key= openai_api_key,
    model_name='gpt-4',
    temperature=0.0
)


example_selector = SemanticSimilarityExampleSelector.from_examples(
    # This is the list of examples available to select from.
    examples1 + examples2 + examples3 + examples4 + examples5,
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
retriever=docsearch.as_retriever(),
chain_type_kwargs={
    "verbose": True,
    "prompt": similar_prompt ,
    "memory": ConversationBufferMemory(
        memory_key="history",
        input_key="question")
        k=5
        ,})
tools = [
    Tool(
        name="Document tool",
        func=qa.run,
        description="useful for when you need to answer questions."
    ),
]

system_msg = "You are a helpful teamraderie assistant."
agent = ConversationalChatAgent.from_llm_and_tools(
            llm=llm,
            tools=tools,
            system_message=system_msg
        )
chat_agent = AgentExecutor.from_agent_and_tools(
            agent=agent, tools=tools, verbose=True, memory=ConversationBufferMemory(memory_key="history",input_key="question", k=5,
                                                                                    return_messages=True)
        )

# st.title("Web Query API")

# question = st.chat_input("Say something")
# if question:
#     # Set the OpenAI API key
#     # Create an instance of the WebQuery clas

#     # Ask the question
#     answer = qa.run({"query": question})
#    st.write("Answer:", answer)
@st.cache_resource
def get_answer(message):
    return qa.run({"query": message["content"]})

def main():
    st.title("MikaelaGPT")
    st.image("https://i.imgur.com/qnE2MRG.png",width= 200)
    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    # if st.button("Start chat"):
    # # Clear values from *all* all in-memory and on-disk data caches:
    # # i.e. clear values from both square and cube
    #     st.cache_data.clear()

    if prompt := st.chat_input("What is up?"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            full_response = ""
            for message in st.session_state.messages:
                response = get_answer(message)
            full_response += response
            message_placeholder.markdown(full_response)
        st.session_state.messages.append({"role": "assistant", "content": full_response})
if __name__ == "__main__":
    main()
