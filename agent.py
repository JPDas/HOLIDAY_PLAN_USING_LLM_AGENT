from langchain.agents import create_tool_calling_agent # set up the agent
from langchain.agents import AgentExecutor # execute agent
from langchain_openai import ChatOpenAI # call openAI as agent llm
from langchain import hub
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

from tools import *

from dotenv import load_dotenv

load_dotenv()

OPENAI_KEY = os.getenv("OPENAI_KEY")
class HolidayAgent:
    def __init__(self) -> None:
        self.temp = 0
        self.model_name = "gpt-4o-mini"

        #load tools
        self.tools = [wiki_tool, weather_tool, image_tool]

        # Load LLM
        self.llm = ChatOpenAI(model=self.model_name, temperature=self.temp, api_key=OPENAI_KEY)

    def get_prompt(self):

        # With this you let the agent know what its purpose is.
        prompt = ChatPromptTemplate.from_messages([
            ("system", "You are a nice assistant"),
            ("human", "{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad")
        ])

        return prompt
    
    def define_agent(self):

        prompt = self.get_prompt()
        # Define the agent (load the LLM and the list of tools)
        agent = create_tool_calling_agent(llm = self.llm, tools = self.tools, prompt = prompt)
        agent_executor = AgentExecutor(agent=agent, tools=self.tools, verbose=True)

        return agent_executor
    
    def run(self, query):

        agent_executor = self.define_agent()

        return agent_executor.invoke({"input": query})

