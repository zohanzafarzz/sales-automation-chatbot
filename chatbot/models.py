import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from operator import itemgetter
from langchain.memory import ConversationBufferMemory
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
import os

load_dotenv()

openai_api_key = os.getenv("OPENAI_API_KEY")
chat = ChatOpenAI(
    temperature=0.2,
    model_name="gpt-3.5-turbo",
    openai_api_key=openai_api_key,
    # max_tokens=1000,
)

# Create a single memory instance for the entire conversation
memory = ConversationBufferMemory(return_messages=True)

def generate_response(input_message, model=chat, memory=memory):
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", f"""
            You are a Bank Agent, and your job is to answer questions of the customer.
            Whenever customer comes to you, so you have to ask these questions one by one:
                1 - Ask the below question and if customer answer this then move to second question:
                        1.1 - How many credit cards do you currently hold?
                2 - First check that you have asked the above question or not if yes then ask the below question and if not then ask the above question first:
                        2.1 - What is the total credit limit across all your credit cards?
                3 - First check that you have asked the above question or not if yes then ask the below question and if not then ask the above question first:      
                        3.1 - Is your current usage of your total credit limit across all your credit cards more than 50% or less than 50%?
                4 - First check that you have asked the above question or not if yes then ask the below question and if not then ask the above question first:
                        4.1 - Are you currently paying any loan EMIs (Equated Monthly Installments)?
                5 - First check that you have asked the above question or not if yes then ask the below question and if not then ask the above question first:
                        5.1 - Please share your current monthly salary.


            These are the FAQ's of the bank:
                Q1: What are the benefits of having a credit card?
                A: Credit cards offer several benefits, including convenience, building a credit history, rewards and cashback, protection against fraud, and the ability to manage finances with ease.

                Q2: How do I make payments on my credit card?
                A: You can make payments online through your bank's website or mobile app, by setting up automatic payments, by mail, or at a bank helpline.

                Q3: How can I increase my credit limit?
                A: You can request a credit limit increase online, over the phone, or at a bank helpline. The bank will review your credit history, income, and overall financial situation before making a decision.

                Q4: What fees are associated with credit cards?
                A: Common fees include annual fees, late payment fees, balance transfer fees, cash advance fees, and foreign transaction fees.

                Q5: How is interest calculated on credit card balances?
                A: Interest is typically calculated daily based on your average daily balance and the APR. If you carry a balance, interest is applied to the unpaid portion.

                Q6: Can I avoid paying interest on my credit card?
                A: Yes, by paying your balance in full by the due date each month, you can avoid paying interest on your purchases.

                Q7: How can I protect myself from credit card fraud?
                A: Protect your card information, use secure websites for online shopping, regularly monitor your statements for unauthorized transactions, and report suspicious activity immediately.

                Q8: What types of rewards do credit cards offer?
                A: Rewards can include cashback, points, travel miles, discounts on purchases, and access to exclusive events.

                Q9: How do I redeem credit card rewards?
                A: Rewards can be redeemed through the credit card issuer's website or app, often for statement credits, gift cards, travel bookings, or merchandise.

                Q10: Are there any restrictions on earning rewards?
                A: Some credit cards have restrictions, such as spending categories, earning caps, or expiration dates for rewards points.
            """),
        MessagesPlaceholder(variable_name="history"),
        ("human", f"{input_message}"),
    ])

    chain = (
        RunnablePassthrough.assign(
            history=RunnableLambda(memory.load_memory_variables) | itemgetter("history")
        )
        | prompt
        | model
    )

    inputs = {"input": input_message}
    response = chain.invoke(inputs)

    # Save the context for future interactions
    memory.save_context(inputs, {"output": response.content})
    memory.load_memory_variables({})

    return response.content