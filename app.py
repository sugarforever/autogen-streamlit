"""
This is a demo of AutoGen chat agents. You can use it to chat with OpenAI's GPT-3 and GPT-4 models. They are able to execute commands, answer questions, and even write code.
An example a question you can ask is: 'How is the S&P 500 doing today? Summarize the news for me.'
UserProxyAgent is used to send messages to the AssistantAgent. The AssistantAgent is used to send messages to the UserProxyAgent.

"""


import streamlit as st
import asyncio
from autogen import AssistantAgent, UserProxyAgent

# setup page title and description
st.set_page_config(page_title="AutoGen Chat app", page_icon="🤖", layout="wide")

st.markdown("Adapted from [this example](https://github.com/sugarforever/autogen-streamlit)")
st.markdown(
    "This is a demo of AutoGen chat agents. You can use it to chat with OpenAI's GPT-3 and GPT-4 models. They are able to execute commands, answer questions, and even write code."
)
st.markdown("An example a question you can ask is: 'How is the S&P 500 doing today? Summarize the news for me.'")
st.markdown("Start by getting an API key from OpenAI. You can get one [here](https://openai.com/pricing).")


class TrackableAssistantAgent(AssistantAgent):
    """
    A custom AssistantAgent that tracks the messages it receives.

    This is done by overriding the `_process_received_message` method.
    """

    def _process_received_message(self, message, sender, silent):
        with st.chat_message(sender.name):
            st.markdown(message)
        return super()._process_received_message(message, sender, silent)


class TrackableUserProxyAgent(UserProxyAgent):
    """
    A custom UserProxyAgent that tracks the messages it receives.

    This is done by overriding the `_process_received_message` method.
    """

    def _process_received_message(self, message, sender, silent):
        with st.chat_message(sender.name):
            st.markdown(message)
        return super()._process_received_message(message, sender, silent)


# add placeholders for selected model and key
selected_model = None
selected_key = None

# setup sidebar: models to choose from and API key input
with st.sidebar:
    st.header("OpenAI Configuration")
    selected_model = st.selectbox("Model", ["gpt-3.5-turbo", "gpt-4-1106-preview"], index=1)
    st.markdown("Press enter to save key")
    st.markdown(
        "For more information about the models, see [here](https://platform.openai.com/docs/models/gpt-4-and-gpt-4-turbo)."
    )
    selected_key = st.text_input("API Key", type="password")

# setup main area: user input and chat messages
with st.container():
    user_input = st.text_input("User Input")
    # only run if user input is not empty and model and key are selected
    if user_input:
        if not selected_key or not selected_model:
            st.warning("You must provide valid OpenAI API key and choose preferred model", icon="⚠️")
            st.stop()
        # setup request timeout and config list
        llm_config = {
            "request_timeout": 600,
            "config_list": [
                {"model": selected_model, "api_key": selected_key},
            ],
            "seed": "42",  # seed for reproducibility
            "temperature": 0,  # temperature of 0 means deterministic output
        }
        # create an AssistantAgent instance named "assistant"
        assistant = TrackableAssistantAgent(name="assistant", llm_config=llm_config)

        # create a UserProxyAgent instance named "user"
        # human_input_mode is set to "NEVER" to prevent the agent from asking for user input
        user_proxy = TrackableUserProxyAgent(
            name="user",
            human_input_mode="NEVER",
            llm_config=llm_config,
            is_termination_msg=lambda x: x.get("content", "").strip().endswith("TERMINATE"),
        )

        # Create an event loop: this is needed to run asynchronous functions
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        # Define an asynchronous function: this is needed to use await
        if "chat_initiated" not in st.session_state:
            st.session_state.chat_initiated = False  # Initialize the session state

        if not st.session_state.chat_initiated:

            async def initiate_chat():
                await user_proxy.a_initiate_chat(
                    assistant,
                    message=user_input,
                    max_consecutive_auto_reply=5,
                    is_termination_msg=lambda x: x.get("content", "").strip().endswith("TERMINATE"),
                )
                st.stop()  # Stop code execution after termination command

            # Run the asynchronous function within the event loop
            loop.run_until_complete(initiate_chat())

            # Close the event loop
            loop.close()

            st.session_state.chat_initiated = True  # Set the state to True after running the chat


# stop app after termination command
st.stop()
