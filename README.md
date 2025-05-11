# GENERATIVE-TEXT-MODEL

*COMPANY* : CODTECH IT SOLUTIONS

*NAME* : LENKALAPALLY TEJASWI

*INTERN ID* : CODF47

*DOMAIN* : GENERATIVE TEXT MODEL

*DURATION* : 4 WEEKS

*MENTOR*: NEELA SANTHOSH

*DESCRIPTION* : This code builds a simple and easy-to-use website that lets you type a topic and get a paragraph written by a computer using artificial intelligence (AI). It uses a tool called Streamlit to create the web page, and a pre-trained AI language model called GPT-2, which was made by a company called OpenAI. The GPT-2 model can write text that sounds like it was written by a human. When the app starts, it sets up the page and loads the GPT-2 model and its tokenizer (which helps the model understand and break down words into smaller parts it can work with). This model loading is done only once using caching, so it doesn't slow down the app every time it runs.

The main function in the code is called generate_paragraph(). This function takes the topic you give, turns it into tokens using the tokenizer, and then uses the GPT-2 model to create a paragraph based on that topic. You can control how long the paragraph will be by adjusting the "Max paragraph length" slider. You can also control how creative or random the paragraph should be by using the "Creativity (temperature)" slider. A low temperature means the paragraph will be more focused and logical, while a high temperature means it will be more random and creative.

The user interface (UI) part of the code makes the app interactive. It shows a title, a short description, a text box where you type your topic, and two sliders for choosing how long the paragraph should be and how creative the AI should be. There is also a button called "Generate Text". When you click this button, the app shows a message saying "Generating..." while the paragraph is being created. After a few seconds, the AI-written paragraph appears below.

This app is a fun and simple way to explore how AI can write text. It can be used for fun, learning, idea generation, or even creative writing. The best part is that all the hard work is done behind the scenes by the GPT-2 model, and you don’t need to know how the AI works to use it. You just give it a topic and adjust a few settings, and the model will write a new paragraph every time. Each time you enter the same topic, you might get a different paragraph because of the random choices the model makes (especially if the temperature is high). This code is a great example of how AI and web development can come together to make powerful and easy-to-use tools.

*output* :

![Image](https://github.com/user-attachments/assets/873db8cc-b0dd-4404-bd84-1be3d2c88e5d)

![Image](https://github.com/user-attachments/assets/ca2979b2-ab1a-44d8-99c1-491816465f01)

