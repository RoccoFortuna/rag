# rag
Building a barebones RAG system to provide large sources of text and have an LLM respond to questions about it.

## Usage
install dependencies with uv, run:
`uv sync`

And activate venv:
`source ./venv/bin/activate`

then add your corpus to the `context_files/` folder and update the path to your context file in rag.py. Finally run:
`python rag.py`

The context file will be parsed and indexed in your local FAISS (Facebook AI Similarity Search) vector databse:
```
Processing book text (LIMITED to 20000 lines for testing)...
❌ FAISS index does not exist or is empty. Creating a new index...
Processing first 20000 lines:   0%|                                                                                                                          | 0/3351 [00:00<?, ? lines/s]🛠 Creating a new FAISS index...
✅ New FAISS index created with 5 entries.
🧐 Checking FAISS size... Current vector count: 5
💾 FAISS index saved at line 17. Contains 5 entries.
Processing first 20000 lines:   0%|▏                                                                                                                 | 5/3351 [00:00<04:56, 11.28 lines/s]
...
```

Once completed you'll be able to prompt the LLM for local inference on your context, for example:
```
Ask a question about the book (or type 'exit'): what should I know about chapter 3 "Offer Service"?

📝 Full Prompt to LLM:
 You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question. If you don't know the answer, just say that you don't know. Use three sentences maximum and keep the answer concise.
Question: what should I know about chapter 3 "Offer Service"?
Context: Chapter 3 Offer Service Remember the people you address are selfish, as we all are. They care nothing about your interest or your profit. They seek service for themselves. Ignoring this fact is a common mistake and a costly mistake in advertising. Ads say in effect, "Buy my brand. Give me the trade you give to others. Let me have the money." That is not a popular appeal. The best ads ask no one to buy. That is useless. Often they do not quote a price. They do not say that dealers handle the product. The ads are based entirely on service. They offer wanted information. They site advantages to users. Perhaps they offer a sample, or to buy the first package, or to send something on approval, so the customer may prove the claims without any cost or risk.
make the effort to get it, even though the thing is a trifle. In the same way it is found that an offer limited to a certain class of people is far more effective than a general offer. For instance, an offer limited to veterans of the war. Or to members of a lodge or sect. Or to executives. Those who are entitled to any seeming advantage will go a long way not to lose that advantage.
washing machines, books, etc. - are sent out in this way on approval. And we find that people are honest. The losses are very small. An advertiser offered a set of books to business men. The advertising was unprofitable, so he consulted another expert. The ads were impressive. The offer seemed attractive, "But," said the second man, "let us add one little touch which I have found effective. Let us offer to put the buyer’s name in gilt Carl Galletti
Answer:

AI Response: <think>
Okay, so I need to explain what Chapter 3 on "Offer Service" is about based on the provided context. The chapter starts by pointing out that people are selfish and care more about their own interests rather than others', which is a common mistake advertisers make.

The key idea here is that effective ads shouldn't focus on asking people to buy products but instead should provide value or information that benefits the customer. For example, offering samples or allowing customers to try products risk-free can build trust. It's important to make these offers specific to certain groups, like veterans or business executives, because targeted offers are more effective.

Additionally, even though there might be some risk in offering free trials or samples, the benefits often outweigh any potential losses since people generally act honestly. This approach shifts the focus from selling a product to serving the customer's needs, which is crucial for building trust and effectiveness in advertising.
</think>

Chapter 3 emphasizes that ads should focus on providing value rather than asking customers to buy. Effective offers include free samples or trials, making it easy for customers to test products without risk. Targeting specific groups can enhance the impact of these offers, as seen with successful campaigns like offering books to business executives.
```