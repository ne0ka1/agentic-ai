# Topic Retrieval Augmented Generation

## Task 1. Open Model RAG v.s No RAG Comparison

According to the outputs (see [1. Model T Ford output](./1_modelt_ford.txt) and [1. Congressional Record output](./1_congressional_record.txt):

1. Yes, the model hallucinate specific value without RAG. In the question "what is the correct spark plug gap for a Model T Ford", the model answers 0.25 inches without RAG, and 0.74 inches with RAG (which is the correct answer).
2. Yes, RAG ground the answers in the actual manual.
3. Yes, there are cases where the model's general knowledge is correct. For example, in the question "what is the purpose of the Main Street Parity Act?", the model's own knowledge is largely correct and in line with the document -- to protect small business by providing them with more equitable access to financing.