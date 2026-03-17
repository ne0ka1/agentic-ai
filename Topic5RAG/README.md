# Topic Retrieval Augmented Generation

## Task 1. Open Model RAG v.s No RAG Comparison

According to the outputs (see [1. Model T Ford output](./11_output.txt) and [1. Congressional Record output](./12_output.txt)):

1. Yes, the model hallucinate specific value without RAG. In the question "what is the correct spark plug gap for a Model T Ford", the model answers 0.25 inches without RAG, and 0.74 inches with RAG (which is also not correct).
2. Yes, RAG ground the answers in the actual manual.
3. Yes, there are cases where the model's general knowledge is correct. For example, in the question "what is the purpose of the Main Street Parity Act?", the model's own knowledge is largely correct and in line with the document -- to protect small business by providing them with more equitable access to financing.

## Task 2: Open Model + RAG vs. Large Model Comparison

The output for the [program](./2_gpt4omini_queries.py) is saved in [2. Model T Ford output](./21_output.txt) and [2. Congressional Record output](./22_output.txt).

1. Indeed, GPT 4o mini does a better job in avoiding hallucinations than Qwen 2.5 1.5B. In the congressional record queries about speeches in January 13, 2026, GPT 4o mini made sure to say things like "I do not have access to rea-time information or events that occurred after October 2023" rather than making things up like Qwen does.
2. In congressional record related queries, GPT fo mini answer correctly on the question "What is the purpose of the Main Street Parity Act". Indeed, it is more detailed than Qwen 2.5 1.5B both with and without RAG. The cut-off date of GPT 4o mini pre-training seems to be October 2023, while the age of the Model T Ford corpora is , and the Congressional Record corpora is Janurary 2026.

## Task 3: Open Model + RAG vs. State-of-the-Art Chat Model

The output for ChatGPT-5.3 Instant is saved in [3_output.txt](./3_output.txt).

#TODO

## Task 4: Effect of Top-K Retrieval Count

Two sample queries and relevant output (with context) are saved in [41_output.txt](./41_output.txt) and [42_output.txt](./42_output.txt).
For the question "What is the correct spark plug gap for a Model T Ford?", when the k is too small, the model can't get relevant information and unable to answer correctly; when k >= 5, the model gets the relevant context but fails to process it correctly.
For the quesiton "What is the purpose of the Main Street Parity Act?", the model's answer is increasingly more detailed when k becomes larger.

1. At roughly k = 10, adding more context stop helping.
2. I don't see too much context hurt from this simple experimentation, when k is below 20.
3. In general, the needed k and chunk size are inversely correlated.

## Task 5: Handling Unanswerable Questions

The outputs of example unanswerable questions are saved in [5_output.txt](./5_output.txt).

1. The model sometimes admits it doesn't know: "The given context does not provide any information about the horsepower of a 1925 Model T. Therefore, I cannot determine its horsepower based solely on this text." At other times, the model would not admit but infer something confidently from the context, like the advantage of synthetic oil.
2. The model produces wrong thinking processes in answering "what is the capital of France?": "The capital of France is Paris. This can be inferred from the fact that France is listed as one of the countries mentioned in the context, with its currency (Euro) being discussed alongside other European currencies like the U.S. dollar equivalent or U.S. currency...". The model says it infers this knowledge from the context which does not actually states the fact.
3. As can be seen in 2., the retrieved context is unnecessary (as the model already knows the capital of France) and adds false reasoning steps to the answers.
4. I tried adding the warning "If the context doesn't contain the answer, say 'I cannot answer this from the available documents.'", but this does not help, the model still confidently says the manual recommends synthetic oil.

## Task 6. Query Phrasing Sensitivity

The outputs of example "engine maintenance intervals" question variations are saved in [6_output.txt](./6_output.txt).
This table records the chunk origin and similarity scores of five results.

| | Formal | Casual | Keywords | Question | Indirect |
|--|--|--|--|--|--|
| 1st Chunk| ATA_71 0.584 | ModelTNew 0.509 | ATA_05 0.616 | ATA_71 0.589 | ATA_71 0.621 |
| 2nd Chunk | ATA_24 0.572 | ModelTNew 0.492 | ATA_24 0.577 | ATA_71 0.550 | ATA_35 0.609 |
| 3rd Chunk | ATA_71 0.570 | ModelTNew 0.490 | ATA_05 0.565 | ModelTNew 0.550 | ATA_71 0.608 |
| 4st Chunk | ATA_71 0.537 | ModelTNew 0.490 | ATA_05 0.533 | ATA_71 0.529 | ATA_34 0.570 |
| 5st Chunk | ATA_27 0.535 | ATA_12 0.479 | ATA_05 0.547 | ATA_71 0.526 | ATA_34 0.564 |

1. Only Casual and Question style phrasing gets the correct answers (14 days or twice a month), so we may say they retrieve the best chunks.
2. Keyword-style queries work worse.
3. Ask direct, and in natural language form.

## Task 7 Chunk Overlap Experiment

The question I asked is "What is EU AI Act's definition of an AI system?"
The output with different overlap configuration is recorded in [7_output.txt](./7_output.txt).

1. It is clear that higher overlap improves retrieval of complete information and the quality of the answer!
2. The cost is certainly index size and time to build it.
3. Yes, it is also clear that the improvement obtained from overlap=128 to 256 is not as worthwhile as that from 0 to 64 or 64 to 128.