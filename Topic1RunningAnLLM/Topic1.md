# Topic 1: Running an LLM

## 4. Time the code

The timing is obtained from my Apple M2 Macbook Air machine (16GB memory).

1. Using GPU and no quantization: ` 12.77s user 3.69s system 43% cpu 37.714 total`
2. Using CPU and no quantization: `120.33s user 10.19s system 104% cpu 2:05.41 total`
3. Using CPU and 4-bit quantization. `37.89s user 10.87s system 16% cpu 5:03.84 total`

Using GPU (mps) is much faster than purely using CPU: 38 seconds versus 125 seconds.
However, using CPU with 4-bit quantization is actually slower than using CPU without quantization.
The reason could be that Apple CPU has no fast int4 instructions, and dequantization overhead increases the time.

## 5. Modify the code

The modified code can be seen in the script.

## 6. Mistakes

From a manual inspection on the philosophy questions answered by llama3.2-1b, qwen2.5-1.5b, and gemma2-2b, it can be seen that the mistakes made by the models are vastly different, and that they appear random.

## 8. Chat Agent

I implemented the fixed window strategy, to solve the long conversation problem.
The comparison of multi-turn conversation is showed in `with_history.txt` and `no_history.txt`.
It is clear that, with chat history turned on, the model is able to utilize the previous user input and model response.
Without chat history, the model is unable to do that.
