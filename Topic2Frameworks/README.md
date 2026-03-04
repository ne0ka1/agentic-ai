# Topic 2: Agent Frameworks

## Step 1

The appropriate verbose option is added, see `1_output.txt`.

## Step 2

From `21_output.txt`, it can be seen that an empty input would make the model output random stuff, and each time different.
After adding an edge that goes back to itself, now an empty input would just pass through (see `22_output.txt`).

## Step 3

Now the model runs in parallel, see `3_output.txt`.

## Step 4

Now only one of the models would run, see `4_output.txt`.

## Step 5

Now the program maintain a chat history context, see `5_output.txt`.

## Step 6

Multi-agent chat with "Hey Qwen" / "Hey Llama" switching, see `6_output.txt`.

## Step 7: Checkpointing and crash recovery

Now the chat uses LangGraph checkpointing so you can kill it mid-conversation (e.g. Ctrl+C) and restart with nothing lost, see `7_output.txt`.
