# Topic 4 Exploring Tools

## Task 3

The example outputs are saved in [toolnode_output.txt](./toolnode_output.txt) and [react_output.txt](./react_output.txt).

1. Tools are defiend with `async def`, and `asyncio.run` ensures multiple tool calls can be run concurrently. The tools that are calls external API or are I/O bounds, and independent of other tools' results would benefit most from parallel dispatch, such as the `get_weather` and `get_population` in the example code.
2. Both programs handle special inputs in `input_node`, and route after input: if the command is `exit`, then go to END; if the command is `verbose` or `quiet`, go back to input; if the command is None, then go to the main agent node.
3. The only structural difference is whether tools are an explicit node (ToolNode) or hidden inside the agent (ReAct agent node).
4. An example would be, we want to process tool outputs before the model call, e.g. summarized, validated, or filtered before they're sent back to the model. In ToolNode, we can just add a node between `tools` and `call_model`, but it would be awkward to do that in ReAct.

## Task 5

I built a simple "Research Assistant" agent that will query both DuckDuckGo and Wikipedia, compare the result, and then generate a brief report, with link to the website sources attached.
A sample run is shown in [output.txt](./output.txt).
With multiple sources, the result is more credible.
For exmaple, for the question "what is Ang Lee's best movie?", the generated report takes intersection of results of both sources and picks "Brokeback Mountain" and "Crouching Tiger, Hidden Dragon".