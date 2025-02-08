# Customer Support Chatbot

This epic example is from
the [official doc](https://langchain-ai.github.io/langgraph/tutorials/customer-support/customer-support/)

Here's an overview of what we're implementing:
![](https://langchain-ai.github.io/langgraph/tutorials/customer-support/img/part-4-diagram.png)

## Part 1: Zero Shot Agent

![](https://langchain-ai.github.io/langgraph/tutorials/customer-support/img/part-1-diagram.png)

### Review

[Source](https://langchain-ai.github.io/langgraph/tutorials/customer-support/customer-support/#part-1-review)
> If this were a simple Q&A bot, we'd probably be happy with the results above. Since our customer support bot is taking
> actions on behalf of the user, some of its behavior above is a bit concerning:
>
>
> 1. The assistant booked a car when we were focusing on lodging, then had to cancel and rebook later on: oops! The user
     should have final say before booking to avoid unwanted feeds.
> 1. The assistant struggled to search for recommendations. We could improve this by adding more verbose instructions
     and examples using the tool, but doing this for every tool can lead to a large prompt and overwhelmed agent.
> 1. The assistant had to do an explicit search just to get the user's relevant information. We can save a lot of time
     by fetching the user's relevant travel details immediately so the assistant can directly respond.
>
> In the next section, we will address the first two of these issues.

## Part 2: With Confirmation

![](https://langchain-ai.github.io/langgraph/tutorials/customer-support/img/part-2-diagram.png)

### Review

[Source](https://langchain-ai.github.io/langgraph/tutorials/customer-support/customer-support/#part-2-review)
> This graph worked pretty well! We didn't really need to be involved in EVERY assistant action, though...

## Part 3: Conditional Interrupt

![](https://langchain-ai.github.io/langgraph/tutorials/customer-support/img/part-3-diagram.png)

### Review

[Source](https://langchain-ai.github.io/langgraph/tutorials/customer-support/customer-support/#part-3-review)
> Much better! Our agent is now working well - check out a LangSmith trace of our latest run to inspect its work! You may be satisfied with this design. The code is contained, and it's behaving as desired.
>
> One problem with this design is that we're putting a lot of pressure on a single prompt. If we want to add more tools, or if each tool gets more complicated (more filters, more business logic constraining behavior, etc), it's likely the tool usage and overall behavior of the bot will start to suffer.

## Part 4: Specialized Workflows

![](https://langchain-ai.github.io/langgraph/tutorials/customer-support/img/part-4-diagram.png)

### Review

[Source ?????]()
> TODO ???

