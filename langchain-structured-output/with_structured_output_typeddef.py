from langchain_groq import ChatGroq
import os
from dotenv import load_dotenv
from typing import TypedDict, Annotated, Optional, Literal

load_dotenv()

class Review(TypedDict):
    key_themes: Annotated[list[str], "List down all the key themes discussed in the review"]
    summary: Annotated[str, "A brief summary of the review"]
    sentiment: Annotated[str, "The overall sentiment of the review, e.g., positive, negative, neutral"]
    props: Annotated[Optional[list[str]], "Write down all the pros inside a list from the review"]
    cons: Annotated[Optional[list[str]], "Write down all the cons inside a list from the review"]

model = ChatGroq(model="openai/gpt-oss-20b", api_key=os.getenv("GROK_API_KEY"), temperature=0,
                  reasoning_effort="medium"
                  )

structured_model = model.with_structured_output(Review)

# result = structured_model.invoke("""
# The smartphone offers a smooth overall experience with a bright and sharp display, fast performance, and good battery life. The camera takes detailed photos in daylight, but low-light photos can look noisy and lack detail. The phone also feels slightly heavy, and some pre-installed apps are unnecessary. Gaming performance is good, but the device can get warm during longer gaming sessions. Charging could also be faster. Overall, it is a solid smartphone, but there are a few areas that could be improved.""")


result = structured_model.invoke("""I’ve been using this smartphone for a few weeks and overall I’m quite impressed. The display is bright and sharp, and the 120Hz refresh rate makes scrolling feel very smooth. Performance is also good for everyday tasks, multitasking, and gaming.

The camera takes detailed photos in good lighting, although low-light images could be better. Battery life easily lasts a full day with moderate usage, and charging is reasonably fast.

The phone feels premium and comfortable to hold, but it is slightly heavier than expected. The software experience is smooth, though there are a few pre-installed apps that I would prefer not to have.

Overall: A solid smartphone with a great display, good performance, and reliable battery life. There are a few minor drawbacks, but it offers a good overall experience.""")

print(result)
