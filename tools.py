import requests
import json
import os
import io
from PIL import Image
from dotenv import load_dotenv

from langchain.pydantic_v1 import BaseModel, Field
from langchain.tools import StructuredTool
from langchain_community.utilities import WikipediaAPIWrapper

load_dotenv()

##### Wikipedia tool #####

def wiki_api_caller(query):
    wiki_api_wrapper = WikipediaAPIWrapper(top_k_results=1)
    return wiki_api_wrapper.run(query)

# Input parameter definition
class QueryInput(BaseModel):
     query: str = Field(description="Input search query")


# the tool description
wiki_tool_description: str = (
        "A wrapper around Wikipedia. "
        "Useful for when you need to answer general questions about "
        "people, places, companies, facts, historical events, or other subjects. "
        "Input should be a search query."
    )

# fuse the function, input parameters and description into a tool. 
wiki_tool = StructuredTool.from_function(
    func=wiki_api_caller,
    name="wikipedia",
    description=wiki_tool_description,
    args_schema=QueryInput,
    return_direct=False,
)


##### Weather tool ######

def extract_weather(city_name):
    base_url = "http://api.openweathermap.org/data/2.5/weather?"

    WEATHER_API_KEY = os.getenv("WEATHER_API_KEY")

    complete_url = base_url + "appid=" + WEATHER_API_KEY + "&q=" + city_name

    response = requests.get(complete_url)

    x = response.json()

    if x["cod"] != "404":
 
        # store the value of "main"
        # key in variable y
        y = x["main"]
    
        # store the value corresponding
        # to the "temp" key of y
        current_temperature = y["temp"] - 273.15      
    
        # store the value of "weather"
        # key in variable z
        z = x["weather"]
    
        # store the value corresponding 
        # to the "description" key at 
        # the 0th index of z
        weather_description = str(z[0]["description"])

        output = f"Current temperature in {city_name}: {current_temperature}°C and it is {weather_description}"
    else:
        output = f"Error: City {city_name} is not found"

    return output


# Input parameter definition
class WeatherInput(BaseModel):
    city_name: str = Field(description="City name")

# the tool description
weather_tool_description: str = (
        """
        Allows to extract the current temperature in a specific city. 
        """
    )

# fuse the function, input parameters and description into a tool. 
weather_tool = StructuredTool.from_function(
    func=extract_weather,
    name="weather",
    description=weather_tool_description,
    args_schema=WeatherInput,
    return_direct=False,
)



###### Image Generator tool #######
def text_to_image(payload):
    HUGGING_FACE_KEY = os.getenv("HUGGING_FACE_KEY")
    # API_URL = "https://api-inference.huggingface.co/models/Corcelio/mobius"
    API_URL = "https://api-inference.huggingface.co/models/stabilityai/stable-diffusion-3.5-large"
    headers = {"Authorization": f"Bearer {HUGGING_FACE_KEY}"}

    def query(payload):
            response = requests.post(API_URL, headers=headers, json=payload)
            return response.content
        
    image_bytes = query({
        "inputs": payload,
    })

    image = Image.open(io.BytesIO(image_bytes))
        
    # Resize the image
    new_size = (400, 400)  # Example new size (width, height)
    resized_image = image.resize(new_size)



    # Save the resized image to a file
    image_path = f'images/image_{payload.replace(" ", "_")}.jpg'
    resized_image.save(image_path)

    return image_path


# Input parameter definition
class ImageInput(BaseModel):
    payload: str = Field(description="What should be converted into image")


# the tool description
images_tool_description: str = (
       "Generate an image based on the input text and return its path"
    )

# fuse the function, input parameters and description into a tool. 
image_tool = StructuredTool.from_function(
    func=text_to_image,
    name="create_image",
    description=images_tool_description,
    args_schema=ImageInput,
    return_direct=False,
)
