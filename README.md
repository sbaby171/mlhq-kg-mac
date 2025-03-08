# mlhq-kg-mac
Project for builiing Agent Flow with Knowledge Graphs on MAC M4 Pro (and 
some API stuff)


**NOTE** Most of the inspiration for this project came from here (https://github.com/PeterGriffinJin/Graph-CoT/tree/main)
and some of the code related to GraphAgent were taken from there - but updated 
for my needs. 


# Follow-ups

QUES: How to route my HuggingFace InferenceClient inference to the MLHQ org
acount rather than my own? 


# Current Issues: 


1. Huggingface InferenceClient does not support local execution 
  * Therefore I could either create an account and use their infra or 
  * I could build my own wrapper for the support. 


^^^ For now, I am going with the first option


> [!NOTE]
> For this specific project, I think I can forgo the requirement of OpenAI-API 
chat completion capatibility. As long as I know the prompt, I can simply push
that into the locally hosted LLM via HugginFace Pipelines. 

# Cost: 

So far my general costs are: 
* M4 Max 128G &rarr; 5k
* HF Entrprise &rarr; 216 (billed yearly at 18 a month )
* Claude Pro &rarr; 20 bucks a month 
* OpenAI Businsess &rarr; 600 for a year. 
