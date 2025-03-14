# mlhq-kg-mac
Project for builiing Agent Flow with Knowledge Graphs on MAC M4 Pro (and 
some API stuff)


**NOTE** Most of the inspiration for this project came from here (https://github.com/PeterGriffinJin/Graph-CoT/tree/main)
and some of the code related to GraphAgent were taken from there - but updated 
for my needs. 


# Follow-ups

QUES: How to route my HuggingFace InferenceClient inference to the MLHQ org
acount rather than my own? 

https://www.reddit.com/r/huggingface/comments/1idkjx5/hf_new_inference_providers_pricing_confusion/ 


https://huggingface.co/docs/api-inference/en/pricing


# Current Issues: 


1. Huggingface InferenceClient does not support local execution 
  * Therefore I could either create an account and use their infra or 
  * I could build my own wrapper for the support. 


^^^ For now, I am going with the first option


> [!NOTE]
> For this specific project, I think I can forgo the requirement of OpenAI-API 
chat completion capatibility. As long as I know the prompt, I can simply push
that into the locally hosted LLM via HugginFace Pipelines. 

