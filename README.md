# **NEUROMETRIC INDEX FOR LARGE LANGUAGE MODELS** 

# **ARTIFICIAL INTELLIGENCE NEUROMETRIC INDEX (AINI)**



### Script structure and execution order



**1. dataset\_{name}.py** 

downloading datasets



**2. models\_vlm\_multitask.py**

model selection from HuggingFace



**3. download\_models\_multitask.py**

download models locally



**4. vlm\_multitask\_varA.py**

main script testing the models on the datasets



**5. connectivity\_vlm\_multitask\_pca.py**

building a connectome from model activations of all layers. 

A variant without PCA and using only layer-mean is also available. 



**6. plot\_connectomes\_vlm\_multitask.py**

plotting connectomes 



**7. graph\_analysis\_vlm\_multitask.py**

analyses of the connectomes with graph theory



**8. plot\_graphs\_vlm\_multitask**

plots from graph analyses

