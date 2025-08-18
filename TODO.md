

- Take another look at the calc questions.
- Derive an extensive testing set from them. 
- With a seperate training set. In Sequential order. 

- Test basic models on testing set. 

- Fine tune the models on training data. In sequential order and random order. 
- Test them again. 



nvitop is command of interest. 


# Tomorrow:
- Can probably parralelize the api calling process for much faster testing.
1. Evaluate 7b 
2. Compare 2b and 7b. 
3. See how you can fine tune with RLHF.



# Issues with CLP dataset
. Generating context lenght is too short.
. Some problems include graphics. Could be fixed for searching of /includegraphics tags.
. model response includes question prompt

# Observations:
- gemma_2b_it got 11/900 on math benchmark pre-train
- gemma_7b_it got 1/900! on math benchmark pre-train
