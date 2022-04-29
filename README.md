# Brain_Segmentation
Applying Deep Image Segmentation methods to neurological pathways


## Instructions for applying saved model 
  Open shell and run `https://github.com/anthony-c-cuturrufo/Brain_Segmentation.git` in desired directory 
  
  Run `pip install -r requirements.txt` 
  
  To apply saved model to file, run `python predict.py -i <PATH_TO_INPUT_IMAGE> --model <PATH_TO_MODEL>` 
  
  The output from the model, will be placed in the same directory and will end with the suffix `_OUT` 
  
  See `python predict.py -h` for more info 
