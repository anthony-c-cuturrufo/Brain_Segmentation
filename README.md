# Brain_Segmentation
Applying Deep Image Segmentation methods to neurological pathways

## Setup
  Open shell and run ` git clone https://github.com/anthony-c-cuturrufo/Brain_Segmentation.git` in desired directory to clone this repository.  
  
  Run `pip install -r requirements.txt` 

## Instructions for applying saved model 
  To apply saved model to file, run `python predict.py -i <PATH_TO_INPUT_IMAGE> --model <PATH_TO_MODEL>` 
  
  The output from the model, will be placed in the same directory and will end with the suffix `_OUT` 
  
  See `python predict.py -h` for more info 
