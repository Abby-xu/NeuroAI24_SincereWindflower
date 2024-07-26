import pickle
import os
import sys
import os
CTD_path = os.path.join(os.getcwd(), '..', 'ComputationThruDynamicsBenchmark')
sys.path.append(CTD_path)
from ctd.comparison.analysis.tt.tt import Analysis_TT
import pandas as pd

Models_path = os.path.join(os.getcwd(),"NeuroAI24_SincereWindflower", "Models", "Trained_Models")
Models_fnames = os.listdir(Models_path)

for model_fname in Models_fnames:
    # Load the analysis object with the model
    model_path = os.path.join(Models_path, model_fname, "")
    analysis = Analysis_TT(run_name = "GRU_128_RT", filepath = model_path)
    
    # PLot 2 representative trials
    analysis.plot_trial_io(num_trials = 2)

    # Get the model inputs, outputs, and latents
    ics, inputs, targets = analysis.get_model_inputs()
    out_dict = analysis.get_model_outputs()
    latents = out_dict["latents"].detach().numpy()
    hand_pos = out_dict["controlled"].detach().numpy()

    print("Model Inputs: ", inputs.shape)


