

default_fitting_pipeline = {
    "structure": ["Load_Images", "Create_Models_Spec", "Initialize_Models", "fit loop", "Save_Models"],
    "node_kwargs": {
        "fit loop": {
            "structure": ["Sample_Models", "Loss_Image", "Compute_Loss", ("Stop_Iteration", ("Update_Parameters_Random_Grad", "End")), "Update_Parameters_Random_Grad"]
        }
    }
}
    
