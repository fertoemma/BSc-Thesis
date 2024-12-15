import optuna
import os


#####
# # working-> no modifications
# # Define the main directory for Optuna studies
# optuna_2dof_opt_main_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "optuna_2dof_k_inits")

# # Define the study name
# study_name = "2dof_k_inits"

# # Define the storage name (path to the SQLite database)
# storage_name = os.path.join(f"sqlite:///{optuna_2dof_opt_main_dir}", f"{study_name}.db")

# # Load the study
# study = optuna.load_study(study_name=study_name, storage=storage_name)

####


# # Define the main directory for Optuna studies
# optuna_2dof_opt_main_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "optuna_2dof_2_k_inits_pelv_chst_head_obj")

# # Define the study name
# study_name = "3dof_k_inits"

# # Define the storage name (path to the SQLite database)
# storage_name = os.path.join(f"sqlite:///{optuna_2dof_opt_main_dir}", f"{study_name}.db")

# # Load the study
# study = optuna.load_study(study_name=study_name, storage=storage_name)

# Define the main directory for Optuna studies
optuna_2dof_opt_main_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "optuna_2dof_2_k_inits_pelv_obj")

# Define the study name
study_name = "2dof_2_k_inits"

# Define the storage name (path to the SQLite database)
storage_name = os.path.join(f"sqlite:///{optuna_2dof_opt_main_dir}", f"{study_name}.db")

# Load the study
study = optuna.load_study(study_name=study_name, storage=storage_name)