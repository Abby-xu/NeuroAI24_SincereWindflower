from ctd.task_modeling.task_env.task_env import RandomTarget
from motornet.effector import RigidTendonArm26
from motornet.muscle import MujocoHillMuscle
from ctd.task_modeling.datamodule.task_datamodule import TaskDataModule
from ctd.task_modeling.task_wrapper.task_wrapper import TaskTrainedWrapper
from pytorch_lightning import Trainer
# from Models.LSTM import LSTM_Methyl
from ctd.task_modeling.model.rnn import LSTM_Methyl

# Create the analysis object:
rt_task_env = RandomTarget(effector = RigidTendonArm26(muscle = MujocoHillMuscle()))

# Initialize the task environment 
task_env = rt_task_env

# Initialize the model 
in_size = task_env.observation_space.shape[0] + task_env.context_inputs.shape[0]
out_size = task_env.action_space.shape[0]
lstm_model = LSTM_Methyl(output_size=2, input_size= in_size, latent_size=128)

# Setup the task in the datamodule
task_dm = TaskDataModule(task_env,n_samples=1000 ,batch_size=256)

# Setup the environment and model in the task wrapper
task_wrapper = TaskTrainedWrapper(learning_rate=1e-3, weight_decay=1e-8)
task_wrapper.set_environment(task_env)
task_wrapper.set_model(lstm_model)

# Define trainer
trainer = Trainer(accelerator = 'cpu', max_epochs=500, enable_progress_bar=False,)

# Train the model
trainer.fit(task_wrapper, task_dm)