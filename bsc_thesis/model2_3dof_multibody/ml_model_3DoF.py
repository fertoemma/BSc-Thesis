# Imports
import os
import sys 
sys.path.append(r'C:\Users\emmaf\Documents\7. félév\SZAKDOLGOZAT\osd_dummy_modeling_2\osd_dummy_modeling')
import matplotlib.pyplot as plt
import function_files.krc_reader as krc_reader
import function_files.filterSAEJ211 as filter
import numpy as np
from lmfit import Parameters, Minimizer
import pickle
import model1_2dof_inverse_pendulum.equation_solver_2DoF as solv
import function_files.data_saver as data_saver 
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import optuna


class EncoderDecoder(nn.Module):    # Hyperparaméter optimalizáció használható
    def __init__(self):
        super().__init__()
        # enc_layer_1 = trial.suggest_int()
        
        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(140, 100).double(),
            nn.Tanh().double(),
            nn.Linear(100, 70).double(),
            nn.Tanh().double(),
            nn.Linear(70, 40).double(),
        )
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(40, 70).double(),
            nn.Tanh().double(),
            nn.Linear(70, 100).double(),
            nn.Tanh().double(),
            nn.Linear(100, 140).double(),
        )
    
    def forward(self, x):
        y_encoded = self.encoder(x)
        y_decoded = self.decoder(y_encoded)
        return y_decoded


# Loading data
folder = 'saved_data'
# fname = 'EU2_11HEAD0000H3ACX0_3dof_3ch'
fname = 'EU2_11PELV0000H3ACX0_m2_3ch'
# fname = 'EU2_11CHST0000H3ACX0_m2_3ch'
# fname = 'EU2_11HEAD0000H3ACX0_m2_3ch'
dict = data_saver.load(folder, fname) # pelvis data, később cserélhető, bővíthető

sled_acceleration = [v[0] for v in dict.values()]
sled_acceleration = np.array(sled_acceleration)
pelvis_acceleration_computed = [v[1] for v in dict.values()]
pelvis_acceleration_computed = np.array(pelvis_acceleration_computed)
pelvis_acceleration_simulated = [v[2] for v in dict.values()]
pelvis_acceleration_simulated = np.array(pelvis_acceleration_simulated)
pelv_diff = [v[3] for v in dict.values()]
pelv_diff = np.array(pelv_diff)
print(pelv_diff[0].shape)


# filtering 140
index = [i for i in range(0, 1400, 10)]

y_list = []
for i in pelv_diff:
    y = i[index]
    y_list.append(y)
x_list = []
for i in sled_acceleration:
    x = i[index]
    x_list.append(x)

comp_list = []
for i in pelvis_acceleration_computed:
    x = i[index]
    comp_list.append(x)
simu_list = []
for i in pelvis_acceleration_simulated:
    x = i[index]
    simu_list.append(x)

# Train-test split 
print(len(sled_acceleration))
train_test_indices = list (range(0, len(sled_acceleration)))
train_indices , test_indices = train_test_split(train_test_indices, train_size=0.8, random_state=42)
X_train = [x_list[i] for i in train_indices]    # input: sled acceleration
X_test = [x_list[i] for i in test_indices]
y_train = [y_list[i] for i in train_indices]    # output: difference between simu and computed values
y_test = [y_list[i] for i in test_indices]

# Scaler
scaler_X = StandardScaler()
scaler_y = StandardScaler()
scaler_X.fit(X_train)
scaler_y.fit(y_train)
# Transform the training and testing data
X_train_scaled = scaler_X.transform(X_train)
X_test_scaled = scaler_X.transform(X_test)
y_train_scaled = scaler_y.transform(y_train)
y_test_scaled = scaler_y.transform(y_test)

X_train_tensor = torch.tensor(X_train_scaled).double()
X_test_tensor = torch.tensor(X_test_scaled).double()
y_train_tensor = torch.tensor(y_train_scaled).double()
y_test_tensor = torch.tensor(y_test_scaled).double()

print(f"X_train_tensor shape: {X_train_tensor.shape}")
print(f"X_test_tensor shape: {X_test_tensor.shape}")
print(f"y_train_tensor shape: {y_train_tensor.shape}")
print(f"y_test_tensor shape: {y_test_tensor.shape}")

train_dataset = torch.utils.data.TensorDataset(X_train_tensor, y_train_tensor)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True)

test_dataset = torch.utils.data.TensorDataset(X_test_tensor, y_test_tensor)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=32, shuffle=False)

model = EncoderDecoder()
criterion = nn.MSELoss()    # lehet más hibaszámítási mód, MSELoss a leggyakoribb
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)  # lehet más, Adam a leggyakoribb

train_losses = list()
test_losses = list()


# Training
epochs = 100
for epoch in range(epochs):
    model.train()
    train_loss = 0.0
    for inputs, targets in train_loader:
        # Zero the parameter gradients
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        train_loss += loss.item() * inputs.size(0) # Always adds the total (unaveraged) loss for a batch to the total training loss.
    
    train_loss_over_epoch = train_loss / len( train_loader.sampler ) # Calculate average train loss over the entire epoch.
    train_losses.append( train_loss_over_epoch )

    # before_lr = optimizer.param_groups[0]["lr"]
    # scheduler.step()
    # after_lr = optimizer.param_groups[0]["lr"]
    
    model.eval()
    test_loss = 0.0
    with torch.no_grad(): # Evaluating test loss does not require backprop and therefore gradients.
        for inputs, targets in test_loader:

            outputs = model( inputs )
            loss = criterion(outputs, targets)
            test_loss += loss.item() * inputs.size(0)

    test_loss_over_epoch = test_loss / len( test_loader.sampler )
    test_losses.append( test_loss_over_epoch )

    print(f"Epoch [{epoch+1}/{epochs}], Train loss: {train_loss_over_epoch:.4f}, Test loss: {test_loss_over_epoch:.4f}")
    # print("Epoch %d: lr %.4f -> %.4f" % (epoch, before_lr, after_lr))

diff_model = model(X_test_tensor)
diff_model = diff_model.detach().numpy()

output_dir = os.path.join(os.path.dirname(__file__), 'output_plots')
os.makedirs(output_dir, exist_ok=True)
# Plot template
plt.rc('text', usetex=True)
plt.rc('font', family='serif', size=11)
# Validating the model
plt.figure(1)
plt.plot( train_losses, label="train loss", color='#c18739' )
plt.plot( test_losses, label="test loss", color='#50b47b')
plt.legend()
plt.grid()
plt.xlabel('Epoch', fontsize=11)
plt.ylabel('Loss', fontsize=11)
plt.title(f"Validation \nEU2_11PELV0000H3ACX0_m3_3ch", fontsize=14)
# plt.title(f"Validation \nEU2_11CHST0000H3ACX0_m3_3ch", fontsize=14)
# plt.title(f"Validation \nEU2_11HEAD0000H3ACX0_m3_3ch", fontsize=14)
# plt.show()
plt.savefig(os.path.join(output_dir, 'validation_EU2_11PELV0000H3ACX0_m3_3ch.png'))
plt.savefig(os.path.join(output_dir, 'validation_EU2_11PELV0000H3ACX0_m3_3ch.svg'))
plt.close()


# Checking the results of the training
predicted_diff = scaler_y.inverse_transform(diff_model)
predicted_diff = np.array(predicted_diff)
computed_acceleration = [comp_list[i] for i in test_indices]
simulated_acceleration = [simu_list[i] for i in test_indices]
# simulated_acceleration = np.array(simu_list)
print(f"Diff: {predicted_diff.shape}")

print( len(X_test) % 32 ) # last batch of the test eval round is not a "full" batch. Only 7 samples are left for it. -> last value of outputs has the shape of (7,140)

element_num = 10
# Missmatch between computed_acceleration and predicted_diff: predicted diff is 7 long, and it is the 7 last element in the test set, computed acceleration is 194 long and contains the full dataset.
predicted_with_diff = computed_acceleration[element_num] + predicted_diff[element_num]
predicted = computed_acceleration[element_num]

# plt.figure(2)
# plt.plot(predicted_with_diff, label = "predicted_with_diff")
# plt.plot( simulated_acceleration[element_num], label = "simulated")
# plt.plot( computed_acceleration[element_num], label = "fitted from eom")
# plt.legend()
# plt.grid()
# plt.title(f"Comparison of results - {fname}")
# plt.show()

# Plot template
plt.rc('text', usetex=True)
plt.rc('font', family='serif', size=11)

plt.figure(3)
plt.plot(simulated_acceleration[element_num], label="simulation", color='#86a542')
plt.plot(computed_acceleration[element_num], label="fitted", color='#6881d8')
plt.plot(predicted_with_diff, label="predicted with diff.", color='#b84c3e')
plt.legend()
plt.grid()
plt.xlabel('Time [s]', fontsize=11)
plt.ylabel('Acceleration [mm/s$^2$]', fontsize=11)  
plt.title(f"Comparison of results  \nEU2_11PELV0000H3ACX0_m3_3ch", fontsize=14)
# plt.title(f"Comparison of results  \nEU2_11CHST0000H3ACX0_m3_3ch", fontsize=14)
# plt.title(f"Comparison of results  \nEU2_11HEAD0000H3ACX0_m3_3ch", fontsize=14)
# plt.show()
plt.savefig(os.path.join(output_dir, 'comparison_EU2_11PELV0000H3ACX0_m3_3ch.png'))
plt.savefig(os.path.join(output_dir, 'comparison_EU2_11PELV0000H3ACX0_m3_3ch.svg'))
plt.close()