import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import hessian_spectrum
import json
import torch.nn.functional as F
import seaborn as sns



# Hyperparameters
n_total = 5000
n_classes =  500
n_clusters = 500
input_dim = 500
width = 8
num_epochs = 0
cosine_epochs = 50000 
lr = 1e-4
model_type = 'nn' 
visual_degree = 100# hyperparamter for visualization: heatmap_max = hessian_max / visual_degree
visual_degree_block_hessian = [10, 100]
loss_type = 'CE'
optimizer_type = 'adam'
cmap = 'cividis'
gaussian_data = False
comment = 'clusterd-data-n_total-'+str(n_total)+'-'+model_type+'-dim-'+str(input_dim)+'-width-'+str(width)+'-cluster-'+ str(n_clusters)+'-class-'+str(n_classes)+'-adam-'+loss_type+'visiondegree'+str(visual_degree)+'T'+str(num_epochs)+'optimizer-'+optimizer_type +'lr-'+str(lr)

torch.manual_seed(0)
np.random.seed(0)



def generate_cluster_data(n_total, n_classes, n_clusters, input_dim):
    # Generate C-cluster synthetic data for specified dimensions
    # used for ablation study with Li et al. 19 https://arxiv.org/pdf/1903.11680, as suggested by Reviewer 1.
    # n_total is the total number of samples
    # n_classes is the number of classes (smaller than n_clusters)
    # input_dim is the dimension of the data
    # raise error if n_cluster is larger than n_classes
    assert n_classes<= n_clusters, f"n_cluster = {n_classes} is not smaller than n_classes = {n_classes}"
    
    # n_samples_per_class is the number of samples per class
    X = []
    y = []
    n_cluster_per_class = n_clusters // n_classes
    n_samples_per_cluster = n_total // n_clusters
    cluster_idx = 0
    for class_idx in range(n_classes):

        for _ in range(n_cluster_per_class):
            cluster_idx += 1
            if input_dim == 2:
                center = np.array([np.cos(2 * np.pi * cluster_idx / n_clusters), np.sin(2 * np.pi * (cluster_idx) / n_clusters)]) * 5  # Class centers on a circle
            else:
                #extend the 2D case to higher dimension
                # Generate random points in higher dimensions and project onto hypersphere
                center = np.random.randn(input_dim)
                # Normalize to create a unit vector (point on unit hypersphere)
                center = center / np.linalg.norm(center)

            cluster_samples = np.random.randn(n_samples_per_cluster, input_dim) * 0.05 + center  # Add some noise
            X.append(cluster_samples)
            # assign label
            y.extend([class_idx]*n_samples_per_cluster)

    X = np.vstack(X)  # Combine all class samples
    y = np.array(y)    # Convert labels to a NumPy array
    return X, y


def generate_gaussian_data(n_total, n_classes, input_dim):
    # Generate synthetic data for specified dimensions
    X = []
    y = []
    for _ in range(n_total):
        class_samples = np.random.randn(1, input_dim) 
        X.append(class_samples)
        y.extend([np.random.randint(0, n_classes)]) 

    X = np.vstack(X)  # Combine all class samples
    y = np.array(y)    # Convert labels to a NumPy array
    return X, y



# Generate synthetic data
if gaussian_data:
    X, y = generate_gaussian_data(n_total, n_classes, input_dim)
else:
    X, y = generate_cluster_data(n_total, n_classes, n_clusters, input_dim)
# Convert to PyTorch tensors
X_tensor = torch.from_numpy(X).float()
y_tensor = torch.from_numpy(y)  # Use class indices directly


# Define the neural network
class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()


        #self.fc_linear = nn.Linear(input_dim, n_classes, bias = False)   
        # # add a convolutional layer
        # self.conv = nn.Conv2d(1, 1, kernel_size = 3, stride = 1, padding = 1)
        self.fc1 = nn.Linear(input_dim, width, bias = False) 
        #self.fc_hidden = nn.Linear(width, width, bias = False)
        self.fc2 = nn.Linear(width, n_classes, bias = False)   
        #self.fc1_linear = nn.Linear(input_dim, n_classes, bias = False)
        # use LeCun initialization
        # nn.init.kaiming_normal_(self.fc1_linear.weight)
    def forward(self, x):

        #x = self.fc1_linear(x)
        #x = self.conv(x)
        x = self.fc1(x) # Activation function
        #x = torch.tanh(x)
        x = torch.relu(x)
        # x = self.fc_hidden(x)
        # x = torch.relu(x)
        x = self.fc2(x) # Activation function
        #x = self.fc1_linear(x)
        return x
    



class linearNN(nn.Module):
    def __init__(self):
        super(linearNN, self).__init__()


        #self.fc_linear = nn.Linear(input_dim, n_classes, bias = False)   
        # # add a convolutional layer
        # self.conv = nn.Conv2d(1, 1, kernel_size = 3, stride = 1, padding = 1)
        #self.fc1 = nn.Linear(input_dim, width, bias = False) 
        #self.fc_hidden = nn.Linear(width, width, bias = False)
        #self.fc2 = nn.Linear(width, n_classes, bias = False)   
        self.fc1_linear = nn.Linear(input_dim, n_classes, bias = False)
        # use LeCun initialization
        # nn.init.kaiming_normal_(self.fc1_linear.weight)
    def forward(self, x):

        #x = self.fc1_linear(x)
        #x = self.conv(x)
        #x = self.fc1(x) # Activation function
        #x = torch.tanh(x)
        #x = torch.relu(x)
        # x = self.fc_hidden(x)
        # x = torch.relu(x)
        #x = self.fc2(x) # Activation function
        x = self.fc1_linear(x)
        return x

# Initialize the model, loss function, and optimizer
if model_type == 'nn':
    model = SimpleNN()
elif model_type == 'linear':
    model = linearNN()

if loss_type == 'CE':
    criterion = nn.CrossEntropyLoss()  # Cross Entropy Loss
elif loss_type == 'MSE':
    #if use MSE loss, the label should be one-hot encoding
    y_tensor = F.one_hot(y_tensor, num_classes=n_classes).float()
    criterion = nn.MSELoss()  # Mean Squared Error Loss


if optimizer_type == 'adam':
    optimizer = optim.Adam(model.parameters(), lr = lr)
elif optimizer_type == 'sgd':
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum = 0)

# Training loop
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cosine_epochs,eta_min = 0) #lr / 10)

loss_list = []
for epoch in range(num_epochs):
    model.train()
    
    optimizer.zero_grad()  # Zero gradients
    outputs = model(X_tensor)  # Forward pass
    #loss = F.cross_entropy(outputs, y_tensor)  # Compute loss
    loss  = criterion(outputs, y_tensor)    
    loss.backward()  # Backward pass
    optimizer.step()  # Update weights
    scheduler.step()

    avg_loss = loss.item() / (n_total)
    loss_list.append(avg_loss)
    if (epoch + 1) % 10 == 0:
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {avg_loss:.4f}')

# Visualization
def plot_dataset(): # might not use this
    with torch.no_grad():
        model.eval()
        outputs = model(X_tensor)
        # softmax
        predicted_cls = torch.argmax(outputs, dim=1).numpy()
        groundtruth_cls = y_tensor.numpy()

        # accuracy
        accuracy = (predicted_cls == groundtruth_cls).mean()
        print(f'Accuracy: {accuracy:.2f}')
    plt.figure()
    # plt.scatter(X[:, 0], X[:, 1], c=predicted_cls, cmap='coolwarm', alpha=0.5)

    plt.scatter(X[:, 0], X[:, 1], c=groundtruth_cls, cmap='coolwarm', alpha=0.5,marker='x')
    plt.title('Classification Results')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.savefig(f'figure_gaussian/{comment}_dataset.png')

# plot loss list
plot_dataset()
plt.rcParams["axes.autolimit_mode"] = "round_numbers"
plt.rcParams["axes.xmargin"] = 0
plt.rcParams["axes.ymargin"] = 0
plt.rcParams["axes.labelsize"] = 20
plt.rcParams["axes.titlesize"] = 20
plt.rcParams["legend.fontsize"] = 10#25 #20
plt.rcParams["legend.frameon"] = True
plt.rcParams["legend.fancybox"] = True
plt.rcParams["legend.loc"] = 'upper right'
plt.rcParams["xtick.labelsize"] = 10#20
plt.rcParams["ytick.labelsize"] = 7#20
plt.rcParams["lines.markersize"] = 10
plt.rcParams["font.family"] = "serif"
plt.figure()
plt.plot(loss_list)
plt.ylim(0.15 * 1e-5, 1.2*1e-5)
# plt.title('Loss')
plt.xlabel('Iteration')
plt.ylabel('Loss')
plt.savefig(f'figure_gaussian/{comment}_loss_T_{num_epochs}.png')




'store the data into list to prevent unexpected randomness'

train_data = [{'X_train': X_tensor,  'Y_train': y_tensor}]



hessian = hessian_spectrum.Hessian(model, train_data = train_data, batch_size= 1, use_minibatch = False, gradient_accumulation_steps= 1 , device = 'cpu', comment = comment, loss = criterion)



full_hessian = hessian.get_full_hessian()

# elementwise square
full_hessian_abs = np.abs(full_hessian)
# full_hessian = full_hessian * full_hessian #np.abs(full_hessian)
#full_hessian = np.log(full_hessian)
max_value = np.max(full_hessian_abs) / visual_degree
plt.figure()
# plt.imshow(full_hessian, cmap= 'viridis', interpolation='nearest', vmax = max_value  / visual_degree , vmin = 0)
plt.imshow(full_hessian_abs, cmap= cmap, interpolation='nearest', vmax = max_value  , vmin = 0)
plt.colorbar()
#plt.title(f'Hessian of {name} step 300')
plt.savefig(f'figure_gaussian/{comment}_fullhessian_T_{num_epochs}.png',bbox_inches='tight')
plt.savefig(f'figure_gaussian/{comment}_fullhessian_T_{num_epochs}.pdf', bbox_inches='tight')
plt.close()



# average over blocks
rows_A, cols_A = full_hessian.shape
rows_B, cols_B = rows_A// input_dim, cols_A// input_dim

# Initialize array B
B = np.zeros((rows_B, cols_B))

# Calculate averages using nested loops
for i in range(rows_B):
    for j in range(cols_B):
        # Determine the indices in A for each entry of B
        row_start, row_end = i * rows_A // rows_B, (i + 1) * rows_A // rows_B
        col_start, col_end = j * cols_A // cols_B, (j + 1) * cols_A // cols_B
        
        # Calculate the average for the specified indices in A
        average_value = np.mean(full_hessian[row_start:row_end, col_start:col_end])
        
        # Assign the average to the corresponding entry in B
        B[i, j] = average_value



avg_value = np.mean(B)

med_value = np.median(B)

max_value = np.max(B)

min_value = np.min(B)



plt.figure()
# ax = sns.heatmap(full_hessian, vmin=0, vmax=1.5 * max_value, annot=False)
ax = sns.heatmap(B, vmin=0, vmax= max_value / 2 , annot=True, annot_kws={"size": 7}, cmap=cmap)

plt.savefig(f'figure_gaussian/{comment}_avg_hessian_T_{num_epochs}.png')




## plot principal block hessian
full_hessian_dic = hessian.get_full_hessian_layer_by_layer()

for idx, (name, full_hessian) in enumerate(full_hessian_dic.items()):
    print(f"frobenious norm of block hessian_{name} = {np.linalg.norm(full_hessian, 'fro' )}")

    full_hessian = np.abs(full_hessian)
    max_value = np.max(full_hessian)

    plt.figure()
    # plt.imshow(full_hessian, cmap= 'viridis', interpolation='nearest', vmax = max_value  / visual_degree , vmin = 0)
    plt.imshow(full_hessian, cmap= cmap, interpolation='nearest', vmax = max_value  / visual_degree_block_hessian[idx] , vmin = 0)
    plt.colorbar()
    #plt.title(f'Hessian of {name} step 300')
    plt.savefig(f'figure_gaussian/{comment}_block_hessian_{name}_T_{num_epochs}.png',bbox_inches='tight')
    plt.savefig(f'figure_gaussian/{comment}_block_hessian_{name}_T_{num_epochs}.pdf', bbox_inches='tight')
    plt.close()



