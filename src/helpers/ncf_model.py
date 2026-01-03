# ============================================================================
# STEP 5: NCF MODEL ARCHITECTURE
# ============================================================================
import torch
import torch.nn as nn

"""
This step implements the Neural Collaborative Filtering (NCF) model.

The NCF model has three variants:
1. GMF: Only Generalized Matrix Factorization (linear)
2. MLP: Only Multi-Layer Perceptron (non-linear)
3. NeuMF: Neural Matrix Factorization (combines GMF + MLP)
"""

class NCF(nn.Module):
    """
    Neural Collaborative Filtering Model
    
    This model learns user and item embeddings and combines them using
    either GMF (linear) or MLP (non-linear) or both (NeuMF).
    
    Architecture:
    1. Embedding layers: Convert user/item IDs to dense vectors
    2. GMF path: Element-wise product of embeddings (linear interaction)
    3. MLP path: Deep neural network (non-linear interaction)
    4. Prediction layer: Combines GMF and/or MLP outputs to predict score
    """
    
    def __init__(self, user_num, item_num, factor_num, num_layers,
                 dropout, model_name, GMF_model=None, MLP_model=None):
        """
        Initialize the NCF model.
        
        Parameters:
        - user_num: Total number of users
        - item_num: Total number of items
        - factor_num: Dimension of embedding vectors (e.g., 32)
        - num_layers: Number of layers in MLP component
        - dropout: Dropout rate for regularization
        - model_name: 'MLP', 'GMF', 'NeuMF-end', or 'NeuMF-pre'
        - GMF_model: Pre-trained GMF model (for NeuMF-pre)
        - MLP_model: Pre-trained MLP model (for NeuMF-pre)
        """
        super(NCF, self).__init__()
        
        # Store configuration
        self.dropout = dropout
        self.model_name = model_name
        self.GMF_model = GMF_model
        self.MLP_model = MLP_model
        
        # ====================================================================
        # EMBEDDING LAYERS
        # ====================================================================
        # Embeddings convert user/item IDs (integers) to dense vectors
        
        # GMF embeddings: factor_num dimensions
        # Used for Generalized Matrix Factorization (linear interactions)
        if model_name != 'MLP':  # MLP doesn't use GMF
            self.embed_user_GMF = nn.Embedding(user_num, factor_num)
            self.embed_item_GMF = nn.Embedding(item_num, factor_num)
        
        # MLP embeddings: Larger dimension for deeper networks
        # Dimension = factor_num * 2^(num_layers-1)
        # Example: factor_num=32, num_layers=3 → 32 * 2^2 = 128 dimensions
        if model_name != 'GMF':  # GMF doesn't use MLP
            mlp_embed_dim = factor_num * (2 ** (num_layers - 1))
            self.embed_user_MLP = nn.Embedding(user_num, mlp_embed_dim)
            self.embed_item_MLP = nn.Embedding(item_num, mlp_embed_dim)
        
        # ====================================================================
        # MLP LAYERS (Multi-Layer Perceptron)
        # ====================================================================
        # Build MLP with decreasing dimensions
        # Example with factor_num=32, num_layers=3:
        #   Input: 128*2 = 256 (concatenated user + item embeddings)
        #   Layer 1: 256 → 128
        #   Layer 2: 128 → 64
        #   Layer 3: 64 → 32
        #   Output: 32 dimensions
        
        if model_name != 'GMF':  # GMF doesn't use MLP
            MLP_modules = []
            for i in range(num_layers):
                # Calculate input size for this layer
                input_size = factor_num * (2 ** (num_layers - i))
                
                # Add dropout for regularization
                MLP_modules.append(nn.Dropout(p=self.dropout))
                
                # Add linear layer (halves the dimension)
                MLP_modules.append(nn.Linear(input_size, input_size // 2))
                
                # Add ReLU activation (non-linearity)
                MLP_modules.append(nn.ReLU())
            
            # Combine all MLP layers into a sequential module
            self.MLP_layers = nn.Sequential(*MLP_modules)
        
        # ====================================================================
        # PREDICTION LAYER
        # ====================================================================
        # Final layer that outputs the interaction score
        
        if self.model_name in ['MLP', 'GMF']:
            # Single path: just factor_num dimensions
            predict_size = factor_num
        else:
            # NeuMF: concatenate GMF (factor_num) + MLP (factor_num) = 2*factor_num
            predict_size = factor_num * 2
        
        self.predict_layer = nn.Linear(predict_size, 1)
        
        # Initialize weights
        self._init_weight_()
    
    def _init_weight_(self):
        """
        Initialize model weights.
        
        Different initialization strategies:
        - Embeddings: Small random values (std=0.01)
        - MLP layers: Xavier uniform (good for ReLU)
        - Prediction layer: Kaiming uniform (good for sigmoid)
        - Biases: Zero
        """
        if not self.model_name == 'NeuMF-pre':
            # Random initialization for training from scratch
            
            # Embedding initialization: Small random values
            # This prevents embeddings from starting too large
            if hasattr(self, 'embed_user_GMF'):
                nn.init.normal_(self.embed_user_GMF.weight, std=0.01)
            if hasattr(self, 'embed_item_GMF'):
                nn.init.normal_(self.embed_item_GMF.weight, std=0.01)
            if hasattr(self, 'embed_user_MLP'):
                nn.init.normal_(self.embed_user_MLP.weight, std=0.01)
            if hasattr(self, 'embed_item_MLP'):
                nn.init.normal_(self.embed_item_MLP.weight, std=0.01)
            
            # MLP layer initialization: Xavier uniform
            # Good for layers with ReLU activation
            if hasattr(self, 'MLP_layers'):
                for m in self.MLP_layers:
                    if isinstance(m, nn.Linear):
                        nn.init.xavier_uniform_(m.weight)
            
            # Prediction layer initialization: Kaiming uniform
            # Good for layers before sigmoid activation
            nn.init.kaiming_uniform_(self.predict_layer.weight, 
                                    a=1, nonlinearity='sigmoid')
            
            # Initialize all biases to zero
            for m in self.modules():
                if isinstance(m, nn.Linear) and m.bias is not None:
                    m.bias.data.zero_()
        else:
            # Pre-trained initialization (for NeuMF-pre)
            # Copy weights from pre-trained GMF and MLP models
            
            # Copy embedding weights
            self.embed_user_GMF.weight.data.copy_(
                self.GMF_model.embed_user_GMF.weight)
            self.embed_item_GMF.weight.data.copy_(
                self.GMF_model.embed_item_GMF.weight)
            self.embed_user_MLP.weight.data.copy_(
                self.MLP_model.embed_user_MLP.weight)
            self.embed_item_MLP.weight.data.copy_(
                self.MLP_model.embed_item_MLP.weight)
            
            # Copy MLP layer weights
            for (m1, m2) in zip(self.MLP_layers, self.MLP_model.MLP_layers):
                if isinstance(m1, nn.Linear) and isinstance(m2, nn.Linear):
                    m1.weight.data.copy_(m2.weight)
                    m1.bias.data.copy_(m2.bias)
            
            # Combine prediction layer weights from GMF and MLP
            predict_weight = torch.cat([
                self.GMF_model.predict_layer.weight, 
                self.MLP_model.predict_layer.weight], dim=1)
            predict_bias = (self.GMF_model.predict_layer.bias + 
                           self.MLP_model.predict_layer.bias) / 2
            
            self.predict_layer.weight.data.copy_(0.5 * predict_weight)
            self.predict_layer.bias.data.copy_(predict_bias)
    
    def forward(self, user, item):
        """
        Forward pass: Predict user-item interaction scores.
        
        Parameters:
        - user: Tensor of user IDs [batch_size]
        - item: Tensor of item IDs [batch_size]
        
        Returns:
        - prediction: Tensor of predicted scores [batch_size]
        """
        # ====================================================================
        # GMF PATH (Generalized Matrix Factorization)
        # ====================================================================
        # Linear interaction: element-wise product of embeddings
        # Similar to traditional matrix factorization
        
        if self.model_name != 'MLP':
            # Get embeddings
            embed_user_GMF = self.embed_user_GMF(user)  # [batch_size, factor_num]
            embed_item_GMF = self.embed_item_GMF(item)  # [batch_size, factor_num]
            
            # Element-wise product (linear interaction)
            output_GMF = embed_user_GMF * embed_item_GMF  # [batch_size, factor_num]
        
        # ====================================================================
        # MLP PATH (Multi-Layer Perceptron)
        # ====================================================================
        # Non-linear interaction: deep neural network
        
        if self.model_name != 'GMF':
            # Get embeddings
            embed_user_MLP = self.embed_user_MLP(user)  # [batch_size, mlp_dim]
            embed_item_MLP = self.embed_item_MLP(item)   # [batch_size, mlp_dim]
            
            # Concatenate user and item embeddings
            interaction = torch.cat((embed_user_MLP, embed_item_MLP), -1)  # [batch_size, mlp_dim*2]
            
            # Pass through MLP layers (with dropout and ReLU)
            output_MLP = self.MLP_layers(interaction)  # [batch_size, factor_num]
        
        # ====================================================================
        # COMBINE PATHS
        # ====================================================================
        if self.model_name == 'GMF':
            # Only GMF path
            concat = output_GMF
        elif self.model_name == 'MLP':
            # Only MLP path
            concat = output_MLP
        else:
            # NeuMF: Concatenate both paths
            concat = torch.cat((output_GMF, output_MLP), -1)  # [batch_size, factor_num*2]
        
        # ====================================================================
        # PREDICTION
        # ====================================================================
        # Final linear layer outputs interaction score
        prediction = self.predict_layer(concat)  # [batch_size, 1]
        
        # Flatten to [batch_size]
        return prediction.view(-1)