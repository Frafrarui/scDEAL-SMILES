import torch
from torch import nn
from torch.nn import functional as F
from torch import Tensor
import numpy as np

#import scipy.io as sio
from copy import deepcopy
        
        
class AEBase(nn.Module):
    def __init__(self,
                 input_dim,
                 latent_dim=128,
                 h_dims=[512],
                 drop_out=0.3):
                 
        super(AEBase, self).__init__()

        self.latent_dim = latent_dim

        modules = []
        hidden_dims = deepcopy(h_dims)
        
        hidden_dims.insert(0,input_dim)

        # Build Encoder
        for i in range(1,len(hidden_dims)):
            i_dim = hidden_dims[i-1]
            o_dim = hidden_dims[i]

            modules.append(
                nn.Sequential(
                    nn.Linear(i_dim, o_dim),
                    nn.BatchNorm1d(o_dim),
                    #nn.ReLU(),
                    nn.Dropout(drop_out))
            )
            #in_channels = h_dim

        self.encoder = nn.Sequential(*modules)
        self.bottleneck = nn.Linear(hidden_dims[-1], latent_dim)

        # Build Decoder
        modules = []

        self.decoder_input = nn.Linear(latent_dim, hidden_dims[-1])

        hidden_dims.reverse()

        for i in range(len(hidden_dims) - 2):
            modules.append(
                nn.Sequential(
                    nn.Linear(hidden_dims[i],
                                       hidden_dims[i + 1]),
                    nn.BatchNorm1d(hidden_dims[i + 1]),
                    #nn.ReLU(),
                    nn.Dropout(drop_out))
            )


        self.decoder = nn.Sequential(*modules)

        self.final_layer = nn.Sequential(
                            nn.Linear(hidden_dims[-2],
                                       hidden_dims[-1])
                                       ,nn.Sigmoid()
                            )
        # self.feature_extractor =nn.Sequential(
        #     self.encoder,
        #     self.bottleneck
        # )            

             
    def encode(self, input: Tensor):
        """
        Encodes the input by passing through the encoder network
        and returns the latent codes.
        """
        result = self.encoder(input)
        embedding = self.bottleneck(result)

        return embedding

    def decode(self, z: Tensor):
        """
        Maps the given latent codes
        """
        result = self.decoder_input(z)
        result = self.decoder(result)
        result = self.final_layer(result)
        return result

    def forward(self, input: Tensor, **kwargs):
        embedding = self.encode(input)
        output = self.decode(embedding)
        return  output        

# Model of Predictor
class Predictor(nn.Module):
    def __init__(self,
                 input_dim,
                 output_dim=1,
                 h_dims=[512],
                 drop_out=0.3):
                 
        super(Predictor, self).__init__()

        modules = []

        hidden_dims = deepcopy(h_dims)
        
        hidden_dims.insert(0,input_dim)

        # Build Encoder
        for i in range(1,len(hidden_dims)):
            i_dim = hidden_dims[i-1]
            o_dim = hidden_dims[i]

            modules.append(
                nn.Sequential(
                    nn.Linear(i_dim, o_dim),
                    nn.BatchNorm1d(o_dim),
                    nn.ReLU(),
                    nn.Dropout(drop_out))
            )
            #in_channels = h_dim

        self.predictor = nn.Sequential(*modules)
        #self.output = nn.Linear(hidden_dims[-1], output_dim)

        self.output = nn.Sequential(
                            nn.Linear(hidden_dims[-1],
                                       output_dim),
                                       nn.Sigmoid()
                            )            

    def forward(self, input: Tensor, **kwargs):
        embedding = self.predictor(input)
        output = self.output(embedding)
        return  output
        
        
        
    
# Model of Pretrained P
class PretrainedPredictor(AEBase):
    def __init__(self,
                 # Params from AE model
                 input_dim,
                 latent_dim=128,
                 h_dims=[512],
                 drop_out=0.3,
                 ### Parameters from predictor models
                 pretrained_weights=None,                 
                 hidden_dims_predictor=[256],
                 drop_out_predictor=0.3,
                 output_dim = 1,
                 freezed = False):
        
        # Construct an autoencoder model
        AEBase.__init__(self,input_dim,latent_dim,h_dims,drop_out)
        
        # Load pretrained weights
        if pretrained_weights !=None:
            self.load_state_dict((torch.load(pretrained_weights)))
        
        ## Free parameters until the bottleneck layer
        if freezed == True:
            bottlenect_reached = False
            for p in self.parameters():
                if ((bottlenect_reached == True)&(p.shape.numel()>self.latent_dim)):
                    break
                p.requires_grad = False
                print("Layer weight is freezed:",format(p.shape))
                # Stop until the bottleneck layer
                if p.shape.numel() == self.latent_dim:
                    bottlenect_reached = True
        # Only extract encoder
        del self.decoder
        del self.decoder_input
        del self.final_layer

        self.predictor = Predictor(input_dim=self.latent_dim,
                 output_dim=output_dim,
                 h_dims=hidden_dims_predictor,
                 drop_out=drop_out_predictor)

    def forward(self, input, **kwargs):
        embedding = self.encode(input)
        output = self.predictor(embedding)
        return  output
   
    def predict(self, embedding, **kwargs):
        output = self.predictor(embedding)
        return  output 
     

def vae_loss(recon_x, x, mu, logvar,reconstruction_function,weight=1):
    """
    recon_x: generating images
    x: origin images
    mu: latent mean
    logvar: latent log variance
    """
    BCE = reconstruction_function(recon_x, x)  # mse loss
    # loss = 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    KLD_element = mu.pow(2).add_(logvar.exp()).mul_(-1).add_(1).add_(logvar)
    KLD = torch.sum(KLD_element).mul_(-0.5)
    # KL divergence
    return BCE + KLD * weight

class VAEBase(nn.Module):
    def __init__(self,
                 input_dim,
                 latent_dim=128,
                 h_dims=[512],
                 drop_out=0.3):
                 
        super(VAEBase, self).__init__()

        self.latent_dim = latent_dim

        modules = []
    
        hidden_dims = deepcopy(h_dims)
        
        hidden_dims.insert(0,input_dim)
        
        # Build Encoder
        for i in range(1,len(hidden_dims)):
            i_dim = hidden_dims[i-1]
            o_dim = hidden_dims[i]

            modules.append(
                nn.Sequential(
                    nn.Linear(i_dim, o_dim),
                    nn.BatchNorm1d(o_dim),
                    nn.Dropout(drop_out),
                    nn.LeakyReLU()
                    )
            )
            #in_channels = h_dim

        self.encoder = nn.Sequential(*modules)
        self.fc_mu = nn.Linear(hidden_dims[-1], latent_dim)
        self.fc_var = nn.Linear(hidden_dims[-1], latent_dim)


        # Build Decoder
        modules = []

        self.decoder_input = nn.Linear(latent_dim, hidden_dims[-1])

        hidden_dims.reverse()

        for i in range(len(hidden_dims) - 2):
            modules.append(
                nn.Sequential(
                    nn.Linear(hidden_dims[i],
                                       hidden_dims[i + 1]),
                    nn.BatchNorm1d(hidden_dims[i + 1]),
                    nn.Dropout(drop_out),
                    nn.LeakyReLU()
                    )
            )


        self.decoder = nn.Sequential(*modules)

        self.final_layer = nn.Sequential(
                            nn.Linear(hidden_dims[-2],
                                       hidden_dims[-1],
                            nn.Sigmoid())
                            ) 
        # self.feature_extractor = nn.Sequential(
        #     self.encoder,
        #     self.fc_mu
        # )
    
    def encode_(self, input: Tensor):
        """
        Encodes the input by passing through the encoder network
        and returns the latent codes.
        :param input: (Tensor) Input tensor to encoder [N x C x H x W]
        :return: (Tensor) List of latent codes
        """
        result = self.encoder(input)
        #result = torch.flatten(result, start_dim=1)

        # Split the result into mu and var components
        # of the latent Gaussian distribution
        mu = self.fc_mu(result)
        log_var = self.fc_var(result)

        return [mu, log_var]
    
    def encode(self, input: Tensor,repram=False):
        """
        Encodes the input by passing through the encoder network
        and returns the latent codes.
        :param input: (Tensor) Input tensor to encoder [N x C x H x W]
        :return: (Tensor) List of latent codes
        """
        mu, log_var = self.encode_(input)

        if (repram==True):
            z = self.reparameterize(mu, log_var)
            return z
        else:
            return mu

    def decode(self, z: Tensor):
        """
        Maps the given latent codes
        onto the image space.
        :param z: (Tensor) [B x D]
        :return: (Tensor) [B x C x H x W]
        """
        result = self.decoder_input(z)
        #result = result.view(-1, 512, 2, 2)
        result = self.decoder(result)
        result = self.final_layer(result)
        return result

    def reparameterize(self, mu: Tensor, logvar: Tensor):
        """
        Reparameterization trick to sample from N(mu, var) from
        N(0,1).
        :param mu: (Tensor) Mean of the latent Gaussian [B x D]
        :param logvar: (Tensor) Standard deviation of the latent Gaussian [B x D]
        :return: (Tensor) [B x D]
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps * std + mu

    def forward(self, input: Tensor, **kwargs):
        mu, log_var = self.encode_(input)
        z = self.reparameterize(mu, log_var)
        return  [self.decode(z), input, mu, log_var]

    def loss_function(self,
                      *args,
                      **kwargs) -> dict:
        """
        Computes the VAE loss function.
        KL(N(\mu, \sigma), N(0, 1)) = \log \frac{1}{\sigma} + \frac{\sigma^2 + \mu^2}{2} - \frac{1}{2}
        :param args:
        :param kwargs:
        :return:
         M_N = self.params['batch_size']/ self.num_train_imgs,
        """
        recons = args[0]
        input = args[1]
        mu = args[2]
        log_var = args[3]

        kld_weight = kwargs['M_N'] 
        # Account for the minibatch samples from the dataset
        # M_N = self.params['batch_size']/ self.num_train_imgs,
        recons_loss =F.mse_loss(recons, input)


        kld_loss = torch.mean(-0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim = 1), dim = 0)

        loss = recons_loss + kld_weight * kld_loss
        return {'loss': loss, 'Reconstruction_Loss':recons_loss, 'KLD':-kld_loss}

    def sample(self,
               num_samples:int,
               current_device: int, **kwargs):
        """
        Samples from the latent space and return the corresponding
        image space map.
        :param num_samples: (Int) Number of samples
        :param current_device: (Int) Device to run the model
        :return: (Tensor)
        """
        z = torch.randn(num_samples,
                        self.latent_dim)

        z = z.to(current_device)

        samples = self.decode(z)
        return samples

    def generate(self, x: Tensor, **kwargs):
        """
        Given an input image x, returns the reconstructed image
        :param x: (Tensor) [B x C x H x W]
        :return: (Tensor) [B x C x H x W]
        """

        return self.forward(x)[0]

def idx2onehot(idx, n):

    assert torch.max(idx).item() < n

    if idx.dim() == 1:
        idx = idx.unsqueeze(1)
    onehot = torch.zeros(idx.size(0), n).to(idx.device)
    onehot.scatter_(1, idx, 1)
    
    return onehot

class CVAEBase(VAEBase):

    def __init__(self,
                 input_dim,
                 n_conditions,
                 latent_dim=128,
                 h_dims=[512],
                 drop_out=0.3):

        super(VAEBase, self).__init__()

        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.n_condition = n_conditions

        # There are conditions therefore input size is different
        self.encoder_dim = input_dim + n_conditions

        modules_e = []
    
        hidden_dims = deepcopy(h_dims)
        
        hidden_dims.insert(0,self.encoder_dim)
        
        # Build Encoder
        for i in range(1,len(hidden_dims)):
            i_dim = hidden_dims[i-1]
            o_dim = hidden_dims[i]

            modules_e.append(
                nn.Sequential(
                    nn.Linear(i_dim, o_dim),
                    nn.BatchNorm1d(o_dim),
                    nn.Dropout(drop_out),
                    nn.LeakyReLU()
                    )
            )
            #in_channels = h_dim

        self.encoder = nn.Sequential(*modules_e)
        self.fc_mu = nn.Linear(hidden_dims[-1], latent_dim)
        self.fc_var = nn.Linear(hidden_dims[-1], latent_dim)


        # Build Decoder
        modules_d = []

        # There are conditions therefore input size is different
        self.decoder_input = nn.Linear(latent_dim+n_conditions, hidden_dims[-1])

        # Replace the output shape
        hidden_dims.reverse()
        hidden_dims[-1]=self.input_dim

        for i in range(len(hidden_dims) - 2):
            modules_d.append(
                nn.Sequential(
                    nn.Linear(hidden_dims[i],
                                       hidden_dims[i + 1]),
                    nn.BatchNorm1d(hidden_dims[i + 1]),
                    nn.Dropout(drop_out),
                    nn.LeakyReLU()
                    )
            )


        self.decoder = nn.Sequential(*modules_d)

        self.final_layer = nn.Sequential(
                            nn.Linear(hidden_dims[-2],
                                       hidden_dims[-1],
                            nn.Sigmoid())
                            ) 
        # self.feature_extractor = nn.Sequential(
        #     self.encoder,
        #     self.fc_mu
        # )
    

    
    def forward(self, input: Tensor,c: Tensor, **kwargs):
        mu, log_var = self.encode_(input,c)
        z = self.reparameterize(mu, log_var)
        return  [self.decode(z,c), input, mu, log_var]

    def encode_(self, input: Tensor,c:Tensor):
        """
        Encodes the input by passing through the encoder network
        and returns the latent codes.
        :param input: (Tensor) Input tensor to encoder [N x C x H x W]
        :return: (Tensor) List of latent codes
        """

        # One hot encoding of inputs 
        c = idx2onehot(c, n=self.n_condition)
        input_c = torch.cat((input, c), dim=-1)

        result = self.encoder(input_c)
        #result = torch.flatten(result, start_dim=1)

        # Split the result into mu and var components
        # of the latent Gaussian distribution
        mu = self.fc_mu(result)
        log_var = self.fc_var(result)

        return [mu, log_var]
    
    def encode(self, input: Tensor,c:Tensor,repram=False):
        """
        Encodes the input by passing through the encoder network
        and returns the latent codes.
        :param input: (Tensor) Input tensor to encoder [N x C x H x W]
        :return: (Tensor) List of latent codes
        """

        # One hot encoding of inputs 
        #c = idx2onehot(c, n=self.n_condition)
        #input_c = torch.cat((input, c), dim=-1)

        mu, log_var = self.encode_(input,c)

        if (repram==True):
            z = self.reparameterize(mu, log_var)
            return z
        else:
            return mu

    def decode(self, z: Tensor,c:Tensor):
        """
        Maps the given latent codes
        onto the image space.
        :param z: (Tensor) [B x D]
        :return: (Tensor) [B x C x H x W]
        """
        
        # One hot encoding of inputs 
        c = idx2onehot(c, n=self.n_condition)
        z_c = torch.cat((z, c), dim=-1)

        result = self.decoder_input(z_c)
        #result = result.view(-1, 512, 2, 2)
        result = self.decoder(result)
        result = self.final_layer(result)
        return result

class PretrainedVAEPredictor(VAEBase):
    def __init__(self,
                 # Params from AE model
                 input_dim,
                 latent_dim=128,
                 h_dims=[512],
                 drop_out=0.3,
                 ### Parameters from predictor models
                 pretrained_weights=None,                 
                 hidden_dims_predictor=[256],
                 drop_out_predictor=0.3,
                 output_dim = 1,
                 freezed = False,
                 z_reparam=True):
        
        self.z_reparam=z_reparam
        # Construct an autoencoder model
        VAEBase.__init__(self,input_dim,latent_dim,h_dims,drop_out)
        
        # Load pretrained weights
        if pretrained_weights !=None:
            self.load_state_dict((torch.load(pretrained_weights)))
        
        ## Free parameters until the bottleneck layer
        if freezed == True:
            bottlenect_reached = False
            for p in self.parameters():
                if ((bottlenect_reached == True)&(p.shape[0]>self.latent_dim)):
                    break
                p.requires_grad = False
                print("Layer weight is freezed:",format(p.shape))
                # Stop until the bottleneck layer
                if p.shape[0] == self.latent_dim:
                    bottlenect_reached = True

        # Only extract encoder
        del self.decoder
        del self.decoder_input
        del self.final_layer

        self.predictor = Predictor(input_dim=self.latent_dim,
                 output_dim=output_dim,
                 h_dims=hidden_dims_predictor,
                 drop_out=drop_out_predictor)

        # self.feature_extractor = nn.Sequential(
        #     self.encoder,
        #     self.fc_mu
        # )

    def forward(self, input, **kwargs):
        embedding = self.encode(input,repram=self.z_reparam)
        output = self.predictor(embedding)
        return  output

    def predict(self, embedding, **kwargs):
        output = self.predictor(embedding)
        return  output

class DaNN(nn.Module):
    def __init__(self, source_model,target_model,fix_source=False):
        super(DaNN, self).__init__()
        self.source_model = source_model
        if fix_source == True:
            for p in self.parameters():
                p.requires_grad = False
                print("Layer weight is freezed:",format(p.shape))
                # Stop until the bottleneck layer
        self.target_model = target_model
    '''
    def __init__(self, source_model,target_model):
        super(DaNN, self).__init__()
        self.source_model = source_model
        self.target_model = target_model
    '''
    def forward(self, X_source, X_target,C_target=None):
     
        x_src_mmd = self.source_model.encode(X_source)

        if(type(C_target)==type(None)):
            x_tar_mmd = self.target_model.encode(X_target)
        else:
            x_tar_mmd = self.target_model.encode(X_target,C_target)

        y_src = self.source_model.predictor(x_src_mmd)
        return y_src, x_src_mmd, x_tar_mmd
    

class TargetModel(nn.Module):
    def __init__(self, source_predcitor,target_encoder):
        super(TargetModel, self).__init__()
        self.source_predcitor = source_predcitor
        self.target_encoder = target_encoder

    def forward(self, X_target,C_target=None):

        if(type(C_target)==type(None)):
            x_tar = self.target_encoder.encode(X_target)
        else:
            x_tar = self.target_encoder.encode(X_target,C_target)
        y_src = self.source_predcitor.predictor(x_tar)
        return y_src

def g_loss_function(preds, labels, mu, logvar, n_nodes, norm, pos_weight):
    cost = norm * F.binary_cross_entropy_with_logits(preds, labels, pos_weight=labels * pos_weight)

    # Check if the model is simple Graph Auto-encoder
    if logvar is None:
        return cost

    # see Appendix B from VAE paper:
    # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
    # https://arxiv.org/abs/1312.6114
    # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    KLD = -0.5 / n_nodes * torch.mean(torch.sum(
        1 + 2 * logvar - mu.pow(2) - logvar.exp().pow(2), 1))
    return cost + KLD


# ============================================================================
# SMILES-aware models for multi-drug prediction
# ============================================================================

class DrugEncoder(nn.Module):
    """
    Encodes SMILES token sequences into a fixed-size drug embedding.
    
    Architecture:
      Embedding(vocab_size, embed_dim) 
      → Flatten 
      → Linear(embed_dim * max_len, h_dim) → ReLU → Dropout
      → Linear(h_dim, drug_latent_dim)
    
    Input:  (batch, max_smiles_length) of LongTensor token indices
    Output: (batch, drug_latent_dim) float embedding
    """
    def __init__(self, vocab_size, max_smiles_len, embed_dim=32, 
                 h_dim=128, drug_latent_dim=32, drop_out=0.3):
        super(DrugEncoder, self).__init__()
        
        self.max_smiles_len = max_smiles_len
        self.drug_latent_dim = drug_latent_dim
        
        # Character embedding (index 0 = padding)
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        
        # MLP to compress flattened embeddings
        flat_dim = embed_dim * max_smiles_len
        self.fc = nn.Sequential(
            nn.Linear(flat_dim, h_dim),
            nn.ReLU(),
            nn.Dropout(drop_out),
            nn.Linear(h_dim, drug_latent_dim)
        )
    
    def forward(self, tokens):
        """
        Args:
            tokens: LongTensor of shape (batch, max_smiles_len)
        Returns:
            drug_embedding: FloatTensor of shape (batch, drug_latent_dim)
        """
        # (batch, max_len) → (batch, max_len, embed_dim)
        x = self.embedding(tokens)
        # (batch, max_len, embed_dim) → (batch, max_len * embed_dim)
        x = x.view(x.size(0), -1)
        # (batch, flat_dim) → (batch, drug_latent_dim)
        drug_embedding = self.fc(x)
        return drug_embedding


class PredictorWithDrug(nn.Module):
    """
    MLP predictor that takes concatenated [gene_embedding, drug_embedding] as input.
    Same architecture as Predictor, but with input_dim = gene_latent_dim + drug_latent_dim.
    
    Input:  gene_embedding  (batch, gene_latent_dim)
            drug_embedding  (batch, drug_latent_dim)
    Output: prediction      (batch, output_dim)  with Sigmoid
    """
    def __init__(self, gene_latent_dim, drug_latent_dim, output_dim=2,
                 h_dims=[128, 64], drop_out=0.3):
        super(PredictorWithDrug, self).__init__()
        
        self.gene_latent_dim = gene_latent_dim
        self.drug_latent_dim = drug_latent_dim
        combined_dim = gene_latent_dim + drug_latent_dim
        
        modules = []
        hidden_dims = deepcopy(h_dims)
        hidden_dims.insert(0, combined_dim)
        
        for i in range(1, len(hidden_dims)):
            i_dim = hidden_dims[i-1]
            o_dim = hidden_dims[i]
            modules.append(
                nn.Sequential(
                    nn.Linear(i_dim, o_dim),
                    nn.BatchNorm1d(o_dim),
                    nn.ReLU(),
                    nn.Dropout(drop_out))
            )
        
        self.predictor = nn.Sequential(*modules)
        self.output = nn.Sequential(
            nn.Linear(hidden_dims[-1], output_dim),
            nn.Sigmoid()
        )
    
    def forward(self, gene_emb, drug_emb, **kwargs):
        """
        Args:
            gene_emb: (batch, gene_latent_dim)
            drug_emb: (batch, drug_latent_dim)
        """
        combined = torch.cat([gene_emb, drug_emb], dim=1)
        h = self.predictor(combined)
        return self.output(h)


class PretrainedPredictorWithDrug(AEBase):
    """
    Combines:
      1. Pre-trained AEBase encoder (for gene expression)
      2. DrugEncoder (for SMILES tokens)
      3. PredictorWithDrug (for combined prediction)
    
    forward(x_gene, x_drug):
      gene_emb = encode(x_gene)           → (batch, latent_dim)
      drug_emb = drug_encoder(x_drug)     → (batch, drug_latent_dim)
      output   = predictor(gene_emb, drug_emb) → (batch, output_dim)
    """
    def __init__(self,
                 # AE params
                 input_dim, latent_dim=128, h_dims=[512], drop_out=0.3,
                 pretrained_weights=None, freezed=False,
                 # Drug encoder params
                 vocab_size=36, max_smiles_len=272, drug_embed_dim=32,
                 drug_h_dim=128, drug_latent_dim=32, drug_drop_out=0.3,
                 # Predictor params
                 hidden_dims_predictor=[128, 64], drop_out_predictor=0.3,
                 output_dim=2):
        
        # Build AE encoder
        AEBase.__init__(self, input_dim, latent_dim, h_dims, drop_out)
        
        # Load pretrained AE weights
        if pretrained_weights is not None:
            self.load_state_dict(torch.load(pretrained_weights))
        
        # Freeze encoder if requested
        if freezed:
            bottleneck_reached = False
            for p in self.parameters():
                if bottleneck_reached and p.shape.numel() > self.latent_dim:
                    break
                p.requires_grad = False
                if p.shape.numel() == self.latent_dim:
                    bottleneck_reached = True
        
        # Remove decoder (only need encoder)
        del self.decoder
        del self.decoder_input
        del self.final_layer
        
        # Drug encoder
        self.drug_encoder = DrugEncoder(
            vocab_size=vocab_size,
            max_smiles_len=max_smiles_len,
            embed_dim=drug_embed_dim,
            h_dim=drug_h_dim,
            drug_latent_dim=drug_latent_dim,
            drop_out=drug_drop_out
        )
        
        # Combined predictor
        self.predictor = PredictorWithDrug(
            gene_latent_dim=latent_dim,
            drug_latent_dim=drug_latent_dim,
            output_dim=output_dim,
            h_dims=hidden_dims_predictor,
            drop_out=drop_out_predictor
        )
    
    def forward(self, x_gene, x_drug, **kwargs):
        """
        Args:
            x_gene: FloatTensor (batch, input_dim) - gene expression
            x_drug: LongTensor (batch, max_smiles_len) - SMILES tokens
        Returns:
            output: (batch, output_dim) - prediction probabilities
        """
        gene_emb = self.encode(x_gene)
        drug_emb = self.drug_encoder(x_drug)
        output = self.predictor(gene_emb, drug_emb)
        return output
    
    def predict(self, gene_embedding, drug_embedding, **kwargs):
        """Predict from pre-computed embeddings."""
        output = self.predictor(gene_embedding, drug_embedding)
        return output


class DaNNWithDrug(nn.Module):
    """
    Domain Adaptation Neural Network with Drug (SMILES) support.
    
    Extends DaNN to pass drug tokens through the source model's drug encoder
    for classification, while MMD operates only on gene embeddings.
    
    forward(X_source, X_target, drug_tokens):
      x_src_mmd = source_model.encode(X_source)      → gene emb (bulk)
      x_tar_mmd = target_model.encode(X_target)       → gene emb (SC)
      drug_emb  = source_model.drug_encoder(drug_tokens)
      y_src     = source_model.predictor(x_src_mmd, drug_emb)
      return y_src, x_src_mmd, x_tar_mmd
    """
    def __init__(self, source_model, target_model, fix_source=False):
        super(DaNNWithDrug, self).__init__()
        self.source_model = source_model
        if fix_source:
            for p in self.source_model.parameters():
                p.requires_grad = False
        self.target_model = target_model
    
    def forward(self, X_source, X_target, drug_tokens, C_target=None):
        """
        Args:
            X_source:    FloatTensor (batch_src, n_genes_bulk)
            X_target:    FloatTensor (batch_tar, n_genes_sc)
            drug_tokens: LongTensor (batch_src, max_smiles_len) - SMILES for each source sample
            C_target:    Optional condition labels for CVAE
        Returns:
            y_src:     (batch_src, output_dim) classification output
            x_src_mmd: (batch_src, latent_dim) bulk gene embedding
            x_tar_mmd: (batch_tar, latent_dim) SC gene embedding
        """
        # Encode gene expressions
        x_src_mmd = self.source_model.encode(X_source)
        
        if C_target is not None:
            x_tar_mmd = self.target_model.encode(X_target, C_target)
        else:
            x_tar_mmd = self.target_model.encode(X_target)
        
        # Encode drug and predict
        drug_emb = self.source_model.drug_encoder(drug_tokens)
        y_src = self.source_model.predictor(x_src_mmd, drug_emb)
        
        return y_src, x_src_mmd, x_tar_mmd


class TargetModelWithDrug(nn.Module):
    """
    For prediction on single-cell data with any drug.
    Uses the SC encoder + source model's drug encoder + predictor.
    
    forward(X_target, drug_tokens):
      gene_emb = target_encoder.encode(X_target)
      drug_emb = source_predictor.drug_encoder(drug_tokens)
      y = source_predictor.predictor(gene_emb, drug_emb)
    """
    def __init__(self, source_predictor, target_encoder):
        super(TargetModelWithDrug, self).__init__()
        self.source_predictor = source_predictor
        self.target_encoder = target_encoder
    
    def forward(self, X_target, drug_tokens, C_target=None):
        """
        Args:
            X_target:    FloatTensor (batch, n_genes_sc)
            drug_tokens: LongTensor (batch, max_smiles_len)
            C_target:    Optional condition labels
        """
        if C_target is not None:
            gene_emb = self.target_encoder.encode(X_target, C_target)
        else:
            gene_emb = self.target_encoder.encode(X_target)
        
        drug_emb = self.source_predictor.drug_encoder(drug_tokens)
        y = self.source_predictor.predictor(gene_emb, drug_emb)
        return y


# ============================================================================
# Sanity check for all SMILES models
# ============================================================================

if __name__ == '__main__':
    print("=" * 70)
    print("SANITY CHECK: models.py (SMILES-aware classes)")
    print("=" * 70)
    
    device = 'cpu'
    batch_size = 16
    
    # Dimensions matching real data
    n_genes = 2000       # typical HVG count
    latent_dim = 32      # bottleneck
    vocab_size = 36      # from smiles_encoder
    max_smiles_len = 272 # from smiles_encoder
    drug_latent_dim = 32
    output_dim = 2       # binary classification
    
    # Dummy tensors
    x_gene = torch.randn(batch_size, n_genes).to(device)
    x_drug = torch.randint(0, vocab_size, (batch_size, max_smiles_len)).to(device)
    
    print(f"\nInput shapes:")
    print(f"  x_gene:  {x_gene.shape}  (batch, n_genes)")
    print(f"  x_drug:  {x_drug.shape}  (batch, max_smiles_len)")
    
    # --- Test DrugEncoder ---
    print(f"\n--- DrugEncoder ---")
    drug_enc = DrugEncoder(vocab_size=vocab_size, max_smiles_len=max_smiles_len,
                           embed_dim=32, h_dim=128, drug_latent_dim=drug_latent_dim).to(device)
    drug_emb = drug_enc(x_drug)
    print(f"  Output shape: {drug_emb.shape}  (expected: [{batch_size}, {drug_latent_dim}])")
    assert drug_emb.shape == (batch_size, drug_latent_dim), "❌ Shape mismatch!"
    print(f"  ✅ DrugEncoder OK")
    
    # --- Test PredictorWithDrug ---
    print(f"\n--- PredictorWithDrug ---")
    gene_emb_dummy = torch.randn(batch_size, latent_dim).to(device)
    pred_drug = PredictorWithDrug(gene_latent_dim=latent_dim, drug_latent_dim=drug_latent_dim,
                                   output_dim=output_dim, h_dims=[128, 64]).to(device)
    pred_out = pred_drug(gene_emb_dummy, drug_emb)
    print(f"  Output shape: {pred_out.shape}  (expected: [{batch_size}, {output_dim}])")
    assert pred_out.shape == (batch_size, output_dim), "❌ Shape mismatch!"
    print(f"  ✅ PredictorWithDrug OK")
    
    # --- Test PretrainedPredictorWithDrug ---
    print(f"\n--- PretrainedPredictorWithDrug ---")
    model = PretrainedPredictorWithDrug(
        input_dim=n_genes, latent_dim=latent_dim, h_dims=[512, 256], drop_out=0.3,
        vocab_size=vocab_size, max_smiles_len=max_smiles_len,
        drug_latent_dim=drug_latent_dim,
        hidden_dims_predictor=[128, 64], output_dim=output_dim
    ).to(device)
    
    out = model(x_gene, x_drug)
    print(f"  Output shape: {out.shape}  (expected: [{batch_size}, {output_dim}])")
    assert out.shape == (batch_size, output_dim), "❌ Shape mismatch!"
    
    # Test encode + predict separately
    gene_emb_test = model.encode(x_gene)
    drug_emb_test = model.drug_encoder(x_drug)
    out_sep = model.predict(gene_emb_test, drug_emb_test)
    print(f"  Separate encode+predict: {out_sep.shape}")
    assert out_sep.shape == (batch_size, output_dim), "❌ Shape mismatch!"
    print(f"  ✅ PretrainedPredictorWithDrug OK")
    
    # --- Test DaNNWithDrug ---
    print(f"\n--- DaNNWithDrug ---")
    n_genes_sc = 1500  # SC may have different gene count
    x_target = torch.randn(batch_size, n_genes_sc).to(device)
    
    target_encoder = AEBase(input_dim=n_genes_sc, latent_dim=latent_dim,
                            h_dims=[512, 256]).to(device)
    
    dann_model = DaNNWithDrug(source_model=model, target_model=target_encoder).to(device)
    y_src, x_src_mmd, x_tar_mmd = dann_model(x_gene, x_target, x_drug)
    
    print(f"  y_src shape:     {y_src.shape}      (expected: [{batch_size}, {output_dim}])")
    print(f"  x_src_mmd shape: {x_src_mmd.shape}  (expected: [{batch_size}, {latent_dim}])")
    print(f"  x_tar_mmd shape: {x_tar_mmd.shape}  (expected: [{batch_size}, {latent_dim}])")
    assert y_src.shape == (batch_size, output_dim)
    assert x_src_mmd.shape == (batch_size, latent_dim)
    assert x_tar_mmd.shape == (batch_size, latent_dim)
    print(f"  ✅ DaNNWithDrug OK")
    
    # --- Test TargetModelWithDrug ---
    print(f"\n--- TargetModelWithDrug ---")
    target_pred = TargetModelWithDrug(source_predictor=model, target_encoder=target_encoder).to(device)
    y_pred = target_pred(x_target, x_drug)
    print(f"  Output shape: {y_pred.shape}  (expected: [{batch_size}, {output_dim}])")
    assert y_pred.shape == (batch_size, output_dim)
    print(f"  ✅ TargetModelWithDrug OK")
    
    print(f"\n{'=' * 70}")
    print(f"✅ ALL SANITY CHECKS PASSED")
    print(f"{'=' * 70}")