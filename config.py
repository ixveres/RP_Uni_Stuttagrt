class Config:
    def __init__(self):
        # Transformer
        self.n_testing_samples = 6
        self.epochs = 100
        self.batch_size = 10
        self.lr = 5e-4
        self.print_every = 10

        self.embed_dim = 32
        self.group_size = 4
        self.num_heads = 4
        self.dropout_rate = 0.0

        # GMM
        self.epochs_gmm = 500
        self.lr_gmm = 1e-2
        self.k = 128 // 4
        self.seeds = 1000
        self.feature_size = 6

        # Poisson equation
        self.domain_min = -1.0
        self.domain_max = 1.0

        # VAE
        self.epochs_vae = 100
        self.lr_vae = 5e-3
        self.hidden_dim = 64
        self.latent_dim = 32
        self.codebook_size = 70

