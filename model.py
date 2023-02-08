class OnlineUpdateModel(ABC):
    @abstractmethod
    def forward(self,
                input: np.ndarray) -> np.ndarray:
        raise NotImplementedError()

class GLNBase(OnlineUpdateModel):
    def __init__(self,
                 layer_sizes: Sequence[int],
                 input_size: int,
                 context_map_size: int = 6,
                 base_predictor: Optional[Callable[[np.ndarray], np.ndarray]] = None,
                 learning_rate: float = 1e-3,
                 weight_clipping: float = 10000000,
                 base_variances: float=0.5):
        super().__init__()
        assert all(size > 0 for size in layer_sizes)
        self.layer_sizes = tuple(layer_sizes)
        assert input_size > 0
        self.input_size = int(input_size)
        assert context_map_size > 0
        self.context_map_size = int(context_map_size)
        self.base_variances=base_variances
        self.base_pred_size = self.input_size
        self.base_predictor=base_predictor
        assert not isinstance(learning_rate, float) or learning_rate > 0.0
        self.learning_rate = learning_rate
        assert weight_clipping > 0.0
        self.weight_clipping = float(weight_clipping)

class DynamicParameter(nn.Module):
    def __init__(self, name: Optional[str] = None):
        super().__init__()
        self.step = 0
        self.name = name

    @property
    def value(self):
        self.step += 1
        return self.step

class ConstantParameter(DynamicParameter):
    def __init__(self, constant_value: float, name: Optional[str] = None):
        super().__init__(name)
        assert isinstance(constant_value, float)
        self.constant_value = constant_value
    @property
    def value(self):
        return self.constant_value

class Linear(nn.Module):
    def __init__(self,
                 size: int,
                 input_size: int,
                 context_size: int,
                 context_map_size: int,
                 learning_rate: DynamicParameter,
                 min_sigma_sq:float,
                 max_sigma_sq:float,
                 weight_clipping: float):
        super().__init__()

        assert size > 0 and input_size > 0 and context_size > 0
        assert context_map_size > 0
        self.context_map_size = context_map_size
        self.learning_rate = learning_rate
        self.weight_clipping = weight_clipping
        self.size = size if size>1 else 1
        self._context_maps = torch.as_tensor(np.random.normal(size=(1, self.size,context_map_size, context_size)),dtype=torch.float32)
        context_bias_shape = (1, self.size,context_map_size, 1)
        self._context_bias = torch.tensor(np.random.normal(size=context_bias_shape),dtype=torch.float32)
        self._context_maps /= torch.norm(self._context_maps,dim=-1,keepdim=True)
        self._context_maps = nn.Parameter(self._context_maps,requires_grad=False)
        self._context_bias = nn.Parameter(self._context_bias,requires_grad=False)
        self._boolean_converter = nn.Parameter(torch.as_tensor(np.array([[2**i] for i in range(context_map_size)])),requires_grad=False)
        weights_shape = (1, self.size, 2**context_map_size,input_size)
        self._weights = nn.Parameter(torch.randn(size=weights_shape),requires_grad=True)
        #self._weights =torch.randn(size=weights_shape)
        self.bias_mu=torch.tensor(5.0,dtype=torch.float32)
        self.bias_variance=torch.tensor(0.5,dtype=torch.float32)
        self.min_sigma_sq=0.0000000001
        self.max_sigma_sq=100000000
        self.min_weight=-100000
        if torch.cuda.is_available():
            self.cuda()

    def mu_sigma_giver(self,mu,sigma_sq,current_selected_weights,size):
       output_mu=torch.zeros(size=(current_selected_weights.shape[2],current_selected_weights.shape[1],1),requires_grad=True)
       output_sigma_sq=torch.zeros(size=(current_selected_weights.shape[2],current_selected_weights.shape[1],1),requires_grad=True)
       temp=torch.zeros(size=(current_selected_weights.shape[2],current_selected_weights.shape[3],1),requires_grad=True)         
       sigma_sq=sigma_sq.pow(-1)
       output_sigma_sq1=output_sigma_sq.clone()
       for i in range(0,self.size):
           temp=current_selected_weights[:,i,:,:].reshape(current_selected_weights.shape[2],current_selected_weights.shape[3],1).clone()
           temp=torch.mul(temp,sigma_sq).clone()
           output_sigma_sq1=output_sigma_sq1.clone()
           output_sigma_sq1[:,i,:]=torch.sum(temp,dim=1).clone()
       output_sigma_sq1=output_sigma_sq1.pow(-1)
       output_mu1=output_mu.clone()
       for i in range(0,self.size):  
           temp=current_selected_weights[:,i,:,:].reshape(current_selected_weights.shape[2],current_selected_weights.shape[3],1).clone()
           #temp=current_selected_weights[:,i,:,:].clone()
           temp=torch.mul(temp,sigma_sq).clone()
           temp=torch.mul(temp,mu).clone()
           output_mu1=output_mu1.clone()
           output_mu1[:,i,:]=torch.mul(torch.sum(temp,dim=1),output_sigma_sq1[:,i,:]).clone()
       return output_mu1,output_sigma_sq1                

    def forward(self,mu,sigma_sq, context):
        distances = torch.matmul(self._context_maps, context.T)
        mapped_context_binary = (distances > self._context_bias).int()
        current_context_indices = torch.sum(mapped_context_binary *self._boolean_converter,dim=-2)
        current_selected_weights=self._weights[torch.arange(1).reshape(-1, 1, 1),torch.arange(self.size).reshape(1, -1, 1), current_context_indices, :].clone()
        #self.current_selected_weights= nn.Parameter(current_selected_weights,requires_grad=True)
        self.current_selected_weights=current_selected_weights
        current_selected_weights = torch.clamp(current_selected_weights,min=-self.weight_clipping,max=self.weight_clipping)   
        output_mu,output_sigma_sq=self.mu_sigma_giver(mu,sigma_sq,current_selected_weights.clone(),self.size) 
        return output_mu,output_sigma_sq

class GLN(nn.Module, GLNBase):
    def __init__(self,
                 layer_sizes: Sequence[int],
                 input_size: int,
                 context_map_size: int = 4,
                 base_predictor: Optional[
                     Callable[[np.ndarray], np.ndarray]] = None,
                 learning_rate: Union[float, DynamicParameter] = 1e-3,
                 min_sigma_sq=0.5,
                 max_sigma_sq=100,
                 weight_clipping: float = 10000000):

        nn.Module.__init__(self)
        GLNBase.__init__(self, layer_sizes, input_size,
                         context_map_size, base_predictor,
                         learning_rate,weight_clipping)
        self.layers = nn.ModuleList()
        previous_size = self.base_pred_size
        self.min_sigma_sq=min_sigma_sq
        self.max_sigma_sq=max_sigma_sq

        if isinstance(learning_rate, float):
            self.learning_rate = ConstantParameter(learning_rate,'learning_rate')
        elif isinstance(learning_rate, DynamicParameter):
            self.learning_rate = learning_rate
        else:
            raise ValueError('Invalid learning rate')
        for size in self.layer_sizes:
            layer = Linear(size, previous_size, self.input_size,
                           self.context_map_size,self.learning_rate,
                           self.min_sigma_sq,self.max_sigma_sq,
                           self.weight_clipping)
            self.layers.append(layer)
            previous_size = size
        if torch.cuda.is_available():
            self.cuda()

    def forward(self,input: np.ndarray)-> np.ndarray:
        #base_preds_mu,base_preds_sigma_sq= self.base_model(input)
        base_preds_mu=input.reshape(input.shape[0],input.shape[1],1)
        base_preds_sigma_sq=torch.tensor(self.base_variances*np.ones(shape=(input.shape[0],self.input_size)),dtype=torch.float32).reshape(input.shape[0],input.shape[1],1)
        input=input.float()
        if torch.cuda.is_available():
            input = input.cuda()
            base_preds_mu = base_preds_mu.cuda()
            base_preds_sigma_sq = base_preds_sigma_sq.cuda()
        context = input
        for layer in self.layers:
            base_preds_mu,base_preds_sigma_sq = layer.forward(mu=base_preds_mu,sigma_sq=base_preds_sigma_sq,context=context)
        return base_preds_mu.squeeze(dim=2),base_preds_sigma_sq.squeeze(dim=2)


import torchvision.transforms.functional as TF

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)

class Encoder(nn.Module):
  def __init__(
            self, in_channels=3, out_channels=1,features=[4, 8, 16, 32]
    ):
        super(Encoder, self).__init__()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.downs=(DoubleConv(in_channels, features[0]))
        self.latent_dims=4
        self.c=4
        self.model=GLN(layer_sizes=[self.latent_dims],input_size=self.latent_dims,context_map_size=4,learning_rate=0.001)
        self.fc_mu = nn.Linear(in_features=self.c*32*32, out_features=self.latent_dims)

  def forward(self, x):
        x=self.downs(x)
        x_mu=self.pool(x)
        x_mu = x_mu.view(x_mu.size(0), -1)
        x_mu = self.fc_mu(x_mu)
        x_mu,x_var=self.model.forward(x_mu)
        return x,x_mu,x_var        

class Decoder(nn.Module):
  def __init__(
            self, in_channels=3, out_channels=1,features=[4, 8, 16, 32]
    ):
        super(Decoder, self).__init__()
        self.ups = nn.ModuleList()
        self.downs = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        for feature in features:
            self.downs.append(DoubleConv(in_channels, feature))
            in_channels = feature
        for feature in reversed(features):
            self.ups.append(
                nn.ConvTranspose2d(
                    feature*2, feature, kernel_size=2, stride=2,
                )
            )
            self.ups.append(DoubleConv(feature*2, feature))
        self.bottleneck = DoubleConv(features[-1], features[-1]*2)
        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)
        self.latent_dims=4
        self.bilinear=nn.Upsample(mode='bilinear',size=(65,65))
        self.c=4
        self.fc=nn.Linear(in_features=self.latent_dims, out_features=self.latent_dims*32*32)

  def forward(self,x,x_mu):
        skip_connections=[]
        x_mu=self.fc(x_mu)
        x_mu = x_mu.view(x_mu.size(0),self.c,32,32)
        x_mu=self.bilinear(x_mu)
        skip_connections.append(x_mu)
        x=self.pool(x)
        for i in range(1,len(self.downs)):
            x = self.downs[i](x)
            skip_connections.append(x)
            x = self.pool(x)
        x = self.bottleneck(x)
        skip_connections = skip_connections[::-1]
        for idx in range(0, len(self.ups), 2):
            x = self.ups[idx](x)
            skip_connection = skip_connections[idx//2]
            if x.shape != skip_connection.shape:
                x = TF.resize(x, size=skip_connection.shape[2:])
            concat_skip = torch.cat((skip_connection, x), dim=1)
            x = self.ups[idx+1](concat_skip)
        return self.final_conv(x)

class UNET(nn.Module):
    def __init__(
            self, in_channels=3, out_channels=1, features=[4, 8, 16, 32]
    ):
        super(UNET, self).__init__()
        self.in_channels=in_channels
        self.out_channels=out_channels
        self.features=features
        self.encoder=Encoder(self.in_channels,self.out_channels,self.features)
        self.decoder=Decoder(self.in_channels,self.out_channels,self.features)

    def forward(self, x):
        x,x_mu,x_var=self.encoder(x)
        x=self.decoder(x,x_mu)
        return x