import torch
from torch import nn
from base_models import base_model


class model_semiCNN(nn.Module):
    def __init__(self, num_class=101, num_segments=1,
                 model_name='semi_c3d', model_depth=None, 
                 new_length=16, pretrained=True):
        super(model_semiCNN, self).__init__()
        
        self.num_class = num_class
        self.num_segments = num_segments
        self.new_length = new_length*num_segments
        self.model_name = model_name
        self.model_depth = model_depth
        self.pretrained = pretrained
        self.input_size = 224
        self._generate_base_model()
        

    def _generate_base_model(self):
       
        assert self.model_name in ['semi_c3d', 'semi_resnet', 'semi_densenet']
        self.base_model = base_model(self.model_name, self.model_depth, self.num_class, self.pretrained)


    def forward(self, input):

        input = input.contiguous().view(-1, self.new_length, 3, self.input_size, self.input_size)
        input = input.permute(0,2,1,3,4)
        output = self.base_model(input.view(-1, 3, self.new_length, self.input_size, self.input_size))

        return output



# test
if __name__=='__main__':

    model = model_semiCNN(model_name='semi_resnet', model_depth=18)
    input = torch.randn(1, 48, 224, 224)
    output = model(sample_input)
    print('output: ', output.shape)

