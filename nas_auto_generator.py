###################################################################################################
#
# Copyright (C) Maxim Integrated Products, Inc. All Rights Reserved.
#
# Maxim Integrated Products, Inc. Default Copyright Notice:
# https://www.maximintegrated.com/en/aboutus/legal/copyrights.html
#
###################################################################################################
"""
Model autogenerator script from NAS results.
"""
import argparse
import json
import os


class AutoGen:
    """
    Class for auto generator from NAS to model file
    """
    def __init__(self, model_name, arch_dict):
        self.f = open(os.path.join('models', (model_name.lower()+'.py')), "w+")
        self.model_name = model_name
        self.arch_dict = arch_dict

        if self.arch_dict['type'].lower() == 'conv2d':
            if self.arch_dict['bn']:
                self.pool_layer = 'ai8x.FusedMaxPoolConv2dBNReLU'
                self.layer = 'ai8x.FusedConv2dBNReLU'
            else:
                self.pool_layer = 'ai8x.FusedMaxPoolConv2dReLU'
                self.layer = 'ai8x.FusedConv2dReLU'
        elif self.arch_dict['type'].lower() == 'conv1d':
            if self.arch_dict['bn']:
                self.pool_layer = 'ai8x.FusedMaxPoolConv1dBNReLU'
                self.layer = 'ai8x.FusedConv1dBNReLU'
            else:
                self.pool_layer = 'ai8x.FusedMaxPoolConv1dReLU'
                self.layer = 'ai8x.FusedConv1dReLU'
        else:
            print('Architecure type must be either Conv1d or Conv2d')

    def write_line(self, string):
        """Appends a line and a linebreak"""
        self.f.write(string)
        self.f.write(os.linesep)

    def append_line(self, string):
        """Appends a line"""
        self.f.write(string)

    def generate(self):
        """Function that fills the model file"""
        ind = '    '
        self.write_line('#####################################################' +
                        '##############################################')
        self.write_line('#')
        self.write_line('# Copyright (C) Maxim Integrated Products,' +
                        ' Inc. All Rights Reserved.')
        self.write_line('#')
        self.write_line('# Maxim Integrated Products, Inc. Default Copyright Notice:')
        self.write_line('# https://www.maximintegrated.com/en/aboutus/legal/copyrights.html')
        self.write_line('#')
        self.write_line('#####################################################' +
                        '##############################################')
        self.write_line('')
        self.write_line('import torch.nn as nn')
        self.write_line('')
        self.write_line('import ai8x')
        self.write_line('')
        self.write_line('')
        self.write_line(f'class {self.model_name.upper()}(nn.Module):')
        self.write_line('')
        self.write_line(ind+'def __init__(')
        self.write_line(3*ind+'self,')
        self.write_line(3*ind+'num_classes,')
        self.write_line(3*ind+f'num_channels={self.arch_dict["in_shape"][0]},')
        self.write_line(3*ind+f'dimensions=({self.arch_dict["in_shape"][1]},' +
                        f' {self.arch_dict["in_shape"][2]}),')
        self.write_line(3*ind+f'bias={self.arch_dict["bias_list"][0][0]},')
        self.write_line(3*ind+'**kwargs')
        self.write_line(ind+'):')
        self.write_line(2*ind+'super().__init__()')
        size1 = self.arch_dict["in_shape"][1]
        size2 = self.arch_dict["in_shape"][2]
        last_ch = 'num_channels'
        for u_idx, unit in enumerate(self.arch_dict['width_list']):
            for l_idx, width in enumerate(unit):
                kernel = self.arch_dict['kernel_list'][u_idx][l_idx]
                if kernel == 5:
                    pad = 2
                elif kernel == 3:
                    pad = 1
                elif kernel == 1:
                    pad = 0
                else:
                    print('Supported kernels are 5, 3, or 1.')
                    raise ValueError
                self.append_line(2*ind)
                if l_idx == 0 and u_idx != 0:
                    size1 = size1//2
                    if size1 < 1:
                        size1 = 1
                    if self.arch_dict['type'].lower() == 'conv2d':
                        size2 = size2//2
                        if size2 < 1:
                            size2 = 1
                    self.append_line(f'self.conv{u_idx+1}_{l_idx+1} = {self.pool_layer}(' +
                                     f'{last_ch}, {width}, {kernel}, stride=1, ' +
                                     f'padding={pad}, bias=bias, ')
                else:
                    self.append_line(f'self.conv{u_idx+1}_{l_idx+1} = {self.layer}(' +
                                     f'{last_ch}, {width}, {kernel}, stride=1, ' +
                                     f'padding={pad}, bias=bias, ')
                if self.arch_dict['bn']:
                    self.append_line('batchnorm="NoAffine", ')
                self.write_line('**kwargs)')
                last_ch = str(width)
        fc_in = int(size1 * size2 * width)  # pylint: disable=undefined-loop-variable
        self.write_line(2*ind+f'self.fc = ai8x.Linear({fc_in}, num_classes, bias=bias,' +
                        ' wide=True, **kwargs)')
        self.write_line('')
        self.write_line(ind+'def forward(self, x):  # pylint: disable=arguments-differ')
        for u_idx, unit in enumerate(self.arch_dict['width_list']):
            for l_idx, width in enumerate(unit):
                self.write_line(2*ind+f'x = self.conv{u_idx+1}_{l_idx+1}(x)')
        self.write_line(2*ind+'x = x.view(x.size(0), -1)')
        self.write_line(2*ind+'x = self.fc(x)')
        self.write_line(2*ind+'return x')
        self.write_line('')
        self.write_line('')
        self.write_line(f'def {self.model_name.lower()}(pretrained=False, **kwargs):')
        self.write_line(ind+'assert not pretrained')
        self.write_line(ind+f'return {self.model_name.upper()}(**kwargs)')
        self.write_line('')
        self.write_line('')
        self.write_line('models = [')
        self.write_line(ind+'{')
        self.write_line(2*ind+f"'name': '{self.model_name.lower()}',")
        self.write_line(2*ind+"'min_input': 1,")
        if self.arch_dict['type'].lower() == 'conv2d':
            self.write_line(2*ind+"'dim': 2,")
        elif self.arch_dict['type'].lower() == 'conv1d':
            self.write_line(2*ind+"'dim': 1,")
        self.write_line(ind+'},')
        self.write_line(']')
        self.write_line('')

    def close(self):
        """File closer"""
        self.f.close()


def main(arguments):
    """
    Main function
    """
    inp_path = arguments.input_filepath
    model_name_base = arguments.model_name
    with open(inp_path) as f:
        arch_dict_list = json.load(f)
    for idx, arch_dict in enumerate(arch_dict_list):
        model_name = model_name_base+'_'+str(idx+1)
        autogen = AutoGen(model_name, arch_dict)
        autogen.generate()
        autogen.close()


if __name__ == '__main__':
    try:
        my_parser = argparse.ArgumentParser()
        my_parser.add_argument('-i', '--input_filepath', type=str, required=True,
                               help='Input json file path')
        my_parser.add_argument('-n', '--model_name', type=str, required=True,
                               help='Output model name')
        args = my_parser.parse_args()
        main(args)
    except KeyboardInterrupt:
        print("\n-- KeyboardInterrupt --")
