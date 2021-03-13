
def get_model(args):
    model_name = args['model_name']
    
    if 'num_slices' in args:
        num_slices = args['num_slices'] 
    if 'pretrained' in args:
        pretrained = args['pretrained']
    
    
    if model_name == 'S3DSegPan3D_2D':
        from model.S3DSegPan3D_2D import S3DSegPan3D_2D
        model_parameters = model_parameters = {'pretrained' : pretrained, 
                                                'out_sigmoid': False,
                                                'out_ch': 2,
                                                'multi_output' : True,
                                                }
        model = S3DSegPan3D_2D(pretrained=model_parameters['pretrained'], 
                               out_sigmoid=model_parameters['out_sigmoid'], 
                               out_ch=model_parameters['out_ch'], 
                               multi_output=model_parameters['multi_output'])
    if model_name == 'S3DSegPan3D_3D':
        from model.S3DSegPan3D_3D import S3DSegPan3D_3D
        model_parameters = {'pretrained' : pretrained, 
                                                'out_sigmoid': False,
                                                'out_ch': 2,
                                                'multi_output' : True,
                                                }
        model = S3DSegPan3D_3D(pretrained=model_parameters['pretrained'], 
                               out_sigmoid=model_parameters['out_sigmoid'], 
                               out_ch=model_parameters['out_ch'], 
                               multi_output=model_parameters['multi_output'], 
                               num_slices= num_slices)
    if model_name == 'S3D_EncDec_3D_3D':
        from model.S3D_EncDec_3D_3D import S3D_EncDec_3D_3D
        model_parameters = {'pretrained' : pretrained, 
                            'out_sigmoid': False,
                            'out_ch': 2}
        model = S3D_EncDec_3D_3D(pretrained=model_parameters['pretrained'], 
                               out_sigmoid=model_parameters['out_sigmoid'], 
                               out_ch=model_parameters['out_ch'],
                               num_slices= num_slices)
    if model_name == 'MobileNetSegPan3D_2D':
        from model.MobileNetSegPan3D_2D import MobileNetSegPan3D_2D
        model_parameters = {'pretrained' : pretrained, 
                                                'out_sigmoid': False,
                                                'out_ch': 2,
                                                'multi_output' : True,
                                                }
        model = MobileNetSegPan3D_2D(pretrained=model_parameters['pretrained'], 
                                    out_sigmoid=model_parameters['out_sigmoid'], 
                                    out_ch=model_parameters['out_ch'], 
                                    multi_output=model_parameters['multi_output'])
    if model_name == 'MobilNetSegPan3D_3D':
        from model.MobilNetSegPan3D_3D import MobilNetSegPan3D_3D
        model_parameters = {'pretrained' : pretrained, 
                                                'out_sigmoid': False,
                                                'out_ch': 2,
                                                'multi_output' : True,
                                                }
        model = MobilNetSegPan3D_3D(pretrained=model_parameters['pretrained'], 
                                    out_sigmoid=model_parameters['out_sigmoid'], 
                                    out_ch=model_parameters['out_ch'], 
                                    multi_output=model_parameters['multi_output'], 
                                    num_slices= num_slices)
    if model_name == 'MobileNet_EncDec_3D_3D':
        from model.MobileNet_EncDec_3D_3D import MobileNet_EncDec_3D_3D
        model_parameters = {'pretrained' : pretrained, 
                                                'out_sigmoid': False,
                                                'out_ch': 2
                                                }
        model = MobileNet_EncDec_3D_3D(pretrained=model_parameters['pretrained'], 
                                    out_sigmoid=model_parameters['out_sigmoid'], 
                                    out_ch=model_parameters['out_ch'], 
                                    num_slices= num_slices)
    if model_name == 'SegPan3D_VGGBackBone':
        from model.SegPan3D_VGGBackbone import SegPan3D_VGGBackBone
        model_parameters = {'pretrained' : pretrained, 
                                                'out_sigmoid': False,
                                                'out_ch': 2,
                                                'multi_output' : True,
                                                'arch': 'vgg16_bn'
                                                }
        model = SegPan3D_VGGBackBone(pretrained=model_parameters['pretrained'],
                                     out_sigmoid=model_parameters['out_sigmoid'], 
                                     out_ch=model_parameters['out_ch'], 
                                     multi_output=model_parameters['multi_output'])
    if model_name == 'UNet3D':
        from monai.networks.nets import UNet
        model_parameters = {
                            'dimensions': 3,
                            'in_channels': 1,
                            'out_channels': 2,
                            'channels': (16,32,64,128,256),
                            'strides' : (2,2,2,2),
                            'num_res_units': 2
                            }
        model = UNet(
                    dimensions=model_parameters['dimensions'],
                    in_channels=model_parameters['in_channels'],
                    out_channels=model_parameters['out_channels'],
                    channels=model_parameters['channels'],
                    strides = model_parameters['strides'],
                    num_res_units=model_parameters['num_res_units']
                    )
    if model_name == 'SegResNet':
        from monai.networks.nets import SegResNet
        model_parameters = {
                            'spatial_dims': 3,
                            'init_filters': 8,
                            'in_channels': 1,
                            'out_channels': 2
                            }
        model = SegResNet(spatial_dims = model_parameters['spatial_dims'],
                          init_filters = model_parameters['init_filters'],
                          in_channels = model_parameters['in_channels'],
                          out_channels = model_parameters['out_channels']
                          )
    if model_name == 'VNet':
        from monai.networks.nets import VNet
        model_parameters = {
                            'spatial_dims': 3,
                            'in_channels': 1,
                            'out_channels': 2, 
                            'act': ("elu", {"inplace": True})
                            }
        model = VNet(spatial_dims = model_parameters['spatial_dims'],
                          in_channels = model_parameters['in_channels'],
                          out_channels = model_parameters['out_channels'],
                          act = model_parameters['act']
                          )
    if model_name == 'AHNet':
        from monai.networks.nets import AHNet
        model_parameters = {
                            'layers': (3,4,6,3),
                            'spatial_dims': 3,
                            'in_channels': 1,
                            'out_channels': 2, 
                            'psp_block_num' : 4, 
                            'upsample_mode' : 'transpose', 
                            'pretrained' : True
                            }
        model = AHNet(layers = model_parameters['layers'],
                      spatial_dims = model_parameters['spatial_dims'],
                      in_channels = model_parameters['in_channels'],
                      out_channels = model_parameters['out_channels'],
                      psp_block_num = model_parameters['psp_block_num'],
                      upsample_mode = model_parameters['upsample_mode'],
                      pretrained = model_parameters['pretrained']
                          )
    return {'model': model, 'model_params': model_parameters}















