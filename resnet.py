import caffe
from caffe import layers as L
from caffe import params as P


def conv_bn_scale_relu(bottom, num_output=64, kernel_size=3, stride=1, pad=0,dilation=1,use_gbs=False):
    conv = L.Convolution(bottom, num_output=num_output, kernel_size=kernel_size, stride=stride, pad=pad,
			 dilation=dilation,	
                         param=[dict(lr_mult=1, decay_mult=1)],
                         weight_filler=dict(type='msra', std=0.01),bias_term=False)
    conv_bn = L.BatchNorm(conv,use_global_stats=use_gbs, in_place=True,param=[dict(lr_mult=0),dict(lr_mult=0),dict(lr_mult=0)])
    conv_scale = L.Scale(conv, scale_param=dict(bias_term=True), in_place=True)
    conv_relu = L.ReLU(conv, in_place=True)

    return conv, conv_bn, conv_scale, conv_relu


def conv_bn_scale(bottom, num_output=64, kernel_size=3, stride=1, pad=0,dilation=1,use_gbs=False):
    conv = L.Convolution(bottom, num_output=num_output, kernel_size=kernel_size, stride=stride, pad=pad,
                         dilation=dilation,param=[dict(lr_mult=1, decay_mult=1)],
                         weight_filler=dict(type='msra', std=0.01),
                         bias_term=False)
    conv_bn = L.BatchNorm(conv, use_global_stats=use_gbs, in_place=True,
	param=[dict(lr_mult=0),dict(lr_mult=0),dict(lr_mult=0)])
    conv_scale = L.Scale(conv, scale_param=dict(bias_term=True), in_place=True)

    return conv, conv_bn, conv_scale


def eltwize_relu(bottom1, bottom2):
    residual_eltwise = L.Eltwise(bottom1, bottom2, eltwise_param=dict(operation=1))
    residual_eltwise_relu = L.ReLU(residual_eltwise, in_place=True)

    return residual_eltwise, residual_eltwise_relu


def residual_branch(bottom, base_output=64,dilation=1,use_gbs=False):
    """
    input:4*base_output x n x n
    output:4*base_output x n x n
    :param base_output: base num_output of branch2
    :param bottom: bottom layer
    :return: layers
    """
    branch2a, branch2a_bn, branch2a_scale, branch2a_relu = \
        conv_bn_scale_relu(bottom, num_output=base_output, kernel_size=1,use_gbs=use_gbs)  # base_output x n x n
    branch2b, branch2b_bn, branch2b_scale, branch2b_relu = \
        conv_bn_scale_relu(branch2a, num_output=base_output, kernel_size=3, pad=dilation,dilation=dilation,
		use_gbs=use_gbs)  # base_output x n x n
    branch2c, branch2c_bn, branch2c_scale = \
        conv_bn_scale(branch2b, num_output=4 * base_output, kernel_size=1,use_gbs=use_gbs)  # 4*base_output x n x n

    residual, residual_relu = \
        eltwize_relu(bottom, branch2c)  # 4*base_output x n x n

    return branch2a, branch2a_bn, branch2a_scale, branch2a_relu, branch2b, branch2b_bn, branch2b_scale, branch2b_relu, \
           branch2c, branch2c_bn, branch2c_scale, residual, residual_relu


def residual_branch_shortcut(bottom, stride=2, base_output=64,dilation=1,use_gbs=False):
    """

    :param stride: stride
    :param base_output: base num_output of branch2
    :param bottom: bottom layer
    :return: layers
    """
    branch1, branch1_bn, branch1_scale = \
        conv_bn_scale(bottom, num_output=4 * base_output, kernel_size=1, stride=stride)

    branch2a, branch2a_bn, branch2a_scale, branch2a_relu = \
        conv_bn_scale_relu(bottom, num_output=base_output, kernel_size=1, stride=stride,use_gbs=use_gbs)
    branch2b, branch2b_bn, branch2b_scale, branch2b_relu = \
        conv_bn_scale_relu(branch2a, num_output=base_output, kernel_size=3, pad=dilation,dilation=dilation,
		use_gbs=use_gbs)
    branch2c, branch2c_bn, branch2c_scale = \
        conv_bn_scale(branch2b, num_output=4 * base_output, kernel_size=1,use_gbs=use_gbs)

    residual, residual_relu = \
        eltwize_relu(branch1, branch2c)  # 4*base_output x n x n

    return branch1, branch1_bn, branch1_scale, branch2a, branch2a_bn, branch2a_scale, branch2a_relu, branch2b, \
           branch2b_bn, branch2b_scale, branch2b_relu, branch2c, branch2c_bn, branch2c_scale, residual, residual_relu


branch_shortcut_string = 'n.res(stage)a_branch1, n.res(stage)a_branch1_bn, n.res(stage)a_branch1_scale, \
        n.res(stage)a_branch2a, n.res(stage)a_branch2a_bn, n.res(stage)a_branch2a_scale, n.res(stage)a_branch2a_relu, \
        n.res(stage)a_branch2b, n.res(stage)a_branch2b_bn, n.res(stage)a_branch2b_scale, n.res(stage)a_branch2b_relu, \
        n.res(stage)a_branch2c, n.res(stage)a_branch2c_bn, n.res(stage)a_branch2c_scale, n.res(stage)a, n.res(stage)a_relu = \
            residual_branch_shortcut((bottom), stride=(stride), base_output=(num),use_gbs=(use_gbs),dilation=(dilation))'

branch_string = 'n.res(stage)b(order)_branch2a, n.res(stage)b(order)_branch2a_bn, n.res(stage)b(order)_branch2a_scale, \
        n.res(stage)b(order)_branch2a_relu, n.res(stage)b(order)_branch2b, n.res(stage)b(order)_branch2b_bn, \
        n.res(stage)b(order)_branch2b_scale, n.res(stage)b(order)_branch2b_relu, n.res(stage)b(order)_branch2c, \
        n.res(stage)b(order)_branch2c_bn, n.res(stage)b(order)_branch2c_scale, n.res(stage)b(order), n.res(stage)b(order)_relu = \
            residual_branch((bottom), base_output=(num),use_gbs=(use_gbs),dilation=(dilation))'
def classifier(bottom,dilation=6,classifier_num=2): 
	return L.Convolution(bottom, 
	      param=[dict(lr_mult=1, decay_mult=1), dict(lr_mult=2, decay_mult=0)],
	      convolution_param=dict(num_output=classifier_num,
	      kernel_size=3,pad=dilation,dilation=dilation,weight_filler=
	      dict(type="gaussian",std=0.01),	
	      bias_filler=dict(type='constant', value=0)))


def classifier_whole(bottom,classifier_num=2): 
	return L.InnerProduct(bottom, 
	      param=[dict(lr_mult=1, decay_mult=1), dict(lr_mult=2, decay_mult=0)],
	      inner_product_param=dict(num_output=classifier_num,
	      weight_filler=
	      dict(type="gaussian",std=0.01),	
	      bias_filler=dict(type='constant', value=0)))


class ResNet(object):
    def __init__(self,num_output=2):
        self.train_data = "/home/ubuntu/exper/voc12/list/train.txt"
        self.test_data = "/home/ubuntu/exper/voc12/list/val.txt"
        self.classifier_num = num_output
	self.new_height = 90
	self.new_width = 90
	self.crop = 81
    def resnet_layers_proto(self, batch_size, phase='TRAIN', stages=(3, 3, 3, 3),num_base=64): #3,4,6,3
        """

        :param batch_size: the batch_size of train and test phase
        :param phase: TRAIN or TEST
        :param stages: the num of layers = 2 + 3*sum(stages), layers would better be chosen from [50, 101, 152]
                       {every stage is composed of 1 residual_branch_shortcut module and stage[i]-1 residual_branch
                       modules, each module consists of 3 conv layers}
                        (3, 4, 6, 3) for 50 layers; (3, 4, 23, 3) for 101 layers; (3, 8, 36, 3) for 152 layers
        """
        n = caffe.NetSpec()
        if phase == 'TRAIN':
            source_data = self.train_data
            mirror = False
	    use_gbs = False	
        else:
            source_data = self.test_data
            mirror = False
            use_gbs = True
        n.data, n.label = L.ImageSegData(ntop=2,
                             	image_data_param= dict(source=source_data,batch_size=batch_size,
				is_color=False,new_height=self.new_height,new_width=self.new_width,
				root_folder="/home/ubuntu/ultrasound/raw/train",
				label_type=P.ImageData.PIXEL,shuffle=True),
				transform_param=dict(crop_size=self.crop, mirror=mirror)) # mean_value=[100]

        n.conv1, n.conv1_bn, n.conv1_scale, n.conv1_relu = \
            conv_bn_scale_relu(n.data, num_output=num_base, kernel_size=7, stride=2, pad=3)  # 64x112x112
        n.pool1 = L.Pooling(n.conv1, kernel_size=3, stride=2, pad=1,pool=P.Pooling.MAX)  

        for num in xrange(len(stages)):  # num = 0, 1, 2, 3
            for i in xrange(stages[num]):
                if i == 0:
                    stage_string = branch_shortcut_string
                    bottom_string = ['n.pool1', 'n.res2b%s' % str(stages[0] - 1), 'n.res3b%s' % str(stages[1] - 1),
                                     'n.res4b%s' % str(stages[2] - 1)][num]
                else:
                    stage_string = branch_string
                    if i == 1:
                        bottom_string = 'n.res%sa' % str(num + 2)
                    else:
                        bottom_string = 'n.res%sb%s' % (str(num + 2), str(i - 1))
		stride = 1
		dilation = 1
		if num == 1:
			stride = 2
		if num>1:
			#dilation = (num-1)*2
			dilation = 1
                exec (stage_string.replace('(stage)', str(num + 2)).replace('(bottom)', bottom_string).
                      replace('(num)', str(2 ** num * num_base)).replace('(order)', str(i)).
                      replace('(stride)', str(stride)).replace('(dilation)',str(dilation)))
	
	exec 'n.avg_pool = L.Pooling((bottom), kernel_size=11, stride=11, pad=0,pool=P.Pooling.AVE)'.\
		replace('(bottom)', 'n.res5b%s' % str(stages[3] - 1))
	n.classifier_whole = classifier_whole(n.avg_pool)
	n.label_binary = L.Pooling(n.label, kernel_size=self.crop,stride=self.crop, pad=0,pool=P.Pooling.MAX)  # 64x56x56
	n.label_whole =L.Flatten(n.label_binary)
	n.loss = L.SoftmaxWithLoss(n.classifier_whole, n.label_whole,loss_param=dict(ignore_label=255))	
	n.accuracy = L.Accuracy(n.classifier_whole,n.label_whole,accuracy_param=dict(ignore_label=255))	
 	
        #exec 'n.classifier = classifier((bottom), dilation=6)'.\
        #    replace('(bottom)', 'n.res5b%s' % str(stages[3] - 1))
	#n.label_shrink=L.Interp(n.label,interp_param=dict(shrink_factor=8,pad_beg=0,pad_end=0))
        #n.loss = L.SoftmaxWithLoss(n.classifier, n.label_shrink,loss_param=dict(ignore_label=255))
        #n.accuracy = L.SegAccuracy(n.classifier, n.label_shrink,
        #                                 seg_accuracy_param=dict(ignore_label=255))

        return n.to_proto()
