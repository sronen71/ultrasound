from caffe.proto import caffe_pb2
import resnet


def save_proto(proto, prototxt):
    with open(prototxt, 'w') as f:
        f.write(str(proto))


if __name__ == '__main__':
    model = resnet.ResNet()
    num_base = 32
    num_batch = 64
    train_proto = model.resnet_layers_proto(num_batch,num_base=num_base)
    test_proto = model.resnet_layers_proto(num_batch,num_base=num_base,phase='TEST')

    save_proto(train_proto, 'train.prototxt')
    save_proto(test_proto, 'test.prototxt')
