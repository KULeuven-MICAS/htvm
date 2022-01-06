"""
 Author(s): Wiebe Van Ranst, Copyright EAVISE

 Based on:
    https://docs.tvm.ai/tutorials/frontend/deploy_ssd_gluoncv.html
"""
import tvm

import matplotlib.pyplot as plt
#from tvm.relay.testing.config import ctx_list
from tvm.contrib.download import download_testdata
from gluoncv import model_zoo, data, utils
from tvm.contrib import graph_runtime
from tvm import relay
import tvm.contrib.utils


class Model(object):
    """ Class containing the model """

    def __init__(self):
        self.model_name = 'ssd_512_mobilenet1.0_coco'
        self.dshape = (1, 3, 512, 512) # Square rgb image
        self.target = 'llvm'

    def download_test_data(self):
        im_fname = download_testdata('https://raw.githubusercontent.com/dmlc/web-data/master/gluoncv/detection/street_small.jpg',
                                     'street_small.jpg', module='data')
        x, img = data.transforms.presets.ssd.load_test(im_fname, short=512)

        return x, img

    def compile_model(self):
        self.block = model_zoo.get_model(self.model_name, pretrained=True)
        mod, params = relay.frontend.from_mxnet(self.block, {'data': self.dshape})
        with relay.build_config(opt_level=3):
            graph, lib, params = relay.build(mod, target=self.target, params=params)

        return graph, lib, params

    def run(self):
        x, img = self.download_test_data()
        graph, lib, params = self.compile_model()

        self.model = graph_runtime.create(graph, lib)#, self.remote.cpu())
        tvm_input = tvm.nd.array(x.asnumpy())#, ctx=self.remote.cpu())
        self.model.set_input('data', tvm_input)
        self.model.set_input(**params)


        time_f = lib.time_evaluator(lib.entry_name, self.remote.cpu(), number=10)
        #time_f.set_input('data', tvm_input)
        #time_f.set_input(**params)
        print(params)
        cost = time_f(tvm_input, *params)

        print(f'cost: {cost}')

        self.model.run()

        class_ids, scores, bounding_boxs = self.model.get_output(0), self.model.get_output(1), self.model.get_output(2)

        ax = utils.viz.plot_bbox(img, bounding_boxs.asnumpy()[0], scores.asnumpy()[0],
                                 class_ids.asnumpy()[0], class_names=self.block.classes)
        plt.show()


def main():
    model = Model()
    model.run()


if __name__ == '__main__':
    main()
