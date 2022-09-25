import torch
import torch.onnx
import torch.nn
import torch.quantization


def main():
    ews_int8()

def ews_int8():
    class Model(torch.nn.Module):
        def __init__(self):
            super(Model, self).__init__()

        def forward(self, a, b):
            return torch.add(a, b)


    a = torch.tensor(list(range(16)))#, dtype=torch.int8)
    b = torch.tensor(list(range(16, 32)))#, dtype=torch.int8)

    model = Model()
    result = model(a, b)
    #torch.save(model, 'ews_net.pth')
    #qmodel = torch.quantization.quantize_dynamic(model, {torch.add}, dtype=torch.qint8)

    torch.onnx.export(model, (a, b), 'ews_net.onnx')


if __name__ == '__main__':
    main()
