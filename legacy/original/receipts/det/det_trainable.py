"""Indra's line detector (det.onnx, LineNet) rebuilt in PyTorch, with its weights and its exact output.

The export folded batch norm into the convolutions, so this net has none. Trained weights go back into a copy of the
original ONNX graph, keeping every other node as shipped. verify_parity compares both.
"""
import onnx
import torch
from onnx import numpy_helper
from torch import nn
from torch.nn import functional as F

DIST_SCALE = 8.0  # the Mul constant after Softplus in the graph


def conv_nodes(graph):
    return [n for n in graph.node if n.op_type == "Conv"]


class DetNet(nn.Module):
    """Convolutions in the graph's order; forward follows the graph's data flow."""

    def __init__(self, path):
        super().__init__()
        graph = onnx.load(str(path)).graph
        init = {t.name: numpy_helper.to_array(t) for t in graph.initializer}
        self.convs = nn.ModuleList()
        for node in conv_nodes(graph):
            attrs = {a.name: onnx.helper.get_attribute_value(a) for a in node.attribute}
            w, b = init[node.input[1]].copy(), init[node.input[2]].copy()
            groups = attrs["group"]
            c = nn.Conv2d(w.shape[1] * groups, w.shape[0], tuple(attrs["kernel_shape"]), tuple(attrs["strides"]), attrs["pads"][0], groups=groups)
            c.weight.data = torch.from_numpy(w)
            c.bias.data = torch.from_numpy(b)
            self.convs.append(c)
        assert len(self.convs) == 39, len(self.convs)

    def run(self, x, *indices):
        """Conv then HardSwish for each index; the last of a residual body has no activation (see res)."""
        for i in indices:
            x = F.hardswish(self.convs[i](x))
        return x

    def res(self, x, i):
        return x + self.convs[i + 2](self.run(x, i, i + 1))

    def forward(self, x):
        stem = self.run(x, 0)
        s0 = self.res(self.run(stem, 1, 2), 3)
        s1 = self.res(self.res(self.run(s0, 6, 7), 8), 11)
        s2 = self.res(self.res(self.run(s1, 14, 15), 16), 19)
        s3 = self.res(self.res(self.run(s2, 22, 23), 24), 27)
        p = self.run(s3, 30)
        p = self.run(s2, 31) + F.interpolate(p, size=s2.shape[2:], mode="nearest")
        p = self.run(s1, 32) + F.interpolate(p, size=s1.shape[2:], mode="nearest")
        p = self.run(s0, 33) + F.interpolate(p, size=s0.shape[2:], mode="nearest")
        p = self.run(p, 34)
        fine = F.pixel_unshuffle(self.run(stem, 35), 2)
        out = F.pixel_shuffle(self.convs[38](self.run(torch.cat([p, fine], 1), 36, 37)), 2)
        return torch.cat([out[:, :1], F.softplus(out[:, 1:]) * DIST_SCALE], 1)

    def write_onnx(self, template, path):
        model = onnx.load(str(template))
        init = {t.name: t for t in model.graph.initializer}
        for node, conv in zip(conv_nodes(model.graph), self.convs):
            for name, param in zip(node.input[1:3], (conv.weight, conv.bias)):
                init[name].CopyFrom(numpy_helper.from_array(param.detach().cpu().numpy(), name))
        onnx.save(model, str(path))


@torch.inference_mode()
def verify_parity(path, model=None, atol=2e-3):
    """Largest output difference between the ONNX graph and the PyTorch net, on noise and on a text-like page."""
    import numpy as np
    import onnxruntime as ort

    model = (model or DetNet(path)).cpu().eval()
    sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    rs = np.random.default_rng(1)
    worst = 0.0
    bars = np.zeros((1, 1, 512, 320), np.float32)
    bars[:, :, 40:60, 30:290] = 1
    bars[:, :, 120:132, 60:250] = .7
    for x in (rs.random((1, 1, 320, 224), np.float32), bars, rs.random((1, 1, 1024, 576), np.float32) * .2):
        a = sess.run(None, {"x": x})[0]
        b = model(torch.from_numpy(x)).numpy()
        assert a.shape == b.shape, (a.shape, b.shape)
        worst = max(worst, float(np.abs(a - b).max()))
    assert worst < atol, worst
    return {"max_abs": worst}
