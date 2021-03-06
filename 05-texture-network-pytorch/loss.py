import torch
import torch.nn as nn


class EarthMoveDistanceLayer(nn.Module):
    def __init__(self, size_average=True):
        super(EarthMoveDistanceLayer, self).__init__()
        self.size_average = size_average

    def forward(self, data):
        generated_feature, target_feature = data
        assert(generated_feature.shape == target_feature.shape)
        B, C, H, W = generated_feature.shape
        generated = generated_feature.view(B, -1)
        target = target_feature.view(B, -1)
        generated = torch.sort(generated, dim=1)[0]
        target = torch.sort(target, dim=1)[0]
        loss = ((generated - target)**2).sum(dim=1)
        if self.size_average:
            loss = loss.mean(dim=0)
        else:
            loss = loss.sum(dim=0)
        return loss


class GramMatrix(nn.Module):
    def __init__(self, size_average=True):
        super(GramMatrix, self).__init__()
        self.size_average = size_average

    def forward(self, feature_map):
        B, C, H, W = feature_map.shape
        flatten_tensor = feature_map.view(B, C, -1)
        gram_matrix = torch.bmm(flatten_tensor, flatten_tensor.permute(0, 2, 1))
        if self.size_average:
            gram_matrix = gram_matrix.mean(dim=0)
        else:
            gram_matrix = gram_matrix.sum(dim=0)
        return gram_matrix


class GramLossLayer(nn.Module):
    def __init__(self, size_average=True):
        super(GramLossLayer, self).__init__()
        self.size_average = size_average
        self.handle_gram = GramMatrix(size_average=size_average)

    def forward(self, data):
        generated_feature, target_feature = data
        assert(generated_feature.shape == target_feature.shape)
        B, C, H, W = generated_feature.shape

        generated = self.handle_gram(generated_feature)
        target = self.handle_gram(target_feature)

        factor = 1.0/(4*(H**2)*(W**2))
        loss = factor * (((generated - target)**2).sum())
        return loss


class GramLoss(nn.Module):
    dist_type_dict = {'gram': GramLossLayer, 'earth_move': EarthMoveDistanceLayer}

    def __init__(self, weight=None, dist_type=None, size_average=True):
        super(GramLoss, self).__init__()
        assert(weight is not None)
        assert(dist_type in self.dist_type_dict.keys())
        self.weight = weight
        self.handle_layer = self.dist_type_dict[dist_type](size_average=size_average)


    def forward(self, feature_list):
        assert(len(self.weight) == len(feature_list))
        handle = map(self.handle_layer, feature_list)
        layer_loss = list(handle)
        handle = map(lambda t: t[0]*t[1], zip(self.weight, layer_loss))
        loss = sum(list(handle))
        return loss


def test():
    inputs = torch.randn(4, 3, 32, 32).requires_grad_(True)
    target = torch.randn(4, 3, 32, 32).requires_grad_(True)
    loss = GramLoss([1.0])
    l1 = loss([[inputs, target]])
    print(l1.item())


if __name__ == "__main__":
    test()
