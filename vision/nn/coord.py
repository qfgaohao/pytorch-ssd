import torch


class Coord(torch.nn.Module):
    def __init__(self, h, w):
        super(Coord, self).__init__()
        x = torch.arange(w).expand(h, -1)
        y = torch.arange(h).unsqueeze(1).expand(-1, w)
        r = torch.sqrt(
            torch.pow(x - w/2, 2) + torch.pow(y - h/2, 2)
        )
        self.coord = torch.stack([x, y, r]).unsqueeze(0)

    def forward(self, input):
        x = torch.cat([input, self.coord.expand(input.size(0), -1, -1, -1)], 1)
        return x