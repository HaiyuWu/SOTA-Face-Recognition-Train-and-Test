from torch.nn import Module


class ModelWrapper(Module):
    def __init__(self, model, head):
        super(ModelWrapper, self).__init__()
        self.model = model
        self.head = head

    def forward(self, x, labels=None):
        x = self.model(x)

        # used for recognition
        if labels is not None:
            x = self.head(x, labels)
        else:
            x = self.head(x)

        return x
